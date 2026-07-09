from __future__ import annotations

import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import export_avl
from export_avl import (
    build_gmsh_cgns_command,
    convert_msh_to_cgns_with_python_api,
    export_avl_cgns_via_gmsh,
    resolve_gmsh_executable,
    strip_prefixed_cgns_names,
)
from export_gmsh import (
    FLUID_BOUNDARY_EXPORT_NAMES,
    FLUID_VOLUME_EXPORT_NAME,
    FULL_BOUNDARY_EXPORT_NAMES,
    FULL_VOLUME_EXPORT_NAMES,
    export_gmsh_discrete,
    reorder_hex_for_export,
)
from extrude import build_mesh3d
from mesh2d import build_mesh2d
from params import HelixMeshParams
from topology import build_topology


def _build_default_3d():
    params = HelixMeshParams(core_mode="annulus_tube", r_inner_tube=1.2e-3)
    points, patches = build_topology(params)
    mesh2d = build_mesh2d(points, patches)
    mesh3d = build_mesh3d(mesh2d, params)
    return params, points, patches, mesh2d, mesh3d


@pytest.fixture
def workspace_tmpdir() -> Path:
    root = Path(__file__).resolve().parents[1] / f".tmp-test-export-avl-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _skip_if_gmsh_unavailable() -> str:
    gmsh_path = resolve_gmsh_executable("gmsh")
    if gmsh_path is None:
        pytest.skip("gmsh executable is not available on PATH in this environment.")
    return gmsh_path


def test_python_api_converter_raises_without_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(export_avl, "resolve_gmsh_python_module", lambda: None)

    with pytest.raises(ModuleNotFoundError, match="Python gmsh module"):
        convert_msh_to_cgns_with_python_api("input.msh", "output.cgns")


def test_strip_prefixed_cgns_names_removes_zone_prefixes(workspace_tmpdir: Path) -> None:
    outfile = workspace_tmpdir / "prefixed.cgns"
    root_name = outfile.name

    with h5py.File(outfile, "w") as handle:
        base = handle.create_group(root_name)
        fam = base.create_group("V_1_DOM_GasVolume")
        fam.attrs["name"] = b"V_1_DOM_GasVolume"
        fam.create_group("DOM_GasVolume")

        part = base.create_group("prefixed_Part0")
        part_elem = part.create_group("8_V_1_DOM_GasVolume")
        part_elem.attrs["name"] = b"8_V_1_DOM_GasVolume"

        zone_bc = part.create_group("ZoneBC")
        bc = zone_bc.create_group("S_1_BND_I_Inlet_Gas")
        bc.attrs["name"] = b"S_1_BND_I_Inlet_Gas"
        family_name = bc.create_group("FamilyName")
        family_name.create_dataset(" data", data=list(b"S_1_BND_I_Inlet_Gas"))

    strip_prefixed_cgns_names(outfile)

    with h5py.File(outfile, "r") as handle:
        base = handle[root_name]
        assert "V_1_DOM_GasVolume" in base
        assert base["V_1_DOM_GasVolume"].attrs["name"] in ("V_1_DOM_GasVolume", b"V_1_DOM_GasVolume")

        part = base["prefixed_Part0"]
        assert "DOM_GasVolume" in part
        assert part["DOM_GasVolume"].attrs["name"] == b"DOM_GasVolume"

        zone_bc = part["ZoneBC"]
        assert "BND_I_Inlet_Gas" in zone_bc
        assert zone_bc["BND_I_Inlet_Gas"].attrs["name"] == b"BND_I_Inlet_Gas"
        assert bytes(zone_bc["BND_I_Inlet_Gas"]["FamilyName"][" data"][()].tolist()) == b"S_1_BND_I_Inlet_Gas"


def test_export_avl_prefers_python_api_before_cli(
    monkeypatch: pytest.MonkeyPatch,
    workspace_tmpdir: Path,
) -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    outfile = workspace_tmpdir / "export_avl_python_first.cgns"

    def fake_python_converter(intermediate_msh: str | Path, outfile_path: str | Path, save_all: bool = False) -> None:
        Path(outfile_path).write_text("python-api-cgns", encoding="utf-8")

    def fail_if_cli_used(*args, **kwargs):
        raise AssertionError("CLI fallback should not be used when Python conversion succeeds.")

    monkeypatch.setattr(export_avl, "write_cgns_directly_with_python_api", lambda *args, **kwargs: fake_python_converter("", outfile))
    monkeypatch.setattr(export_avl, "convert_msh_to_cgns_with_gmsh_executable", fail_if_cli_used)

    export_avl_cgns_via_gmsh(
        mesh3d,
        params,
        str(outfile),
        gmsh_exe="definitely_missing_gmsh_executable",
        keep_intermediate=True,
    )

    assert outfile.exists()
    assert outfile.read_text(encoding="utf-8") == "python-api-cgns"


def test_export_avl_falls_back_to_cli_when_python_api_fails(
    monkeypatch: pytest.MonkeyPatch,
    workspace_tmpdir: Path,
) -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    outfile = workspace_tmpdir / "export_avl_cli_fallback.cgns"

    monkeypatch.setattr(
        export_avl,
        "write_cgns_directly_with_python_api",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("python api failed")),
    )
    monkeypatch.setattr(export_avl, "resolve_gmsh_executable", lambda gmsh_exe="gmsh": "gmsh")

    def fake_cli_converter(intermediate_msh: str | Path, outfile_path: str | Path, gmsh_exe: str, save_all: bool = False):
        Path(outfile_path).write_text("cli-cgns", encoding="utf-8")
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(export_avl, "convert_msh_to_cgns_with_gmsh_executable", fake_cli_converter)

    export_avl_cgns_via_gmsh(mesh3d, params, str(outfile), gmsh_exe="gmsh", keep_intermediate=True)

    assert outfile.exists()
    assert outfile.read_text(encoding="utf-8") == "cli-cgns"


def test_export_avl_cgns_via_gmsh_writes_file(workspace_tmpdir: Path) -> None:
    gmsh_path = _skip_if_gmsh_unavailable()
    params, _, _, _, mesh3d = _build_default_3d()
    outfile = workspace_tmpdir / "export_avl.cgns"

    export_avl_cgns_via_gmsh(mesh3d, params, str(outfile), gmsh_exe=gmsh_path)

    assert outfile.exists()
    assert outfile.stat().st_size > 0


def test_export_avl_cgns_via_gmsh_accepts_hex_order(workspace_tmpdir: Path) -> None:
    gmsh_path = _skip_if_gmsh_unavailable()
    params, _, _, _, mesh3d = _build_default_3d()
    outfile = workspace_tmpdir / "export_avl_flip_i.cgns"

    export_avl_cgns_via_gmsh(mesh3d, params, str(outfile), gmsh_exe=gmsh_path, hex_order="flip_i")

    assert outfile.exists()
    assert outfile.stat().st_size > 0


def test_export_avl_default_hex_order_is_reverse(
    monkeypatch: pytest.MonkeyPatch,
    workspace_tmpdir: Path,
) -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    outfile = workspace_tmpdir / "export_avl_default_reverse.cgns"
    captured: dict[str, str] = {}

    def fake_python_writer(*args, **kwargs):
        captured["hex_order"] = kwargs["hex_order"]
        Path(args[2]).write_text("python-api-cgns", encoding="utf-8")
        Path(kwargs["intermediate_msh"]).write_text("intermediate", encoding="utf-8")

    monkeypatch.setattr(export_avl, "write_cgns_directly_with_python_api", fake_python_writer)
    monkeypatch.setattr(export_avl, "strip_prefixed_cgns_names", lambda outfile: None)

    export_avl_cgns_via_gmsh(mesh3d, params, str(outfile), keep_intermediate=True)

    assert captured["hex_order"] == "reverse"
    assert outfile.exists()


def test_export_avl_does_not_strip_prefixes_by_default(
    monkeypatch: pytest.MonkeyPatch,
    workspace_tmpdir: Path,
) -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    outfile = workspace_tmpdir / "export_avl_no_strip_default.cgns"
    intermediate = workspace_tmpdir / "export_avl_no_strip_default.msh"
    calls = {"strip": 0}

    def fake_python_writer(*args, **kwargs):
        Path(args[2]).write_text("python-api-cgns", encoding="utf-8")
        Path(kwargs["intermediate_msh"]).write_text("intermediate", encoding="utf-8")

    def fake_strip(_outfile):
        calls["strip"] += 1

    monkeypatch.setattr(export_avl, "write_cgns_directly_with_python_api", fake_python_writer)
    monkeypatch.setattr(export_avl, "strip_prefixed_cgns_names", fake_strip)

    export_avl_cgns_via_gmsh(
        mesh3d,
        params,
        str(outfile),
        keep_intermediate=True,
        intermediate_msh=str(intermediate),
    )

    assert outfile.exists()
    assert calls["strip"] == 0


def test_export_avl_can_strip_prefixes_when_requested(
    monkeypatch: pytest.MonkeyPatch,
    workspace_tmpdir: Path,
) -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    outfile = workspace_tmpdir / "export_avl_strip_requested.cgns"
    intermediate = workspace_tmpdir / "export_avl_strip_requested.msh"
    calls = {"strip": 0}

    def fake_python_writer(*args, **kwargs):
        Path(args[2]).write_text("python-api-cgns", encoding="utf-8")
        Path(kwargs["intermediate_msh"]).write_text("intermediate", encoding="utf-8")

    def fake_strip(_outfile):
        calls["strip"] += 1

    monkeypatch.setattr(export_avl, "write_cgns_directly_with_python_api", fake_python_writer)
    monkeypatch.setattr(export_avl, "strip_prefixed_cgns_names", fake_strip)

    export_avl_cgns_via_gmsh(
        mesh3d,
        params,
        str(outfile),
        keep_intermediate=True,
        intermediate_msh=str(intermediate),
        strip_cgns_prefixes=True,
    )

    assert outfile.exists()
    assert calls["strip"] == 1


def test_export_avl_cgns_keeps_intermediate(workspace_tmpdir: Path) -> None:
    gmsh_path = _skip_if_gmsh_unavailable()
    params, _, _, _, mesh3d = _build_default_3d()
    outfile = workspace_tmpdir / "export_avl_keep.cgns"
    intermediate = workspace_tmpdir / "export_avl_keep_intermediate.msh"

    export_avl_cgns_via_gmsh(
        mesh3d,
        params,
        str(outfile),
        gmsh_exe=gmsh_path,
        keep_intermediate=True,
        intermediate_msh=str(intermediate),
    )

    assert outfile.exists()
    assert intermediate.exists()
    assert intermediate.stat().st_size > 0


def test_export_avl_cgns_removes_intermediate_by_default(workspace_tmpdir: Path) -> None:
    gmsh_path = _skip_if_gmsh_unavailable()
    params, _, _, _, mesh3d = _build_default_3d()
    outfile = workspace_tmpdir / "export_avl_remove.cgns"
    intermediate = workspace_tmpdir / "export_avl_remove_intermediate.msh"

    export_avl_cgns_via_gmsh(
        mesh3d,
        params,
        str(outfile),
        gmsh_exe=gmsh_path,
        keep_intermediate=False,
        intermediate_msh=str(intermediate),
    )

    assert outfile.exists()
    assert not intermediate.exists()


def test_missing_gmsh_executable_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
    workspace_tmpdir: Path,
) -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    outfile = workspace_tmpdir / "export_avl_missing_gmsh.cgns"

    monkeypatch.setattr(
        export_avl,
        "write_cgns_directly_with_python_api",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("python api failed")),
    )
    monkeypatch.setattr(export_avl, "resolve_gmsh_executable", lambda gmsh_exe="gmsh": None)

    with pytest.raises(RuntimeError, match="no Gmsh executable was found for fallback"):
        export_avl_cgns_via_gmsh(mesh3d, params, str(outfile), gmsh_exe="definitely_missing_gmsh_executable")


def test_intermediate_gmsh_contains_multidomain_volumes(workspace_tmpdir: Path) -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    intermediate = workspace_tmpdir / "export_multidomain_intermediate.msh"

    export_gmsh_discrete(mesh3d, params, outfile=str(intermediate), fluid_only=False)

    text = intermediate.read_text(encoding="utf-8")
    assert FULL_VOLUME_EXPORT_NAMES["fluid"] in text
    assert FULL_VOLUME_EXPORT_NAMES["rib_solid"] in text


def test_intermediate_export_default_hex_order_matches_internal_order(workspace_tmpdir: Path) -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    intermediate = workspace_tmpdir / "export_default_order_intermediate.msh"

    export_gmsh_discrete(mesh3d, params, outfile=str(intermediate), fluid_only=False)

    first_hex = next(hex_element for hex_element in mesh3d.hexes if hex_element.kind == "fluid")
    assert reorder_hex_for_export(first_hex.node_ids, "default") == first_hex.node_ids
    assert intermediate.exists()
    assert intermediate.stat().st_size > 0


def test_intermediate_export_reverse_hex_order_is_available() -> None:
    node_ids = (10, 11, 12, 13, 14, 15, 16, 17)

    assert reorder_hex_for_export(node_ids, "reverse") == (13, 12, 11, 10, 17, 16, 15, 14)


def test_fluid_only_option_intermediate_contains_no_rib_solid_volume(workspace_tmpdir: Path) -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    intermediate = workspace_tmpdir / "export_fluid_only_intermediate.msh"

    export_gmsh_discrete(mesh3d, params, outfile=str(intermediate), fluid_only=True)

    text = intermediate.read_text(encoding="utf-8")
    assert FLUID_VOLUME_EXPORT_NAME in text
    assert FULL_VOLUME_EXPORT_NAMES["rib_solid"] not in text
    assert FLUID_BOUNDARY_EXPORT_NAMES["rib_wall"] in text


def test_gmsh_command_is_constructed_without_shell_true() -> None:
    command = build_gmsh_cgns_command("input.msh", "output.cgns", "gmsh", save_all=True)

    assert isinstance(command, list)
    assert command == ["gmsh", "input.msh", "-format", "cgns", "-o", "output.cgns", "-save_all"]


def test_previous_gmsh_export_still_works(workspace_tmpdir: Path) -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    outfile = workspace_tmpdir / "export_gmsh_regression.msh"

    export_gmsh_discrete(mesh3d, params, outfile=str(outfile))

    assert outfile.exists()
    assert outfile.stat().st_size > 0


def test_cli_accepts_hex_order_argument_when_gmsh_available(workspace_tmpdir: Path) -> None:
    gmsh_path = _skip_if_gmsh_unavailable()
    outfile = workspace_tmpdir / "cli_hex_order_flip_k.cgns"
    intermediate = workspace_tmpdir / "cli_hex_order_flip_k.msh"
    script_path = Path(__file__).resolve().parents[1] / "helix_mesh.py"
    params_path = Path(__file__).resolve().parents[1] / "parameters.json"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--params",
            str(params_path),
            "--export-avl-cgns",
            str(outfile),
            "--gmsh-exe",
            gmsh_path,
            "--hex-order",
            "flip_k",
            "--keep-intermediate",
            "--intermediate-msh",
            str(intermediate),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert outfile.exists()
    assert outfile.stat().st_size > 0
    assert intermediate.exists()
