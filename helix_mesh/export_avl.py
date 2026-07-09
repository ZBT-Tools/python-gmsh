from __future__ import annotations

import importlib
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
from export_gmsh import build_gmsh_export_data, export_gmsh_discrete, populate_gmsh_model
from extrude import Mesh3D
from params import HelixMeshParams


def resolve_gmsh_python_module():
    try:
        return importlib.import_module("gmsh")
    except ModuleNotFoundError:
        return None


def resolve_gmsh_executable(gmsh_exe: str = "gmsh") -> str | None:
    if Path(gmsh_exe).exists():
        return str(Path(gmsh_exe))
    return shutil.which(gmsh_exe)


def build_gmsh_cgns_command(
    intermediate_msh: str | Path,
    outfile: str | Path,
    gmsh_exe: str,
    save_all: bool = False,
) -> list[str]:
    command = [gmsh_exe, str(intermediate_msh), "-format", "cgns", "-o", str(outfile)]
    if save_all:
        command.append("-save_all")
    return command


def convert_msh_to_cgns_with_python_api(
    intermediate_msh: str | Path,
    outfile: str | Path,
    save_all: bool = False,
) -> None:
    gmsh = resolve_gmsh_python_module()
    if gmsh is None:
        raise ModuleNotFoundError("Python gmsh module is not available.")

    gmsh.initialize()
    try:
        if save_all:
            gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.open(str(intermediate_msh))
        gmsh.write(str(outfile))
    finally:
        gmsh.finalize()


def write_cgns_directly_with_python_api(
    mesh3d: Mesh3D,
    params: HelixMeshParams,
    outfile: str | Path,
    intermediate_msh: str | Path | None = None,
    export_solids: bool = True,
    save_all: bool = False,
    hex_order: str = "reverse",
) -> None:
    gmsh = resolve_gmsh_python_module()
    if gmsh is None:
        raise ModuleNotFoundError("Python gmsh module is not available.")

    export_data = build_gmsh_export_data(mesh3d, params, fluid_only=not export_solids)

    gmsh.initialize()
    try:
        if save_all:
            gmsh.option.setNumber("Mesh.SaveAll", 1)
        populate_gmsh_model(
            gmsh,
            mesh3d,
            params,
            export_data=export_data,
            fluid_only=not export_solids,
            hex_order=hex_order,
            model_name=Path(outfile).stem,
        )
        if intermediate_msh is not None:
            gmsh.write(str(intermediate_msh))
        gmsh.write(str(outfile))
    finally:
        gmsh.finalize()


def convert_msh_to_cgns_with_gmsh_executable(
    intermediate_msh: str | Path,
    outfile: str | Path,
    gmsh_exe: str,
    save_all: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = build_gmsh_cgns_command(
        intermediate_msh=intermediate_msh,
        outfile=outfile,
        gmsh_exe=gmsh_exe,
        save_all=save_all,
    )
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )


def _encode_cgns_string(value: str):
    return value.encode("ascii")


def _write_cgns_name_attr(obj, value: str) -> None:
    if "name" in obj.attrs:
        del obj.attrs["name"]
    obj.attrs["name"] = np.bytes_(value)


def _write_cgns_char_dataset(group, dataset_name: str, value: str) -> None:
    encoded = np.frombuffer(value.encode("ascii"), dtype=np.int8)
    if dataset_name in group:
        del group[dataset_name]
    group.create_dataset(dataset_name, data=encoded)


def _has_cgns_label(obj, expected: bytes) -> bool:
    return obj.attrs.get("label") == np.bytes_(expected)


def _strip_cgns_prefix(name: str) -> str:
    if name.startswith(("S_", "V_")):
        parts = name.split("_", 2)
        if len(parts) == 3:
            return parts[2]
    if name.startswith(("4_S_", "8_V_")):
        parts = name.split("_", 3)
        if len(parts) == 4:
            return parts[3]
    return name


def strip_prefixed_cgns_names(outfile: str | Path) -> None:
    outpath = Path(outfile)
    with h5py.File(outpath, "r+") as handle:
        base_name = outpath.name
        if base_name not in handle:
            raise RuntimeError(f"Unexpected CGNS root layout in {outpath}.")

        base = handle[base_name]

        part_group = None
        for name in base.keys():
            if name.endswith("_Part0"):
                part_group = base[name]
                break
        if part_group is None:
            raise RuntimeError(f"Could not locate CGNS part zone in {outpath}.")

        part_renames: dict[str, str] = {}
        for name in list(part_group.keys()):
            stripped = _strip_cgns_prefix(name)
            if stripped != name:
                part_renames[name] = stripped

        for old_name, new_name in part_renames.items():
            part_group.move(old_name, new_name)
            _write_cgns_name_attr(part_group[new_name], new_name)

        for name in list(part_group.keys()):
            child = part_group[name]
            if _has_cgns_label(child, b"Elements_t"):
                _write_cgns_name_attr(child, name)

        if "ZoneBC" in part_group:
            zone_bc = part_group["ZoneBC"]
            zone_bc_renames: dict[str, str] = {}
            for name in list(zone_bc.keys()):
                stripped = _strip_cgns_prefix(name)
                if stripped != name:
                    zone_bc_renames[name] = stripped

            for old_name, new_name in zone_bc_renames.items():
                zone_bc.move(old_name, new_name)
                bc_group = zone_bc[new_name]
                _write_cgns_name_attr(bc_group, new_name)

            for name in list(zone_bc.keys()):
                bc_group = zone_bc[name]
                _write_cgns_name_attr(bc_group, name)


def export_avl_cgns_via_gmsh(
    mesh3d: Mesh3D,
    params: HelixMeshParams,
    outfile: str,
    gmsh_exe: str = "gmsh",
    keep_intermediate: bool = False,
    intermediate_msh: str | None = None,
    export_solids: bool = True,
    save_all: bool = False,
    hex_order: str = "reverse",
    strip_cgns_prefixes: bool = False,
) -> None:
    resolved_gmsh = resolve_gmsh_executable(gmsh_exe)

    outpath = Path(outfile)
    if outpath.parent != Path("."):
        outpath.parent.mkdir(parents=True, exist_ok=True)

    if intermediate_msh is None:
        intermediate_path = outpath.with_suffix(".intermediate.msh")
    else:
        intermediate_path = Path(intermediate_msh)
    if intermediate_path.parent != Path("."):
        intermediate_path.parent.mkdir(parents=True, exist_ok=True)

    python_error: Exception | None = None
    cli_error: Exception | None = None
    result_stdout = ""
    result_stderr = ""

    try:
        write_cgns_directly_with_python_api(
            mesh3d,
            params,
            outpath,
            intermediate_msh=intermediate_path,
            export_solids=export_solids,
            save_all=save_all,
            hex_order=hex_order,
        )
    except Exception as exc:
        python_error = exc

        if resolved_gmsh is None:
            raise RuntimeError(
                "CGNS conversion failed with the Python gmsh API, and no Gmsh executable was found for fallback.\n"
                f"Python error: {exc}\n"
                f"Requested executable: {gmsh_exe}"
            ) from exc

        export_gmsh_discrete(
            mesh3d,
            params,
            outfile=str(intermediate_path),
            fluid_only=not export_solids,
            hex_order=hex_order,
        )

        try:
            result = convert_msh_to_cgns_with_gmsh_executable(
                intermediate_path,
                outpath,
                gmsh_exe=resolved_gmsh,
                save_all=save_all,
            )
            result_stdout = result.stdout
            result_stderr = result.stderr
        except FileNotFoundError as exc:
            cli_error = exc
            raise RuntimeError(
                "CGNS conversion failed with the Python gmsh API, and the Gmsh executable fallback was not found.\n"
                f"Python error: {python_error}\n"
                f"Requested executable: {gmsh_exe}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            cli_error = exc
            raise RuntimeError(
                "CGNS conversion failed with both the Python gmsh API and the Gmsh executable fallback.\n"
                f"Python error: {python_error}\n"
                f"Command: {' '.join(exc.cmd)}\n"
                f"stdout:\n{exc.stdout}\n"
                f"stderr:\n{exc.stderr}"
            ) from exc

    if not outpath.exists() or outpath.stat().st_size == 0:
        diagnostic = ""
        if cli_error is not None:
            diagnostic = f"stdout:\n{result_stdout}\n" f"stderr:\n{result_stderr}\n"
        elif python_error is not None:
            diagnostic = f"Python error: {python_error}\n"
        raise RuntimeError(
            "CGNS conversion reported success but the CGNS output file is missing or empty.\n"
            f"{diagnostic}"
        )

    if strip_cgns_prefixes:
        strip_prefixed_cgns_names(outpath)

    if not keep_intermediate:
        try:
            intermediate_path.unlink(missing_ok=True)
        except PermissionError:
            pass
