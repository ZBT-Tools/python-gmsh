from __future__ import annotations

import math
import shutil
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_gmsh import (
    BOUNDARY_NAMES,
    FLUID_BOUNDARY_EXPORT_NAMES,
    FLUID_VOLUME_EXPORT_NAME,
    FULL_BOUNDARY_NAMES,
    FULL_BOUNDARY_EXPORT_NAMES,
    FULL_VOLUME_EXPORT_NAMES,
    HEX_ORDER_MODES,
    _assign_entity_and_physical_name,
    build_gmsh_export_data,
    build_gmsh_full_export_data,
    classify_fluid_boundary_faces,
    collect_face_adjacency,
    export_gmsh_discrete,
    reorder_hex_for_export,
)
from extrude import build_mesh3d
from mesh2d import build_mesh2d
from params import HelixMeshParams
from topology import build_topology


def _build_default_3d():
    params = HelixMeshParams()
    points, patches = build_topology(params)
    mesh2d = build_mesh2d(points, patches)
    mesh3d = build_mesh3d(mesh2d, params)
    return params, points, patches, mesh2d, mesh3d
@pytest.fixture
def workspace_tmpdir() -> Path:
    root = Path(__file__).resolve().parents[1] / f".tmp-test-export-gmsh-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_collect_face_adjacency_builds() -> None:
    _, _, _, _, mesh3d = _build_default_3d()
    adjacency = collect_face_adjacency(mesh3d)

    assert adjacency
    assert any(len(owners) == 2 for owners in adjacency.values())
    assert any(len(owners) == 1 for owners in adjacency.values())


def test_reorder_hex_for_export_modes_return_unique_nodes() -> None:
    node_ids = (10, 11, 12, 13, 14, 15, 16, 17)

    for mode in HEX_ORDER_MODES:
        reordered = reorder_hex_for_export(node_ids, mode)
        assert len(reordered) == 8
        assert len(set(reordered)) == 8
        assert set(reordered) == set(node_ids)


def test_reorder_hex_for_export_default_is_identity() -> None:
    node_ids = (10, 11, 12, 13, 14, 15, 16, 17)

    assert reorder_hex_for_export(node_ids, "default") == node_ids


def test_reorder_hex_for_export_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unknown hex export order"):
        reorder_hex_for_export((10, 11, 12, 13, 14, 15, 16, 17), "bad_mode")


def test_assign_entity_and_physical_name_sets_both() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.entity_calls: list[tuple[int, int, str]] = []
            self.physical_calls: list[tuple[int, int, str]] = []

        def setEntityName(self, dim: int, tag: int, name: str) -> None:
            self.entity_calls.append((dim, tag, name))

        def setPhysicalName(self, dim: int, tag: int, name: str) -> None:
            self.physical_calls.append((dim, tag, name))

    class FakeGmsh:
        def __init__(self) -> None:
            self.model = FakeModel()

    fake_gmsh = FakeGmsh()
    _assign_entity_and_physical_name(fake_gmsh, 2, 7, 13, "BND_W_ReactorWall")

    assert fake_gmsh.model.entity_calls == [(2, 7, "BND_W_ReactorWall")]
    assert fake_gmsh.model.physical_calls == [(2, 13, "BND_W_ReactorWall")]


def test_classify_boundary_faces_keys() -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    boundary_faces = classify_fluid_boundary_faces(mesh3d, params)

    assert set(boundary_faces) == set(BOUNDARY_NAMES)


def test_boundary_faces_nonempty_except_axis_core() -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    boundary_faces = classify_fluid_boundary_faces(mesh3d, params)

    assert boundary_faces["inlet"]
    assert boundary_faces["outlet"]
    assert boundary_faces["outer_wall"]
    assert boundary_faces["rib_wall"]
    # The current corrected center-point topology has no open inner core surface.
    assert boundary_faces["axis_core"] == []


def test_boundary_faces_are_quads() -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    boundary_faces = classify_fluid_boundary_faces(mesh3d, params)

    for faces in boundary_faces.values():
        for face in faces:
            assert len(face) == 4


def test_boundary_face_node_ids_valid() -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    boundary_faces = classify_fluid_boundary_faces(mesh3d, params)

    for faces in boundary_faces.values():
        for face in faces:
            for node_id in face:
                assert node_id in mesh3d.nodes


def test_no_duplicate_nodes_within_boundary_face() -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    boundary_faces = classify_fluid_boundary_faces(mesh3d, params)

    for faces in boundary_faces.values():
        for face in faces:
            assert len(set(face)) == 4


def test_no_internal_fluid_faces_exported() -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    adjacency = collect_face_adjacency(mesh3d)
    boundary_faces = classify_fluid_boundary_faces(mesh3d, params)

    for faces in boundary_faces.values():
        for face in faces:
            owners = adjacency[tuple(sorted(face))]
            assert sum(1 for owner in owners if owner.kind == "fluid") == 1


def test_rib_wall_faces_adjacent_to_rib_solid() -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    adjacency = collect_face_adjacency(mesh3d)
    boundary_faces = classify_fluid_boundary_faces(mesh3d, params)

    for face in boundary_faces["rib_wall"]:
        owners = adjacency[tuple(sorted(face))]
        assert {owner.kind for owner in owners} == {"fluid", "rib_solid"}


def test_inlet_faces_are_at_first_layer() -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    boundary_faces = classify_fluid_boundary_faces(mesh3d, params)

    for face in boundary_faces["inlet"]:
        for node_id in face:
            assert math.isclose(mesh3d.nodes[node_id][2], 0.0, abs_tol=1.0e-12)


def test_outlet_faces_are_at_last_layer() -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    boundary_faces = classify_fluid_boundary_faces(mesh3d, params)

    for face in boundary_faces["outlet"]:
        for node_id in face:
            assert math.isclose(mesh3d.nodes[node_id][2], params.length, abs_tol=1.0e-12)


def test_export_gmsh_writes_file(workspace_tmpdir: Path) -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    outfile = workspace_tmpdir / "export.msh"

    export_gmsh_discrete(mesh3d, params, outfile=str(outfile))

    assert outfile.exists()
    assert outfile.stat().st_size > 0


def test_export_file_contains_physical_names(workspace_tmpdir: Path) -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    outfile = workspace_tmpdir / "export.msh"

    export_gmsh_discrete(mesh3d, params, outfile=str(outfile))

    text = outfile.read_text(encoding="utf-8")
    for name in (FLUID_VOLUME_EXPORT_NAME, *FLUID_BOUNDARY_EXPORT_NAMES.values()):
        assert name in text


def test_exported_fluid_hex_count() -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    export_data = build_gmsh_export_data(mesh3d, params)

    assert len(export_data.fluid_hexes) == sum(1 for hex_element in mesh3d.hexes if hex_element.kind == "fluid")


def test_no_rib_solid_volume_exported(workspace_tmpdir: Path) -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    export_data = build_gmsh_export_data(mesh3d, params)
    outfile = workspace_tmpdir / "export.msh"

    export_gmsh_discrete(mesh3d, params, outfile=str(outfile))

    assert all(hex_element.kind == "fluid" for hex_element in export_data.fluid_hexes)
    text = outfile.read_text(encoding="utf-8")
    assert FULL_VOLUME_EXPORT_NAMES["rib_solid"] not in text


def test_full_export_data_contains_two_volume_domains() -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    export_data = build_gmsh_full_export_data(mesh3d, params)

    assert set(export_data.volume_hexes) == {"fluid", "rib_solid"}
    assert len(export_data.volume_hexes["fluid"]) == sum(1 for hex_element in mesh3d.hexes if hex_element.kind == "fluid")
    assert len(export_data.volume_hexes["rib_solid"]) == sum(1 for hex_element in mesh3d.hexes if hex_element.kind == "rib_solid")


def test_full_export_interface_faces_exist() -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    export_data = build_gmsh_full_export_data(mesh3d, params)
    adjacency = collect_face_adjacency(mesh3d)

    assert set(export_data.boundary_faces) == set(FULL_BOUNDARY_NAMES)
    assert export_data.boundary_faces["fluid_solid_interface"]

    for face in export_data.boundary_faces["fluid_solid_interface"]:
        owners = adjacency[tuple(sorted(face.node_ids))]
        assert {owner.kind for owner in owners} == {"fluid", "rib_solid"}


def test_full_export_grouped_boundary_names_exist() -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    export_data = build_gmsh_full_export_data(mesh3d, params)

    assert export_data.boundary_faces["inlet_gas"]
    assert export_data.boundary_faces["inlet_rib"]
    assert export_data.boundary_faces["outlet_gas"]
    assert export_data.boundary_faces["outlet_rib"]
    assert export_data.boundary_faces["reactor_wall"]
    assert export_data.boundary_faces["rib_wall"]


def test_full_export_writes_rib_domain(workspace_tmpdir: Path) -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    outfile = workspace_tmpdir / "export_full.msh"

    export_gmsh_discrete(mesh3d, params, outfile=str(outfile), fluid_only=False)

    text = outfile.read_text(encoding="utf-8")
    assert FULL_VOLUME_EXPORT_NAMES["fluid"] in text
    assert FULL_VOLUME_EXPORT_NAMES["rib_solid"] in text
    for name in FULL_BOUNDARY_EXPORT_NAMES.values():
        assert name in text
    assert "fluid_solid_interface" not in text


def test_annulus_mode_exports_nonempty_axis_core() -> None:
    params = HelixMeshParams(core_mode="annulus_tube", r_inner_tube=1.2e-3)
    points, patches = build_topology(params)
    mesh2d = build_mesh2d(points, patches)
    mesh3d = build_mesh3d(mesh2d, params)

    boundary_faces = classify_fluid_boundary_faces(mesh3d, params)

    assert boundary_faces["axis_core"]
