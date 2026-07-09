from __future__ import annotations

import math
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extrude import build_mesh3d, hex_signed_volume, plot_3d_preview
from mesh2d import build_mesh2d
from params import HelixMeshParams
from topology import build_topology


def _pitch_angle_for_twist_deg(length: float, diameter: float, twist_deg: float) -> float:
    if math.isclose(twist_deg, 0.0, abs_tol=1.0e-15):
        return 90.0
    return math.degrees(math.atan(length * 360.0 / (math.pi * diameter * twist_deg)))


def _build_default_3d():
    params = HelixMeshParams()
    points, patches = build_topology(params)
    mesh2d = build_mesh2d(points, patches)
    mesh3d = build_mesh3d(mesh2d, params)
    return params, points, patches, mesh2d, mesh3d


def test_mesh3d_builds() -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    assert len(mesh3d.nodes) > 0
    assert len(mesh3d.hexes) > 0
    assert mesh3d.n_z == params.n_z


def test_node_count() -> None:
    params, _, _, mesh2d, mesh3d = _build_default_3d()
    assert len(mesh3d.nodes) == len(mesh2d.nodes) * (params.n_z + 1)


def test_hex_count() -> None:
    params, _, _, mesh2d, mesh3d = _build_default_3d()
    assert len(mesh3d.hexes) == len(mesh2d.quads) * params.n_z


def test_hex_node_ids_are_valid() -> None:
    _, _, _, _, mesh3d = _build_default_3d()
    for hex_element in mesh3d.hexes:
        for node_id in hex_element.node_ids:
            assert node_id in mesh3d.nodes


def test_no_duplicate_nodes_within_hex() -> None:
    _, _, _, _, mesh3d = _build_default_3d()
    for hex_element in mesh3d.hexes:
        assert len(set(hex_element.node_ids)) == 8


def test_fluid_and_rib_solid_hex_counts() -> None:
    params, _, _, mesh2d, mesh3d = _build_default_3d()
    n_fluid_quads = sum(1 for quad in mesh2d.quads if quad[5] == "fluid")
    n_rib_quads = sum(1 for quad in mesh2d.quads if quad[5] == "rib_solid")
    n_fluid_hexes = sum(1 for hex_element in mesh3d.hexes if hex_element.kind == "fluid")
    n_rib_hexes = sum(1 for hex_element in mesh3d.hexes if hex_element.kind == "rib_solid")
    assert n_fluid_hexes == n_fluid_quads * params.n_z
    assert n_rib_hexes == n_rib_quads * params.n_z


def test_first_and_last_layer_z_coordinates() -> None:
    params, _, _, mesh2d, mesh3d = _build_default_3d()
    for node2d_id in mesh2d.nodes:
        first_id = mesh3d.node_layers[(node2d_id, 0)]
        last_id = mesh3d.node_layers[(node2d_id, params.n_z)]
        assert math.isclose(mesh3d.nodes[first_id][2], 0.0, abs_tol=1.0e-14)
        assert math.isclose(mesh3d.nodes[last_id][2], params.length, abs_tol=1.0e-14)


def test_zero_twist_preserves_xy() -> None:
    reference = HelixMeshParams()
    params = HelixMeshParams(
        pitch_angle_deg=_pitch_angle_for_twist_deg(reference.length, reference.diameter, 0.0)
    )
    points, patches = build_topology(params)
    mesh2d = build_mesh2d(points, patches)
    mesh3d = build_mesh3d(mesh2d, params)
    for layer in (0, params.n_z // 2, params.n_z):
        for node2d_id, (x2, y2) in mesh2d.nodes.items():
            node3d_id = mesh3d.node_layers[(node2d_id, layer)]
            x3, y3, _ = mesh3d.nodes[node3d_id]
            assert math.isclose(x3, x2, abs_tol=1.0e-12)
            assert math.isclose(y3, y2, abs_tol=1.0e-12)


def test_full_turn_preserves_final_xy() -> None:
    reference = HelixMeshParams()
    params = HelixMeshParams(
        pitch_angle_deg=_pitch_angle_for_twist_deg(reference.length, reference.diameter, 360.0)
    )
    points, patches = build_topology(params)
    mesh2d = build_mesh2d(points, patches)
    mesh3d = build_mesh3d(mesh2d, params)
    for node2d_id, (x2, y2) in mesh2d.nodes.items():
        node3d_id = mesh3d.node_layers[(node2d_id, params.n_z)]
        x3, y3, _ = mesh3d.nodes[node3d_id]
        assert math.isclose(x3, x2, abs_tol=1.0e-10)
        assert math.isclose(y3, y2, abs_tol=1.0e-10)


def test_half_turn_rotates_final_xy() -> None:
    reference = HelixMeshParams()
    params = HelixMeshParams(
        pitch_angle_deg=_pitch_angle_for_twist_deg(reference.length, reference.diameter, 180.0)
    )
    points, patches = build_topology(params)
    mesh2d = build_mesh2d(points, patches)
    mesh3d = build_mesh3d(mesh2d, params)
    for node2d_id, (x2, y2) in mesh2d.nodes.items():
        node3d_id = mesh3d.node_layers[(node2d_id, params.n_z)]
        x3, y3, _ = mesh3d.nodes[node3d_id]
        assert math.isclose(x3, -x2, abs_tol=1.0e-10)
        assert math.isclose(y3, -y2, abs_tol=1.0e-10)


def test_hex_volumes_nonzero_and_consistent_orientation() -> None:
    _, _, _, _, mesh3d = _build_default_3d()
    signs: set[int] = set()
    for hex_element in mesh3d.hexes:
        volume = hex_signed_volume([mesh3d.nodes[node_id] for node_id in hex_element.node_ids])
        assert not math.isclose(volume, 0.0, abs_tol=1.0e-18)
        signs.add(1 if volume > 0.0 else -1)
    assert len(signs) == 1


def test_plot_3d_preview_writes_file() -> None:
    params, _, _, _, mesh3d = _build_default_3d()
    outfile = Path(__file__).resolve().parents[1] / f"preview_{uuid.uuid4().hex}.png"
    plot_3d_preview(mesh3d, params, outfile=str(outfile), max_edges=500)
    assert outfile.exists()
    assert outfile.stat().st_size > 0
