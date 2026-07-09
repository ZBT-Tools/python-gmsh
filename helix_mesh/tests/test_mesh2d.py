from __future__ import annotations

import math
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesh2d import build_mesh2d
from params import HelixMeshParams
from topology import build_topology


def _build_default_mesh():
    params = HelixMeshParams()
    points, patches = build_topology(params)
    mesh = build_mesh2d(points, patches)
    return params, points, patches, mesh


def test_mesh2d_builds() -> None:
    _, _, _, mesh = _build_default_mesh()
    assert len(mesh.nodes) > 0
    assert len(mesh.patch_grids) == 18
    assert len(mesh.quads) > 0


def test_patch_grid_shapes() -> None:
    _, _, patches, mesh = _build_default_mesh()
    for patch in patches:
        assert patch.name in mesh.patch_grids
        assert mesh.patch_grids[patch.name].node_ids.shape == (patch.ni + 1, patch.nj + 1)


def test_quad_count() -> None:
    _, _, patches, mesh = _build_default_mesh()
    expected = sum(patch.ni * patch.nj for patch in patches)
    assert len(mesh.quads) == expected


def test_node_ids_are_valid() -> None:
    _, _, _, mesh = _build_default_mesh()
    for quad in mesh.quads:
        for node_id in quad[:4]:
            assert node_id in mesh.nodes


def test_no_duplicate_nodes_within_quad() -> None:
    _, _, _, mesh = _build_default_mesh()
    for quad in mesh.quads:
        assert len(set(quad[:4])) == 4


def test_quad_areas_positive() -> None:
    _, _, _, mesh = _build_default_mesh()
    for quad in mesh.quads:
        xy = [mesh.nodes[node_id] for node_id in quad[:4]]
        area = -0.5 * sum(
            xy[i][0] * xy[(i + 1) % 4][1] - xy[(i + 1) % 4][0] * xy[i][1]
            for i in range(4)
        )
        assert area > 1.0e-14


def test_shared_edge_subdivision_compatibility_default() -> None:
    _, points, patches, _ = _build_default_mesh()
    build_mesh2d(points, patches)


def test_shared_edge_subdivision_mismatch_raises() -> None:
    _, points, patches, _ = _build_default_mesh()
    modified = deepcopy(patches)
    target_idx = next(i for i, patch in enumerate(modified) if patch.name == "outer_gap_0_left")
    modified[target_idx] = replace(modified[target_idx], nj=modified[target_idx].nj + 1)

    with pytest.raises(ValueError) as exc_info:
        build_mesh2d(points, modified)

    message = str(exc_info.value)
    assert "outer_gap_0_left" in message
    assert "outer_rib_0_right" in message or "RIB_IN_R_0" in message or "RIB_OUT_R_0" in message


def test_gap_inner_midpoints_remain_chord_based() -> None:
    _, points, _, _ = _build_default_mesh()
    for i in range(3):
        a = points[f"RIB_IN_R_{i}"]
        b = points[f"RIB_IN_L_{(i + 1) % 3}"]
        m = points[f"GAP_IN_M_{i}"]
        assert math.isclose(m.x, 0.5 * (a.x + b.x))
        assert math.isclose(m.y, 0.5 * (a.y + b.y))


def test_gap_inner_midpoints_not_on_r_tip() -> None:
    params, points, _, _ = _build_default_mesh()
    for i in range(3):
        radius = math.hypot(points[f"GAP_IN_M_{i}"].x, points[f"GAP_IN_M_{i}"].y)
        assert radius < params.r_tip


def test_outer_boundary_nodes_lie_on_pipe_radius() -> None:
    params, _, patches, mesh = _build_default_mesh()

    for patch in patches:
        if not patch.name.startswith("outer_"):
            continue
        node_ids = mesh.patch_grids[patch.name].node_ids
        top_edge_ids = node_ids[:, -1]
        for node_id in top_edge_ids:
            x, y = mesh.nodes[int(node_id)]
            assert math.isclose(math.hypot(x, y), params.r_pipe, rel_tol=1.0e-10, abs_tol=1.0e-12)


def test_annulus_mode_inner_boundary_nodes_lie_on_inner_tube_radius() -> None:
    params = HelixMeshParams(core_mode="annulus_tube", r_inner_tube=1.2e-3)
    points, patches = build_topology(params)
    mesh = build_mesh2d(points, patches)

    for patch in patches:
        if not patch.name.startswith("inner_"):
            continue
        node_ids = mesh.patch_grids[patch.name].node_ids
        inner_edge_ids = node_ids[:, 0]
        for node_id in inner_edge_ids:
            x, y = mesh.nodes[int(node_id)]
            assert math.isclose(math.hypot(x, y), params.r_inner_tube, rel_tol=1.0e-10, abs_tol=1.0e-12)
