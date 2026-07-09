from __future__ import annotations

import sys
from pathlib import Path
import math
from collections import Counter, defaultdict

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from params import HelixMeshParams
from topology import build_topology, patch_signed_area, point_radius, point_xy, topology_summary


def _patch_edges(corners: tuple[str, str, str, str]) -> list[tuple[str, str]]:
    return [
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0]),
    ]


def _is_outer_wall_edge(edge: frozenset[str]) -> bool:
    return all(name.startswith(("RIB_OUT_", "GAP_OUT_")) for name in edge)


def test_topology_patch_count() -> None:
    points, patches = build_topology(HelixMeshParams())
    inner_patches = [patch for patch in patches if patch.name.startswith("inner_")]
    outer_patches = [patch for patch in patches if patch.name.startswith("outer_")]

    assert points
    assert len(patches) == 18
    assert len(inner_patches) == 6
    assert len(outer_patches) == 12


def test_topology_kind_count() -> None:
    _, patches = build_topology(HelixMeshParams())
    rib_patches = [patch for patch in patches if patch.kind == "rib_solid"]
    fluid_patches = [patch for patch in patches if patch.kind == "fluid"]

    assert len(rib_patches) == 6
    assert len(fluid_patches) == 12


def test_inner_outer_patch_count() -> None:
    _, patches = build_topology(HelixMeshParams())
    inner_patches = [patch for patch in patches if patch.name.startswith("inner_")]
    outer_patches = [patch for patch in patches if patch.name.startswith("outer_")]
    outer_gap_patches = [patch for patch in patches if patch.name.startswith("outer_gap_")]
    outer_rib_patches = [patch for patch in patches if patch.name.startswith("outer_rib_")]

    assert len(inner_patches) == 6
    assert len(outer_patches) == 12
    assert len(outer_gap_patches) == 6
    assert len(outer_rib_patches) == 6


def test_required_patch_names_exist() -> None:
    _, patches = build_topology(HelixMeshParams())
    names = {patch.name for patch in patches}
    required = {
        "inner_0",
        "inner_1",
        "inner_2",
        "inner_3",
        "inner_4",
        "inner_5",
        "outer_rib_0_left",
        "outer_rib_0_right",
        "outer_rib_1_left",
        "outer_rib_1_right",
        "outer_rib_2_left",
        "outer_rib_2_right",
        "outer_gap_0_left",
        "outer_gap_0_right",
        "outer_gap_1_left",
        "outer_gap_1_right",
        "outer_gap_2_left",
        "outer_gap_2_right",
    }

    assert names == required


def test_all_patch_corners_exist() -> None:
    points, patches = build_topology(HelixMeshParams())

    for patch in patches:
        assert len(patch.corners) == 4
        for corner_name in patch.corners:
            assert corner_name in points


def test_no_duplicate_corners_within_patch() -> None:
    _, patches = build_topology(HelixMeshParams())

    for patch in patches:
        assert len(set(patch.corners)) == 4


def test_params_reject_non_three_ribs() -> None:
    with pytest.raises(ValueError):
        build_topology(HelixMeshParams(n_ribs=4))


def test_patch_polygon_areas_positive() -> None:
    points, patches = build_topology(HelixMeshParams())

    for patch in patches:
        area = patch_signed_area(points, patch)
        assert area > 1.0e-12


def test_patch_names_unique() -> None:
    _, patches = build_topology(HelixMeshParams())
    names = [patch.name for patch in patches]
    assert len(names) == len(set(names))


def test_inner_patches_share_center_point() -> None:
    points, patches = build_topology(HelixMeshParams())

    assert "CENTER" in points
    assert math.isclose(points["CENTER"].x, 0.0)
    assert math.isclose(points["CENTER"].y, 0.0)

    for patch in patches:
        if patch.name.startswith("inner_"):
            assert patch.corners[0] == "CENTER"
            assert len(set(patch.corners)) == 4


def test_gap_inner_midpoints_are_chord_midpoints() -> None:
    points, _ = build_topology(HelixMeshParams())

    for i in range(3):
        a = points[f"RIB_IN_R_{i}"]
        b = points[f"RIB_IN_L_{(i + 1) % 3}"]
        m = points[f"GAP_IN_M_{i}"]
        assert math.isclose(m.x, 0.5 * (a.x + b.x))
        assert math.isclose(m.y, 0.5 * (a.y + b.y))


def test_gap_inner_midpoints_not_forced_to_r_tip() -> None:
    params = HelixMeshParams()
    points, _ = build_topology(params)

    for i in range(3):
        assert point_radius(points[f"GAP_IN_M_{i}"]) < params.r_tip


def test_topology_summary_reports_correct_counts() -> None:
    _, patches = build_topology(HelixMeshParams())

    assert topology_summary(patches) == "\n".join(
        [
            "Total patches: 18",
            "Inner fluid patches: 6",
            "Outer fluid patches: 6",
            "Outer rib_solid patches: 6",
            "Total fluid patches: 12",
            "Total rib_solid patches: 6",
        ]
    )


def test_each_rib_and_gap_has_two_outer_half_patches() -> None:
    _, patches = build_topology(HelixMeshParams())
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)

    for patch in patches:
        parts = patch.name.split("_")
        if patch.name.startswith("outer_rib_"):
            grouped[("outer_rib", parts[2])].append(parts[3])
        elif patch.name.startswith("outer_gap_"):
            grouped[("outer_gap", parts[2])].append(parts[3])

    for family in ("outer_rib", "outer_gap"):
        for index in ("0", "1", "2"):
            assert sorted(grouped[(family, index)]) == ["left", "right"]


def test_internal_edges_are_shared_and_boundary_edges_are_outer_wall_only() -> None:
    _, patches = build_topology(HelixMeshParams())
    edge_counts: Counter[frozenset[str]] = Counter()

    for patch in patches:
        for edge in _patch_edges(patch.corners):
            edge_counts[frozenset(edge)] += 1

    for edge, count in edge_counts.items():
        if _is_outer_wall_edge(edge):
            assert count == 1
        else:
            assert count == 2


def test_no_free_internal_edges_exist() -> None:
    _, patches = build_topology(HelixMeshParams())
    edge_to_patches: dict[frozenset[str], list[str]] = defaultdict(list)

    for patch in patches:
        for edge in _patch_edges(patch.corners):
            edge_to_patches[frozenset(edge)].append(patch.name)

    free_internal_edges = {
        tuple(sorted(edge)): owners
        for edge, owners in edge_to_patches.items()
        if not _is_outer_wall_edge(edge) and len(owners) != 2
    }

    assert free_internal_edges == {}


def test_annulus_mode_builds_inner_tube_points() -> None:
    params = HelixMeshParams(core_mode="annulus_tube", r_inner_tube=1.2e-3)
    points, patches = build_topology(params)

    assert "CENTER" not in points
    for name in ("AXIS_RIB_L_0", "AXIS_RIB_R_0", "AXIS_RIB_L_1", "AXIS_RIB_R_1", "AXIS_RIB_L_2", "AXIS_RIB_R_2"):
        assert name in points
        assert math.isclose(point_radius(points[name]), params.r_inner_tube, rel_tol=0.0, abs_tol=1.0e-12)

    assert len(patches) == 12


def test_annulus_mode_inner_patches_no_longer_use_center_point() -> None:
    params = HelixMeshParams(core_mode="annulus_tube", r_inner_tube=1.2e-3)
    _, patches = build_topology(params)

    for patch in patches:
        if patch.name.startswith("inner_"):
            assert patch.corners[0].startswith("AXIS_")
            assert patch.corners[1].startswith("AXIS_")
