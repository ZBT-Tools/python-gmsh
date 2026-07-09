from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch as LegendPatch
import numpy as np

from params import HelixMeshParams
from topology import Patch2D, Point2D, patch_boundary_xy, point_xy


@dataclass
class PatchGrid2D:
    patch_name: str
    kind: str
    node_ids: np.ndarray


@dataclass
class Mesh2D:
    nodes: dict[int, tuple[float, float]]
    patch_grids: dict[str, PatchGrid2D]
    quads: list[tuple[int, int, int, int, str, str]]


def mesh_patch_bilinear(
    p00: tuple[float, float],
    p10: tuple[float, float],
    p11: tuple[float, float],
    p01: tuple[float, float],
    ni: int,
    nj: int,
) -> np.ndarray:
    if ni <= 0 or nj <= 0:
        raise ValueError("ni and nj must both be positive.")

    grid = np.zeros((ni + 1, nj + 1, 2), dtype=float)
    p00_arr = np.asarray(p00, dtype=float)
    p10_arr = np.asarray(p10, dtype=float)
    p11_arr = np.asarray(p11, dtype=float)
    p01_arr = np.asarray(p01, dtype=float)

    for i in range(ni + 1):
        u = i / ni
        for j in range(nj + 1):
            v = j / nj
            grid[i, j] = (
                (1.0 - u) * (1.0 - v) * p00_arr
                + u * (1.0 - v) * p10_arr
                + u * v * p11_arr
                + (1.0 - u) * v * p01_arr
            )
    return grid


def node_key(x: float, y: float, tolerance: float) -> tuple[int, int]:
    return (round(x / tolerance), round(y / tolerance))


def _is_outer_point_name(name: str) -> bool:
    return name.startswith(("RIB_OUT_", "GAP_OUT_"))


def _is_inner_tube_point_name(name: str) -> bool:
    return name.startswith("AXIS_")


def _is_r_tip_point_name(name: str) -> bool:
    return name.startswith("RIB_IN_")


def _point_theta(xy: tuple[float, float]) -> float:
    return math.atan2(xy[1], xy[0])


def _sample_outer_arc(
    p_start: tuple[float, float],
    p_end: tuple[float, float],
    subdivisions: int,
) -> np.ndarray:
    theta_start = _point_theta(p_start)
    theta_end = _point_theta(p_end)
    while theta_end <= theta_start:
        theta_end += 2.0 * math.pi

    radius = 0.5 * (math.hypot(*p_start) + math.hypot(*p_end))
    thetas = np.linspace(theta_start, theta_end, subdivisions + 1)
    return np.column_stack((radius * np.cos(thetas), radius * np.sin(thetas)))


def _sample_arc_between_points(
    p_start: tuple[float, float],
    p_end: tuple[float, float],
    subdivisions: int,
    direction: str,
) -> np.ndarray:
    theta_start = _point_theta(p_start)
    theta_end = _point_theta(p_end)

    if direction == "ccw":
        while theta_end <= theta_start:
            theta_end += 2.0 * math.pi
    else:
        while theta_end >= theta_start:
            theta_end -= 2.0 * math.pi

    radius = 0.5 * (math.hypot(*p_start) + math.hypot(*p_end))
    thetas = np.linspace(theta_start, theta_end, subdivisions + 1)
    return np.column_stack((radius * np.cos(thetas), radius * np.sin(thetas)))


def _same_radius(a: tuple[float, float], b: tuple[float, float], tol: float = 1.0e-12) -> bool:
    return math.isclose(math.hypot(*a), math.hypot(*b), rel_tol=0.0, abs_tol=tol)


def _same_theta(a: tuple[float, float], b: tuple[float, float], tol: float = 1.0e-12) -> bool:
    return math.isclose(_point_theta(a), _point_theta(b), rel_tol=0.0, abs_tol=tol)


def _unwrap_near(theta: float, reference: float) -> float:
    candidates = (theta - 2.0 * math.pi, theta, theta + 2.0 * math.pi)
    return min(candidates, key=lambda candidate: abs(candidate - reference))


def _is_annular_sector_patch(corners: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]) -> bool:
    p00, p10, p11, p01 = corners
    return (
        _same_radius(p00, p10)
        and _same_radius(p01, p11)
        and _same_theta(p00, p01)
        and _same_theta(p10, p11)
    )


def _mesh_patch_annular_sector(
    p00: tuple[float, float],
    p10: tuple[float, float],
    p11: tuple[float, float],
    p01: tuple[float, float],
    ni: int,
    nj: int,
) -> np.ndarray:
    r_inner = 0.5 * (math.hypot(*p00) + math.hypot(*p10))
    r_outer = 0.5 * (math.hypot(*p01) + math.hypot(*p11))
    theta_left = _point_theta(p00)
    theta_right = _point_theta(p10)
    while theta_right <= theta_left:
        theta_right += 2.0 * math.pi

    grid = np.zeros((ni + 1, nj + 1, 2), dtype=float)
    for i in range(ni + 1):
        u = i / ni
        theta = theta_left + u * (theta_right - theta_left)
        for j in range(nj + 1):
            v = j / nj
            radius = r_inner + v * (r_outer - r_inner)
            grid[i, j] = (radius * math.cos(theta), radius * math.sin(theta))
    return grid


def _is_annulus_outer_patch(patch: Patch2D) -> bool:
    if not patch.name.startswith(("outer_rib_", "outer_gap_")):
        return False
    return patch.name.rsplit("_", 1)[-1].isdigit()


def _mesh_patch_polar_trapezoid(
    points: dict[str, Point2D],
    patch: Patch2D,
) -> np.ndarray:
    p00, p10, p11, p01 = (point_xy(points[name]) for name in patch.corners)
    r_inner = 0.5 * (math.hypot(*p00) + math.hypot(*p10))
    r_outer = 0.5 * (math.hypot(*p01) + math.hypot(*p11))
    theta_left_inner = _point_theta(p00)
    theta_right_inner = _point_theta(p10)
    theta_left_outer = _point_theta(p01)
    theta_right_outer = _point_theta(p11)

    while theta_right_inner <= theta_left_inner:
        theta_right_inner += 2.0 * math.pi
    theta_left_outer = _unwrap_near(theta_left_outer, theta_left_inner)
    theta_right_outer = _unwrap_near(theta_right_outer, theta_right_inner)
    while theta_right_outer <= theta_left_outer:
        theta_right_outer += 2.0 * math.pi

    grid = np.zeros((patch.ni + 1, patch.nj + 1, 2), dtype=float)
    for i in range(patch.ni + 1):
        u = i / patch.ni
        for j in range(patch.nj + 1):
            v = j / patch.nj
            radius = r_inner + v * (r_outer - r_inner)
            theta_left = theta_left_inner + v * (theta_left_outer - theta_left_inner)
            theta_right = theta_right_inner + v * (theta_right_outer - theta_right_inner)
            theta = theta_left + u * (theta_right - theta_left)
            grid[i, j] = (radius * math.cos(theta), radius * math.sin(theta))
    return grid


def _apply_arc_mapping(
    patch_coords: np.ndarray,
    patch: Patch2D,
    points: dict[str, Point2D],
) -> None:
    p00_name, p10_name, p11_name, p01_name = patch.corners

    if _is_inner_tube_point_name(p00_name) and _is_inner_tube_point_name(p10_name):
        bottom_arc = _sample_arc_between_points(point_xy(points[p00_name]), point_xy(points[p10_name]), patch.ni, "ccw")
        for i in range(patch.ni + 1):
            patch_coords[i, 0, :] = bottom_arc[i]

    p01_name = patch.corners[3]
    p11_name = patch.corners[2]
    if _is_outer_point_name(p01_name) and _is_outer_point_name(p11_name):
        top_arc = _sample_arc_between_points(point_xy(points[p01_name]), point_xy(points[p11_name]), patch.ni, "ccw")
        for i in range(patch.ni + 1):
            patch_coords[i, patch.nj, :] = top_arc[i]


def _patch_edges(patch: Patch2D) -> list[tuple[str, tuple[str, str], int]]:
    p00, p10, p11, p01 = patch.corners
    return [
        ("bottom", (p00, p10), patch.ni),
        ("right", (p10, p11), patch.nj),
        ("top", (p01, p11), patch.ni),
        ("left", (p00, p01), patch.nj),
    ]


def _check_shared_edge_subdivisions(patches: list[Patch2D]) -> None:
    edge_map: dict[frozenset[str], list[tuple[str, str, int, tuple[str, str]]]] = {}

    for patch in patches:
        for side_name, ordered_points, subdivisions in _patch_edges(patch):
            key = frozenset(ordered_points)
            edge_map.setdefault(key, []).append((patch.name, side_name, subdivisions, ordered_points))

    for edge_points, edge_uses in edge_map.items():
        if len(edge_uses) < 2:
            continue
        reference = edge_uses[0]
        for other in edge_uses[1:]:
            if other[2] != reference[2]:
                point_names = ", ".join(sorted(edge_points))
                raise ValueError(
                    "Shared edge subdivision mismatch on edge "
                    f"({point_names}): {reference[0]}:{reference[1]} has {reference[2]}, "
                    f"{other[0]}:{other[1]} has {other[2]}."
                )


def _quad_signed_area(nodes: dict[int, tuple[float, float]], quad: tuple[int, int, int, int, str, str]) -> float:
    n00, n10, n11, n01, _, _ = quad
    xy = [nodes[n00], nodes[n10], nodes[n11], nodes[n01]]
    return -0.5 * sum(
        xy[i][0] * xy[(i + 1) % 4][1] - xy[(i + 1) % 4][0] * xy[i][1]
        for i in range(4)
    )


def build_mesh2d(
    points: dict[str, Point2D],
    patches: list[Patch2D],
    tolerance: float = 1.0e-12,
) -> Mesh2D:
    _check_shared_edge_subdivisions(patches)

    nodes: dict[int, tuple[float, float]] = {}
    patch_grids: dict[str, PatchGrid2D] = {}
    quads: list[tuple[int, int, int, int, str, str]] = []

    key_to_id: dict[tuple[int, int], int] = {}
    next_node_id = 1

    for patch in patches:
        p00, p10, p11, p01 = (point_xy(points[name]) for name in patch.corners)
        if _is_annular_sector_patch((p00, p10, p11, p01)):
            patch_coords = _mesh_patch_annular_sector(p00, p10, p11, p01, patch.ni, patch.nj)
        elif _is_annulus_outer_patch(patch):
            patch_coords = _mesh_patch_polar_trapezoid(points, patch)
        else:
            patch_coords = mesh_patch_bilinear(p00, p10, p11, p01, patch.ni, patch.nj)
            _apply_arc_mapping(patch_coords, patch, points)
        node_ids = np.zeros((patch.ni + 1, patch.nj + 1), dtype=int)

        for i in range(patch.ni + 1):
            for j in range(patch.nj + 1):
                x, y = patch_coords[i, j]
                key = node_key(x, y, tolerance)
                if key not in key_to_id:
                    key_to_id[key] = next_node_id
                    nodes[next_node_id] = (float(x), float(y))
                    next_node_id += 1
                node_ids[i, j] = key_to_id[key]

        patch_grids[patch.name] = PatchGrid2D(
            patch_name=patch.name,
            kind=patch.kind,
            node_ids=node_ids,
        )

        for j in range(patch.nj):
            for i in range(patch.ni):
                quad = (
                    int(node_ids[i, j]),
                    int(node_ids[i + 1, j]),
                    int(node_ids[i + 1, j + 1]),
                    int(node_ids[i, j + 1]),
                    patch.name,
                    patch.kind,
                )
                area = _quad_signed_area(nodes, quad)
                if area <= 0.0 or math.isclose(area, 0.0, abs_tol=1.0e-15):
                    raise ValueError(
                        f"Non-positive quad area in patch {patch.name} at local cell (i={i}, j={j})."
                    )
                quads.append(quad)

    return Mesh2D(nodes=nodes, patch_grids=patch_grids, quads=quads)


def plot_mesh2d(
    mesh: Mesh2D,
    points: dict[str, Point2D],
    patches: list[Patch2D],
    params: HelixMeshParams,
    outfile: str = "debug_cross_section_mesh.png",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))

    for radius, style in (
        (params.r_pipe, {"color": "#666666", "linewidth": 1.0, "linestyle": "--"}),
        (params.r_tip, {"color": "#999999", "linewidth": 0.9, "linestyle": ":"}),
    ):
        thetas = np.linspace(0.0, 2.0 * math.pi, 128)
        ax.plot(radius * np.cos(thetas), radius * np.sin(thetas), **style, zorder=0)
    if params.core_mode == "annulus_tube":
        thetas = np.linspace(0.0, 2.0 * math.pi, 128)
        ax.plot(
            params.r_inner_tube * np.cos(thetas),
            params.r_inner_tube * np.sin(thetas),
            color="#bbbbbb",
            linewidth=0.9,
            linestyle="-.",
            zorder=0,
        )

    fill_map = {"fluid": "#d8ecff", "rib_solid": "#ffcf99"}

    for patch in sorted(patches, key=lambda item: 0 if item.kind == "fluid" else 1):
        xs, ys = patch_boundary_xy(points, patch, samples_per_edge=96)
        ax.fill(xs, ys, color=fill_map[patch.kind], alpha=0.45, linewidth=0.0, zorder=1)
        ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]), color="#1f1f1f", linewidth=1.0, zorder=2)

        patch_grid = mesh.patch_grids[patch.name]
        node_ids = patch_grid.node_ids

        for j in range(node_ids.shape[1]):
            line = np.array([mesh.nodes[int(node_ids[i, j])] for i in range(node_ids.shape[0])], dtype=float)
            ax.plot(line[:, 0], line[:, 1], color="#0b3a5b", linewidth=0.55, alpha=0.9, zorder=3)

        for i in range(node_ids.shape[0]):
            line = np.array([mesh.nodes[int(node_ids[i, j])] for j in range(node_ids.shape[1])], dtype=float)
            ax.plot(line[:, 0], line[:, 1], color="#0b3a5b", linewidth=0.55, alpha=0.9, zorder=3)

        corner_xy = np.array([point_xy(points[name]) for name in patch.corners], dtype=float)
        ax.text(
            float(corner_xy[:, 0].mean()),
            float(corner_xy[:, 1].mean()),
            patch.name,
            fontsize=8,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            zorder=4,
        )

    legend_handles = [
        Line2D([0], [0], color="#666666", linestyle="--", linewidth=1.0, label="r_pipe"),
        Line2D([0], [0], color="#999999", linestyle=":", linewidth=1.0, label="r_tip"),
        LegendPatch(facecolor=fill_map["fluid"], edgecolor="#1f1f1f", label="fluid patch"),
        LegendPatch(facecolor=fill_map["rib_solid"], edgecolor="#1f1f1f", label="rib_solid patch"),
    ]
    if params.core_mode == "annulus_tube":
        legend_handles.insert(2, Line2D([0], [0], color="#bbbbbb", linestyle="-.", linewidth=1.0, label="r_inner_tube"))
    else:
        legend_handles.insert(2, Line2D([0], [0], color="#bbbbbb", linestyle="-.", linewidth=1.0, label="r_core"))
    ax.legend(handles=legend_handles, loc="upper right")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Helix mesh structured 2D debug plot")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
