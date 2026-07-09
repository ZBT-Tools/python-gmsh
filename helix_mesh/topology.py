from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from matplotlib.lines import Line2D
from matplotlib.patches import Patch as LegendPatch
import matplotlib.pyplot as plt
import numpy as np

from params import HelixMeshParams

PatchKind = Literal["fluid", "rib_solid"]

TAU = 2.0 * math.pi


@dataclass(frozen=True)
class Point2D:
    name: str
    x: float
    y: float


@dataclass(frozen=True)
class Patch2D:
    name: str
    corners: tuple[str, str, str, str]
    kind: PatchKind
    ni: int
    nj: int


def _normalize_theta(theta: float) -> float:
    return theta % TAU


def _unwrap_to_be_after(theta: float, reference: float) -> float:
    while theta <= reference:
        theta += TAU
    return theta


def _interpolate_angle(start: float, end: float, t: float) -> float:
    return start + t * (end - start)


def point_from_polar(name: str, r: float, theta: float) -> Point2D:
    return Point2D(name=name, x=r * math.cos(theta), y=r * math.sin(theta))


def point_midpoint(name: str, a: Point2D, b: Point2D) -> Point2D:
    return Point2D(name=name, x=0.5 * (a.x + b.x), y=0.5 * (a.y + b.y))


def point_xy(point: Point2D) -> tuple[float, float]:
    return point.x, point.y


def point_radius(point: Point2D) -> float:
    return math.hypot(point.x, point.y)


def point_theta(point: Point2D) -> float:
    return _normalize_theta(math.atan2(point.y, point.x))


def patch_signed_area(points: dict[str, Point2D], patch: Patch2D) -> float:
    xy = [point_xy(points[name]) for name in patch.corners]
    return -0.5 * sum(
        xy[i][0] * xy[(i + 1) % 4][1] - xy[(i + 1) % 4][0] * xy[i][1]
        for i in range(4)
    )


def topology_summary(patches: list[Patch2D]) -> str:
    inner_fluid = sum(1 for patch in patches if patch.name.startswith("inner_") and patch.kind == "fluid")
    outer_fluid = sum(1 for patch in patches if patch.name.startswith("outer_gap_") and patch.kind == "fluid")
    outer_rib = sum(1 for patch in patches if patch.name.startswith("outer_rib_") and patch.kind == "rib_solid")
    total_fluid = sum(1 for patch in patches if patch.kind == "fluid")
    total_rib = sum(1 for patch in patches if patch.kind == "rib_solid")
    return "\n".join(
        [
            f"Total patches: {len(patches)}",
            f"Inner fluid patches: {inner_fluid}",
            f"Outer fluid patches: {outer_fluid}",
            f"Outer rib_solid patches: {outer_rib}",
            f"Total fluid patches: {total_fluid}",
            f"Total rib_solid patches: {total_rib}",
        ]
    )


def _build_center_point_inner_patches(
    patches: list[Patch2D],
    params: HelixMeshParams,
) -> None:
    inner_patch_corners = [
        ("CENTER", "GAP_IN_M_0", "RIB_IN_R_0", "RIB_IN_M_0"),
        ("CENTER", "RIB_IN_M_1", "RIB_IN_L_1", "GAP_IN_M_0"),
        ("CENTER", "GAP_IN_M_1", "RIB_IN_R_1", "RIB_IN_M_1"),
        ("CENTER", "RIB_IN_M_2", "RIB_IN_L_2", "GAP_IN_M_1"),
        ("CENTER", "GAP_IN_M_2", "RIB_IN_R_2", "RIB_IN_M_2"),
        ("CENTER", "RIB_IN_M_0", "RIB_IN_L_0", "GAP_IN_M_2"),
    ]

    for idx, corners in enumerate(inner_patch_corners):
        if idx % 2 == 0:
            ni = params.n_tangential_rib_half
            nj = params.n_tangential_gap_half
        else:
            ni = params.n_tangential_gap_half
            nj = params.n_tangential_rib_half

        patches.append(
            Patch2D(
                name=f"inner_{idx}",
                corners=corners,
                kind="fluid",
                ni=ni,
                nj=nj,
            )
        )


def _build_annulus_inner_patches(
    points: dict[str, Point2D],
    patches: list[Patch2D],
    params: HelixMeshParams,
    rib_data: list[dict[str, float]],
) -> None:
    for i, rib in enumerate(rib_data):
        next_i = (i + 1) % params.n_ribs
        add_left = point_from_polar(f"AXIS_RIB_L_{i}", params.r_inner_tube, rib["li_inner"])
        add_right = point_from_polar(f"AXIS_RIB_R_{i}", params.r_inner_tube, rib["ri_inner"])
        points[add_left.name] = add_left
        points[add_right.name] = add_right

        patches.append(
            Patch2D(
                name=f"inner_rib_{i}",
                corners=(f"AXIS_RIB_L_{i}", f"AXIS_RIB_R_{i}", f"RIB_IN_R_{i}", f"RIB_IN_L_{i}"),
                kind="fluid",
                ni=2 * params.n_tangential_rib_half,
                nj=params.n_radial_inner,
            )
        )

        patches.append(
            Patch2D(
                name=f"inner_gap_{i}",
                corners=(f"AXIS_RIB_R_{i}", f"AXIS_RIB_L_{next_i}", f"RIB_IN_L_{next_i}", f"RIB_IN_R_{i}"),
                kind="fluid",
                ni=2 * params.n_tangential_gap_half,
                nj=params.n_radial_inner,
            )
        )


def build_topology(params: HelixMeshParams) -> tuple[dict[str, Point2D], list[Patch2D]]:
    params.validate()

    points: dict[str, Point2D] = {}
    patches: list[Patch2D] = []

    def add_point(point: Point2D) -> str:
        existing = points.get(point.name)
        if existing is None:
            points[point.name] = point
            return point.name
        if not (math.isclose(existing.x, point.x) and math.isclose(existing.y, point.y)):
            raise ValueError(f"Point name {point.name} is reused with different coordinates.")
        return point.name

    rib_data: list[dict[str, float]] = []
    for i in range(params.n_ribs):
        theta_center = TAU * i / params.n_ribs
        theta_li_inner = theta_center - 0.5 * params.rib_width_inner_angle
        theta_ri_inner = theta_center + 0.5 * params.rib_width_inner_angle
        theta_li_outer = theta_center - 0.5 * params.rib_width_outer_angle
        theta_ri_outer = theta_center + 0.5 * params.rib_width_outer_angle
        rib_data.append(
            {
                "center": theta_center,
                "li_inner": theta_li_inner,
                "ri_inner": theta_ri_inner,
                "li_outer": theta_li_outer,
                "ri_outer": theta_ri_outer,
            }
        )

        add_point(point_from_polar(f"RIB_IN_L_{i}", params.r_tip, theta_li_inner))
        add_point(point_from_polar(f"RIB_IN_M_{i}", params.r_tip, theta_center))
        add_point(point_from_polar(f"RIB_IN_R_{i}", params.r_tip, theta_ri_inner))
        add_point(point_from_polar(f"RIB_OUT_L_{i}", params.r_pipe, theta_li_outer))
        add_point(point_from_polar(f"RIB_OUT_M_{i}", params.r_pipe, theta_center))
        add_point(point_from_polar(f"RIB_OUT_R_{i}", params.r_pipe, theta_ri_outer))

    for i, rib in enumerate(rib_data):
        next_i = (i + 1) % params.n_ribs
        start_outer = rib["ri_outer"]
        end_outer = _unwrap_to_be_after(rib_data[next_i]["li_outer"], start_outer)
        theta_mid_outer = _interpolate_angle(start_outer, end_outer, 0.5)

        add_point(
            point_midpoint(
                f"GAP_IN_M_{i}",
                points[f"RIB_IN_R_{i}"],
                points[f"RIB_IN_L_{next_i}"],
            )
        )
        add_point(point_from_polar(f"GAP_OUT_M_{i}", params.r_pipe, theta_mid_outer))

    if params.core_mode == "center_point":
        add_point(Point2D(name="CENTER", x=0.0, y=0.0))
        _build_center_point_inner_patches(patches, params)
        for i in range(params.n_ribs):
            patches.append(
                Patch2D(
                    name=f"outer_rib_{i}_left",
                    corners=(
                        f"RIB_IN_L_{i}",
                        f"RIB_IN_M_{i}",
                        f"RIB_OUT_M_{i}",
                        f"RIB_OUT_L_{i}",
                    ),
                    kind="rib_solid",
                    ni=params.n_tangential_rib_half,
                    nj=params.n_radial_outer,
                )
            )
            patches.append(
                Patch2D(
                    name=f"outer_rib_{i}_right",
                    corners=(
                        f"RIB_IN_M_{i}",
                        f"RIB_IN_R_{i}",
                        f"RIB_OUT_R_{i}",
                        f"RIB_OUT_M_{i}",
                    ),
                    kind="rib_solid",
                    ni=params.n_tangential_rib_half,
                    nj=params.n_radial_outer,
                )
            )

        for i in range(params.n_ribs):
            next_i = (i + 1) % params.n_ribs
            patches.append(
                Patch2D(
                    name=f"outer_gap_{i}_left",
                    corners=(
                        f"RIB_IN_R_{i}",
                        f"GAP_IN_M_{i}",
                        f"GAP_OUT_M_{i}",
                        f"RIB_OUT_R_{i}",
                    ),
                    kind="fluid",
                    ni=params.n_tangential_gap_half,
                    nj=params.n_radial_outer,
                )
            )
            patches.append(
                Patch2D(
                    name=f"outer_gap_{i}_right",
                    corners=(
                        f"GAP_IN_M_{i}",
                        f"RIB_IN_L_{next_i}",
                        f"RIB_OUT_L_{next_i}",
                        f"GAP_OUT_M_{i}",
                    ),
                    kind="fluid",
                    ni=params.n_tangential_gap_half,
                    nj=params.n_radial_outer,
                )
            )
    else:
        _build_annulus_inner_patches(points, patches, params, rib_data)
        for i in range(params.n_ribs):
            next_i = (i + 1) % params.n_ribs
            patches.append(
                Patch2D(
                    name=f"outer_rib_{i}",
                    corners=(f"RIB_IN_L_{i}", f"RIB_IN_R_{i}", f"RIB_OUT_R_{i}", f"RIB_OUT_L_{i}"),
                    kind="rib_solid",
                    ni=2 * params.n_tangential_rib_half,
                    nj=params.n_radial_outer,
                )
            )
            patches.append(
                Patch2D(
                    name=f"outer_gap_{i}",
                    corners=(f"RIB_IN_R_{i}", f"RIB_IN_L_{next_i}", f"RIB_OUT_L_{next_i}", f"RIB_OUT_R_{i}"),
                    kind="fluid",
                    ni=2 * params.n_tangential_gap_half,
                    nj=params.n_radial_outer,
                )
            )

    expected_patch_count = 12 if params.core_mode == "annulus_tube" else 18
    if len(patches) != expected_patch_count:
        raise ValueError(f"Expected {expected_patch_count} patches, got {len(patches)}.")

    return points, patches


def _sample_line(start: Point2D, end: Point2D, samples: int) -> list[tuple[float, float]]:
    if math.isclose(start.x, end.x) and math.isclose(start.y, end.y):
        return [(start.x, start.y)]
    return list(
        zip(
            np.linspace(start.x, end.x, samples, endpoint=False),
            np.linspace(start.y, end.y, samples, endpoint=False),
            strict=True,
        )
    )


def _sample_arc(start: Point2D, end: Point2D, samples: int, direction: str) -> list[tuple[float, float]]:
    theta_start = point_theta(start)
    theta_end = point_theta(end)
    if direction == "ccw":
        theta_end = _unwrap_to_be_after(theta_end, theta_start - 1.0e-12)
    else:
        while theta_end >= theta_start:
            theta_end -= TAU
    thetas = np.linspace(theta_start, theta_end, samples, endpoint=False)
    radius = 0.5 * (point_radius(start) + point_radius(end))
    return [(radius * math.cos(theta), radius * math.sin(theta)) for theta in thetas]


def _sample_inner_rib_arc(start: Point2D, end: Point2D, samples: int) -> list[tuple[float, float]]:
    theta_start = point_theta(start)
    theta_end = point_theta(end)
    delta = ((theta_end - theta_start + math.pi) % TAU) - math.pi
    thetas = np.linspace(theta_start, theta_start + delta, samples, endpoint=False)
    radius = 0.5 * (point_radius(start) + point_radius(end))
    return [(radius * math.cos(theta), radius * math.sin(theta)) for theta in thetas]


def patch_boundary_xy(
    points: dict[str, Point2D],
    patch: Patch2D,
    samples_per_edge: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    corners = list(patch.corners)
    boundary: list[tuple[float, float]] = []
    for idx, start_name in enumerate(corners):
        end_name = corners[(idx + 1) % len(corners)]
        start = points[start_name]
        end = points[end_name]

        if start_name.startswith("RIB_IN_") and end_name.startswith("RIB_IN_"):
            start_idx = start_name.rsplit("_", 1)[1]
            end_idx = end_name.rsplit("_", 1)[1]
            if start_idx == end_idx:
                boundary.extend(_sample_inner_rib_arc(start, end, samples_per_edge))
                continue

        if start_name.startswith("AXIS_") and end_name.startswith("AXIS_"):
            boundary.extend(_sample_arc(start, end, samples_per_edge, direction="ccw"))
            continue

        if idx == 2 and start_name.startswith(("RIB_OUT_", "GAP_OUT_")) and end_name.startswith(("RIB_OUT_", "GAP_OUT_")):
            boundary.extend(_sample_arc(start, end, samples_per_edge, direction="cw"))
            continue

        boundary.extend(_sample_line(start, end, samples_per_edge))

    xs = np.array([xy[0] for xy in boundary], dtype=float)
    ys = np.array([xy[1] for xy in boundary], dtype=float)
    return xs, ys


def _arc_xy(radius: float, theta_a: float, theta_b: float, n: int = 64) -> tuple[np.ndarray, np.ndarray]:
    thetas = np.linspace(theta_a, theta_b, n)
    return radius * np.cos(thetas), radius * np.sin(thetas)


def plot_topology(
    points: dict[str, Point2D],
    patches: list[Patch2D],
    params: HelixMeshParams,
    outfile: str = "debug_cross_section.png",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))

    for radius, style in (
        (params.r_pipe, {"color": "#666666", "linewidth": 1.0, "linestyle": "--"}),
        (params.r_tip, {"color": "#999999", "linewidth": 0.9, "linestyle": ":"}),
    ):
        xs, ys = _arc_xy(radius, 0.0, TAU)
        ax.plot(xs, ys, **style, zorder=0)

    if params.core_mode == "annulus_tube":
        xs, ys = _arc_xy(params.r_inner_tube, 0.0, TAU)
        ax.plot(xs, ys, color="#bbbbbb", linewidth=0.9, linestyle="-.", zorder=0)

    fill_map = {"fluid": "#d8ecff", "rib_solid": "#ffcf99"}

    draw_order = sorted(patches, key=lambda patch: 0 if patch.kind == "fluid" else 1)

    for patch in draw_order:
        xs, ys = patch_boundary_xy(points, patch, samples_per_edge=72)
        corner_xy = [point_xy(points[name]) for name in patch.corners]
        ax.fill(xs, ys, color=fill_map[patch.kind], alpha=0.65, linewidth=0.0, zorder=1)
        ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]), color="#222222", linewidth=1.0, zorder=2)

        x_label = sum(x for x, _ in corner_xy) / 4.0
        y_label = sum(y for _, y in corner_xy) / 4.0
        ax.text(
            x_label,
            y_label,
            patch.name,
            fontsize=8,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            zorder=3,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Helix mesh topology debug plot")
    legend_handles = [
        Line2D([0], [0], color="#666666", linestyle="--", linewidth=1.0, label="r_pipe"),
        Line2D([0], [0], color="#999999", linestyle=":", linewidth=1.0, label="r_tip"),
        LegendPatch(facecolor=fill_map["fluid"], edgecolor="#222222", label="fluid patch"),
        LegendPatch(facecolor=fill_map["rib_solid"], edgecolor="#222222", label="rib_solid patch"),
    ]
    if params.core_mode == "annulus_tube":
        legend_handles.insert(2, Line2D([0], [0], color="#bbbbbb", linestyle="-.", linewidth=1.0, label="r_inner_tube"))
    ax.legend(handles=legend_handles, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.figure.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
