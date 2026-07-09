from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from mesh2d import Mesh2D, PatchGrid2D
from params import HelixMeshParams


@dataclass
class HexElement:
    node_ids: tuple[int, int, int, int, int, int, int, int]
    patch_name: str
    kind: str
    layer: int
    quad_index: int


@dataclass
class Mesh3D:
    nodes: dict[int, tuple[float, float, float]]
    hexes: list[HexElement]
    node_layers: dict[tuple[int, int], int]
    nodes2d: dict[int, tuple[float, float]]
    patch_grids2d: dict[str, PatchGrid2D]
    n_z: int
    length: float


def twist_point(
    x: float,
    y: float,
    z: float,
    phi: float,
) -> tuple[float, float, float]:
    return (
        math.cos(phi) * x - math.sin(phi) * y,
        math.sin(phi) * x + math.cos(phi) * y,
        z,
    )


def tetra_signed_volume(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
    d: tuple[float, float, float],
) -> float:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    c_arr = np.asarray(c, dtype=float)
    d_arr = np.asarray(d, dtype=float)
    return float(np.dot(b_arr - a_arr, np.cross(c_arr - a_arr, d_arr - a_arr)) / 6.0)


def hex_signed_volume(nodes_xyz: list[tuple[float, float, float]]) -> float:
    if len(nodes_xyz) != 8:
        raise ValueError("hex_signed_volume requires exactly 8 node coordinates.")

    tets = (
        (0, 1, 3, 4),
        (1, 2, 3, 6),
        (1, 3, 4, 6),
        (1, 4, 5, 6),
        (3, 4, 6, 7),
    )
    return sum(tetra_signed_volume(nodes_xyz[i], nodes_xyz[j], nodes_xyz[k], nodes_xyz[l]) for i, j, k, l in tets)


def build_mesh3d(
    mesh2d: Mesh2D,
    params: HelixMeshParams,
) -> Mesh3D:
    params.validate()

    nodes: dict[int, tuple[float, float, float]] = {}
    node_layers: dict[tuple[int, int], int] = {}
    next_node_id = 1

    for layer in range(params.n_z + 1):
        z = params.length * layer / params.n_z
        phi = params.total_twist * layer / params.n_z
        for node2d_id in sorted(mesh2d.nodes):
            x2, y2 = mesh2d.nodes[node2d_id]
            nodes[next_node_id] = twist_point(x2, y2, z, phi)
            node_layers[(node2d_id, layer)] = next_node_id
            next_node_id += 1

    hexes: list[HexElement] = []
    orientation_sign: int | None = None

    for layer in range(params.n_z):
        for quad_index, quad in enumerate(mesh2d.quads):
            n00, n10, n11, n01, patch_name, kind = quad
            node_ids = (
                node_layers[(n00, layer)],
                node_layers[(n10, layer)],
                node_layers[(n11, layer)],
                node_layers[(n01, layer)],
                node_layers[(n00, layer + 1)],
                node_layers[(n10, layer + 1)],
                node_layers[(n11, layer + 1)],
                node_layers[(n01, layer + 1)],
            )
            if len(set(node_ids)) != 8:
                raise ValueError(f"Hex for patch {patch_name} at layer {layer} has duplicate node IDs.")

            hex_xyz = [nodes[node_id] for node_id in node_ids]
            signed_volume = hex_signed_volume(hex_xyz)
            if math.isclose(signed_volume, 0.0, abs_tol=1.0e-18):
                raise ValueError(f"Hex for patch {patch_name} at layer {layer} has near-zero volume.")

            current_sign = 1 if signed_volume > 0.0 else -1
            if orientation_sign is None:
                orientation_sign = current_sign
            elif current_sign != orientation_sign:
                raise ValueError(f"Mixed hex orientation detected at patch {patch_name}, layer {layer}.")

            hexes.append(
                HexElement(
                    node_ids=node_ids,
                    patch_name=patch_name,
                    kind=kind,
                    layer=layer,
                    quad_index=quad_index,
                )
            )

    return Mesh3D(
        nodes=nodes,
        hexes=hexes,
        node_layers=node_layers,
        nodes2d=dict(mesh2d.nodes),
        patch_grids2d=dict(mesh2d.patch_grids),
        n_z=params.n_z,
        length=params.length,
    )


def plot_3d_preview(
    mesh3d: Mesh3D,
    params: HelixMeshParams,
    outfile: str = "debug_3d_preview.png",
    max_edges: int | None = None,
) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    edge_color = {"fluid": "#1f5a88", "rib_solid": "#b56b00"}
    face_color = {"fluid": "#d8ecff", "rib_solid": "#ffcf99"}
    all_xyz: list[tuple[float, float, float]] = []

    selected_layers = sorted(
        {
            0,
            mesh3d.n_z,
            mesh3d.n_z // 4,
            mesh3d.n_z // 2,
            (3 * mesh3d.n_z) // 4,
        }
    )

    outer_surface_quads: dict[str, list[list[tuple[float, float, float]]]] = {"fluid": [], "rib_solid": []}
    outer_node_ids: set[int] = set()

    for patch_grid in mesh3d.patch_grids2d.values():
        node_ids_2d = patch_grid.node_ids
        for node2d_id in node_ids_2d[:, -1]:
            outer_node_ids.add(int(node2d_id))

        for layer in range(mesh3d.n_z):
            for i in range(node_ids_2d.shape[0] - 1):
                n00 = mesh3d.node_layers[(int(node_ids_2d[i, -1]), layer)]
                n10 = mesh3d.node_layers[(int(node_ids_2d[i + 1, -1]), layer)]
                n11 = mesh3d.node_layers[(int(node_ids_2d[i + 1, -1]), layer + 1)]
                n01 = mesh3d.node_layers[(int(node_ids_2d[i, -1]), layer + 1)]
                quad_xyz = [mesh3d.nodes[n00], mesh3d.nodes[n10], mesh3d.nodes[n11], mesh3d.nodes[n01]]
                outer_surface_quads[patch_grid.kind].append(quad_xyz)
                all_xyz.extend(quad_xyz)

    for kind, quads in outer_surface_quads.items():
        if not quads:
            continue
        collection = Poly3DCollection(
            quads,
            facecolors=face_color[kind],
            edgecolors=edge_color[kind],
            linewidths=0.12,
            alpha=0.18,
        )
        ax.add_collection3d(collection)

    for patch_grid in mesh3d.patch_grids2d.values():
        color = edge_color[patch_grid.kind]
        node_ids_2d = patch_grid.node_ids

        for layer in selected_layers:
            for j in range(node_ids_2d.shape[1]):
                coords = np.array(
                    [mesh3d.nodes[mesh3d.node_layers[(int(node_ids_2d[i, j]), layer)]] for i in range(node_ids_2d.shape[0])],
                    dtype=float,
                )
                all_xyz.extend(tuple(row) for row in coords)
                ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=0.65, alpha=0.8)

            for i in range(node_ids_2d.shape[0]):
                coords = np.array(
                    [mesh3d.nodes[mesh3d.node_layers[(int(node_ids_2d[i, j]), layer)]] for j in range(node_ids_2d.shape[1])],
                    dtype=float,
                )
                all_xyz.extend(tuple(row) for row in coords)
                ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=0.65, alpha=0.8)

    for node2d_id in sorted(outer_node_ids):
        coords = np.array(
            [mesh3d.nodes[mesh3d.node_layers[(node2d_id, layer)]] for layer in range(mesh3d.n_z + 1)],
            dtype=float,
        )
        radius = math.hypot(mesh3d.nodes2d[node2d_id][0], mesh3d.nodes2d[node2d_id][1])
        kind = "fluid" if math.isclose(radius, params.r_pipe, rel_tol=0.0, abs_tol=1.0e-12) else "fluid"
        all_xyz.extend(tuple(row) for row in coords)
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=edge_color[kind], linewidth=0.5, alpha=0.55)

    if all_xyz:
        xyz = np.asarray(all_xyz, dtype=float)
        mins = xyz.min(axis=0)
        maxs = xyz.max(axis=0)
        spans = np.maximum(maxs - mins, 1.0e-12)
        padding = np.maximum(0.04 * spans, 5.0e-4)
        ax.set_xlim(mins[0] - padding[0], maxs[0] + padding[0])
        ax.set_ylim(mins[1] - padding[1], maxs[1] + padding[1])
        ax.set_zlim(mins[2] - padding[2], maxs[2] + padding[2])
        ax.set_box_aspect(tuple(spans))

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Helix mesh 3D preview")
    ax.view_init(elev=22, azim=-58)
    ax.legend(
        handles=[
            Line2D([0], [0], color=edge_color["fluid"], label="fluid"),
            Line2D([0], [0], color=edge_color["rib_solid"], label="rib_solid"),
        ],
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
