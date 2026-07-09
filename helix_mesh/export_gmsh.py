from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import gmsh

from extrude import HexElement, Mesh3D
from params import HelixMeshParams

HEX_FACE_NODE_IDS: dict[str, tuple[int, int, int, int]] = {
    "bottom": (0, 3, 2, 1),
    "top": (4, 5, 6, 7),
    "side_i0": (0, 4, 7, 3),
    "side_i1": (1, 2, 6, 5),
    "side_j0": (0, 1, 5, 4),
    "side_j1": (3, 7, 6, 2),
}

BOUNDARY_NAMES = ("inlet", "outlet", "outer_wall", "rib_wall", "axis_core")
FULL_BOUNDARY_NAMES = (
    "inlet_gas",
    "inlet_rib",
    "outlet_gas",
    "outlet_rib",
    "reactor_wall",
    "rib_wall",
    "axis_core",
    "fluid_solid_interface",
)
HEX_ORDER_MODES = ("default", "flip_i", "flip_k", "reverse")

FLUID_VOLUME_EXPORT_NAME = "DOM_GasVolume"
FULL_VOLUME_EXPORT_NAMES = {
    "fluid": "DOM_GasVolume",
    "rib_solid": "DOM_RibVolume",
}
FLUID_BOUNDARY_EXPORT_NAMES = {
    "inlet": "BND_I_Inlet_Gas",
    "outlet": "BND_O_Outlet_Gas",
    "outer_wall": "BND_W_ReactorWall",
    "rib_wall": "BND_W_RibWall",
    "axis_core": "axis_core",
}
FULL_BOUNDARY_EXPORT_NAMES = {
    "inlet_gas": "BND_I_Inlet_Gas",
    "inlet_rib": "BND_I_Inlet_Rib",
    "outlet_gas": "BND_O_Outlet_Gas",
    "outlet_rib": "BND_O_Outlet_Rib",
    "reactor_wall": "BND_W_ReactorWall",
    "rib_wall": "BND_W_RibWall",
    "axis_core": "axis_core",
}


@dataclass(frozen=True)
class FaceOwner:
    node_ids: tuple[int, int, int, int]
    patch_name: str
    kind: str
    layer: int
    face_name: str
    hex_index: int


@dataclass(frozen=True)
class BoundaryFace:
    node_ids: tuple[int, int, int, int]
    patch_name: str
    boundary_name: str
    owner_kind: str
    neighbor_kind: str | None


@dataclass(frozen=True)
class GmshExportData:
    fluid_hexes: list[HexElement]
    boundary_faces: dict[str, list[BoundaryFace]]
    used_node_ids: list[int]


@dataclass(frozen=True)
class GmshFullExportData:
    volume_hexes: dict[str, list[HexElement]]
    boundary_faces: dict[str, list[BoundaryFace]]
    used_node_ids: list[int]


def _assign_entity_and_physical_name(gmsh_module, dim: int, entity_tag: int, physical_tag: int, name: str) -> None:
    gmsh_module.model.setEntityName(dim, entity_tag, name)
    gmsh_module.model.setPhysicalName(dim, physical_tag, name)


def reorder_hex_for_export(node_ids: tuple[int, ...], mode: str) -> tuple[int, ...]:
    n0, n1, n2, n3, n4, n5, n6, n7 = node_ids

    if mode == "default":
        return (n0, n1, n2, n3, n4, n5, n6, n7)
    if mode == "flip_i":
        return (n0, n3, n2, n1, n4, n7, n6, n5)
    if mode == "flip_k":
        return (n4, n5, n6, n7, n0, n1, n2, n3)
    if mode == "reverse":
        return (n3, n2, n1, n0, n7, n6, n5, n4)
    raise ValueError(f"Unknown hex export order: {mode}")


def collect_face_adjacency(mesh3d: Mesh3D) -> dict[tuple[int, int, int, int], list[FaceOwner]]:
    adjacency: dict[tuple[int, int, int, int], list[FaceOwner]] = {}

    for hex_index, hex_element in enumerate(mesh3d.hexes):
        for face_name, local_indices in HEX_FACE_NODE_IDS.items():
            face_node_ids = tuple(hex_element.node_ids[idx] for idx in local_indices)
            face_key = tuple(sorted(face_node_ids))
            adjacency.setdefault(face_key, []).append(
                FaceOwner(
                    node_ids=face_node_ids,
                    patch_name=hex_element.patch_name,
                    kind=hex_element.kind,
                    layer=hex_element.layer,
                    face_name=face_name,
                    hex_index=hex_index,
                )
            )

    return adjacency


def _face_average_radius(mesh3d: Mesh3D, face_node_ids: tuple[int, int, int, int]) -> float:
    radii = [math.hypot(mesh3d.nodes[node_id][0], mesh3d.nodes[node_id][1]) for node_id in face_node_ids]
    return sum(radii) / len(radii)


def _is_outer_wall_face(owner: FaceOwner, mesh3d: Mesh3D, params: HelixMeshParams, radius_tol: float = 1.0e-8) -> bool:
    if owner.patch_name.startswith("outer_gap_") and owner.face_name == "side_j1":
        return True

    average_radius = _face_average_radius(mesh3d, owner.node_ids)
    return math.isclose(average_radius, params.r_pipe, rel_tol=0.0, abs_tol=radius_tol)


def _is_axis_core_face(owner: FaceOwner, mesh3d: Mesh3D, params: HelixMeshParams, radius_tol: float = 1.0e-8) -> bool:
    if not owner.patch_name.startswith("inner_"):
        return False

    if owner.face_name not in {"side_i0", "side_j0"}:
        return False

    average_radius = _face_average_radius(mesh3d, owner.node_ids)
    target_radius = params.inner_boundary_radius
    if params.core_mode == "annulus_tube":
        return math.isclose(average_radius, target_radius, rel_tol=0.0, abs_tol=radius_tol)
    return average_radius <= max(target_radius + radius_tol, radius_tol)


def classify_fluid_boundary_faces(
    mesh3d: Mesh3D,
    params: HelixMeshParams,
) -> dict[str, list[tuple[int, int, int, int]]]:
    classified = _classify_fluid_boundary_faces_detailed(mesh3d, params)
    return {name: [face.node_ids for face in faces] for name, faces in classified.items()}


def _classify_fluid_boundary_faces_detailed(
    mesh3d: Mesh3D,
    params: HelixMeshParams,
) -> dict[str, list[BoundaryFace]]:
    boundary_faces: dict[str, list[BoundaryFace]] = {name: [] for name in BOUNDARY_NAMES}
    adjacency = collect_face_adjacency(mesh3d)

    for owners in adjacency.values():
        if len(owners) > 2:
            raise ValueError("Non-manifold face detected in 3D mesh adjacency.")

        fluid_owners = [owner for owner in owners if owner.kind == "fluid"]
        if not fluid_owners:
            continue
        if len(fluid_owners) == 2:
            continue

        owner = fluid_owners[0]
        neighbors = [candidate for candidate in owners if candidate.hex_index != owner.hex_index]
        neighbor_kind = neighbors[0].kind if neighbors else None

        if owner.face_name == "bottom" and owner.layer == 0:
            boundary_name = "inlet"
        elif owner.face_name == "top" and owner.layer == mesh3d.n_z - 1:
            boundary_name = "outlet"
        elif neighbor_kind == "rib_solid":
            boundary_name = "rib_wall"
        elif neighbor_kind == "fluid":
            continue
        elif neighbor_kind is None and _is_outer_wall_face(owner, mesh3d, params):
            boundary_name = "outer_wall"
        elif neighbor_kind is None and _is_axis_core_face(owner, mesh3d, params):
            boundary_name = "axis_core"
        else:
            raise ValueError(
                "Encountered an unclassified exposed fluid face on "
                f"patch {owner.patch_name}, layer {owner.layer}, face {owner.face_name}."
            )

        boundary_faces[boundary_name].append(
            BoundaryFace(
                node_ids=owner.node_ids,
                patch_name=owner.patch_name,
                boundary_name=boundary_name,
                owner_kind=owner.kind,
                neighbor_kind=neighbor_kind,
            )
        )

    return boundary_faces


def build_gmsh_export_data(
    mesh3d: Mesh3D,
    params: HelixMeshParams,
    fluid_only: bool = True,
) -> GmshExportData | GmshFullExportData:
    if not fluid_only:
        return build_gmsh_full_export_data(mesh3d, params)

    fluid_hexes = [hex_element for hex_element in mesh3d.hexes if hex_element.kind == "fluid"]
    boundary_faces = _classify_fluid_boundary_faces_detailed(mesh3d, params)

    used_node_ids: set[int] = set()
    for hex_element in fluid_hexes:
        used_node_ids.update(hex_element.node_ids)
    for faces in boundary_faces.values():
        for face in faces:
            used_node_ids.update(face.node_ids)

    return GmshExportData(
        fluid_hexes=fluid_hexes,
        boundary_faces=boundary_faces,
        used_node_ids=sorted(used_node_ids),
    )


def build_gmsh_full_export_data(
    mesh3d: Mesh3D,
    params: HelixMeshParams,
) -> GmshFullExportData:
    adjacency = collect_face_adjacency(mesh3d)
    boundary_faces: dict[str, list[BoundaryFace]] = {name: [] for name in FULL_BOUNDARY_NAMES}
    volume_hexes = {
        "fluid": [hex_element for hex_element in mesh3d.hexes if hex_element.kind == "fluid"],
        "rib_solid": [hex_element for hex_element in mesh3d.hexes if hex_element.kind == "rib_solid"],
    }

    for owners in adjacency.values():
        if len(owners) > 2:
            raise ValueError("Non-manifold face detected in 3D mesh adjacency.")

        if len(owners) == 2 and {owner.kind for owner in owners} == {"fluid", "rib_solid"}:
            fluid_owner = next(owner for owner in owners if owner.kind == "fluid")
            boundary_faces["fluid_solid_interface"].append(
                BoundaryFace(
                    node_ids=fluid_owner.node_ids,
                    patch_name=fluid_owner.patch_name,
                    boundary_name="fluid_solid_interface",
                    owner_kind="fluid",
                    neighbor_kind="rib_solid",
                )
            )
            continue

        if len(owners) != 1:
            continue

        owner = owners[0]

        if owner.face_name == "bottom" and owner.layer == 0:
            boundary_name = "inlet_gas" if owner.kind == "fluid" else "inlet_rib"
        elif owner.face_name == "top" and owner.layer == mesh3d.n_z - 1:
            boundary_name = "outlet_gas" if owner.kind == "fluid" else "outlet_rib"
        elif _is_outer_wall_face(owner, mesh3d, params):
            boundary_name = "reactor_wall" if owner.kind == "fluid" else "rib_wall"
        elif _is_axis_core_face(owner, mesh3d, params):
            boundary_name = "axis_core"
        else:
            raise ValueError(
                "Encountered an unclassified exposed face in full export on "
                f"patch {owner.patch_name}, layer {owner.layer}, face {owner.face_name}."
            )

        boundary_faces[boundary_name].append(
            BoundaryFace(
                node_ids=owner.node_ids,
                patch_name=owner.patch_name,
                boundary_name=boundary_name,
                owner_kind=owner.kind,
                neighbor_kind=None,
            )
        )

    used_node_ids: set[int] = set()
    for hexes in volume_hexes.values():
        for hex_element in hexes:
            used_node_ids.update(hex_element.node_ids)
    for faces in boundary_faces.values():
        for face in faces:
            used_node_ids.update(face.node_ids)

    return GmshFullExportData(
        volume_hexes=volume_hexes,
        boundary_faces=boundary_faces,
        used_node_ids=sorted(used_node_ids),
    )


def export_gmsh_discrete(
    mesh3d: Mesh3D,
    params: HelixMeshParams,
    outfile: str,
    fluid_only: bool = True,
    hex_order: str = "default",
) -> None:
    export_data = build_gmsh_export_data(mesh3d, params, fluid_only=fluid_only)
    outpath = Path(outfile)
    if outpath.parent != Path("."):
        outpath.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        populate_gmsh_model(gmsh, mesh3d, params, export_data, fluid_only=fluid_only, hex_order=hex_order, model_name="helix_mesh")
        gmsh.write(str(outpath))
    finally:
        gmsh.finalize()


def populate_gmsh_model(
    gmsh_module,
    mesh3d: Mesh3D,
    params: HelixMeshParams,
    export_data: GmshExportData | GmshFullExportData | None = None,
    fluid_only: bool = True,
    hex_order: str = "default",
    model_name: str = "helix_mesh",
) -> None:
    if export_data is None:
        export_data = build_gmsh_export_data(mesh3d, params, fluid_only=fluid_only)

    node_id_map = {old_id: new_id for new_id, old_id in enumerate(export_data.used_node_ids, start=1)}
    coordinates: list[float] = []
    for old_node_id in export_data.used_node_ids:
        coordinates.extend(mesh3d.nodes[old_node_id])

    gmsh_module.model.add(model_name)
    gmsh_module.option.setNumber("Mesh.MshFileVersion", params.msh_version)
    gmsh_module.option.setNumber("Mesh.Binary", 0)
    node_tags = list(range(1, len(export_data.used_node_ids) + 1))
    if fluid_only:
        _export_fluid_only_mesh(gmsh_module, export_data, node_id_map, node_tags, coordinates, hex_order=hex_order)
    else:
        _export_full_mesh(gmsh_module, export_data, node_id_map, node_tags, coordinates, hex_order=hex_order)


def _export_fluid_only_mesh(
    gmsh_module,
    export_data: GmshExportData,
    node_id_map: dict[int, int],
    node_tags: list[int],
    coordinates: list[float],
    hex_order: str,
) -> None:
    surface_tags: dict[str, int] = {}
    for boundary_name in BOUNDARY_NAMES:
        surface_tags[boundary_name] = gmsh_module.model.addDiscreteEntity(2)

    volume_tag = gmsh_module.model.addDiscreteEntity(3, -1, list(surface_tags.values()))
    gmsh_module.model.mesh.addNodes(3, volume_tag, node_tags, coordinates)

    volume_element_tags = list(range(1, len(export_data.fluid_hexes) + 1))
    volume_node_tags = [
        node_id_map[node_id]
        for hex_element in export_data.fluid_hexes
        for node_id in reorder_hex_for_export(hex_element.node_ids, hex_order)
    ]
    gmsh_module.model.mesh.addElements(3, volume_tag, [5], [volume_element_tags], [volume_node_tags])

    next_element_tag = len(volume_element_tags) + 1
    for boundary_name in BOUNDARY_NAMES:
        if export_data.boundary_faces[boundary_name]:
            element_tags = list(range(next_element_tag, next_element_tag + len(export_data.boundary_faces[boundary_name])))
            next_element_tag += len(element_tags)
            element_node_tags = [
                node_id_map[node_id]
                for face in export_data.boundary_faces[boundary_name]
                for node_id in face.node_ids
            ]
            gmsh_module.model.mesh.addElements(2, surface_tags[boundary_name], [3], [element_tags], [element_node_tags])

    gmsh_module.model.mesh.reclassifyNodes()

    physical_volume = gmsh_module.model.addPhysicalGroup(3, [volume_tag])
    _assign_entity_and_physical_name(gmsh_module, 3, volume_tag, physical_volume, FLUID_VOLUME_EXPORT_NAME)

    for boundary_name in BOUNDARY_NAMES:
        physical_surface = gmsh_module.model.addPhysicalGroup(2, [surface_tags[boundary_name]])
        _assign_entity_and_physical_name(
            gmsh_module,
            2,
            surface_tags[boundary_name],
            physical_surface,
            FLUID_BOUNDARY_EXPORT_NAMES[boundary_name],
        )


def _export_full_mesh(
    gmsh_module,
    export_data: GmshFullExportData,
    node_id_map: dict[int, int],
    node_tags: list[int],
    coordinates: list[float],
    hex_order: str,
) -> None:
    surface_tags: dict[str, int] = {}
    for boundary_name in FULL_BOUNDARY_NAMES:
        surface_tags[boundary_name] = gmsh_module.model.addDiscreteEntity(2)

    fluid_surface_names = ["inlet_gas", "outlet_gas", "reactor_wall", "axis_core", "fluid_solid_interface"]
    rib_surface_names = ["inlet_rib", "outlet_rib", "rib_wall", "fluid_solid_interface"]

    fluid_volume_tag = gmsh_module.model.addDiscreteEntity(3, -1, [surface_tags[name] for name in fluid_surface_names])
    rib_volume_tag = gmsh_module.model.addDiscreteEntity(3, -1, [surface_tags[name] for name in rib_surface_names])
    gmsh_module.model.mesh.addNodes(3, fluid_volume_tag, node_tags, coordinates)

    next_element_tag = 1
    volume_tags = {"fluid": fluid_volume_tag, "rib_solid": rib_volume_tag}
    for kind in ("fluid", "rib_solid"):
        hexes = export_data.volume_hexes[kind]
        element_tags = list(range(next_element_tag, next_element_tag + len(hexes)))
        next_element_tag += len(element_tags)
        element_node_tags = [
            node_id_map[node_id]
            for hex_element in hexes
            for node_id in reorder_hex_for_export(hex_element.node_ids, hex_order)
        ]
        gmsh_module.model.mesh.addElements(3, volume_tags[kind], [5], [element_tags], [element_node_tags])

    for boundary_name in FULL_BOUNDARY_NAMES:
        if export_data.boundary_faces[boundary_name]:
            element_tags = list(range(next_element_tag, next_element_tag + len(export_data.boundary_faces[boundary_name])))
            next_element_tag += len(element_tags)
            element_node_tags = [
                node_id_map[node_id]
                for face in export_data.boundary_faces[boundary_name]
                for node_id in face.node_ids
            ]
            gmsh_module.model.mesh.addElements(2, surface_tags[boundary_name], [3], [element_tags], [element_node_tags])

    gmsh_module.model.mesh.reclassifyNodes()

    physical_fluid = gmsh_module.model.addPhysicalGroup(3, [fluid_volume_tag])
    _assign_entity_and_physical_name(
        gmsh_module,
        3,
        fluid_volume_tag,
        physical_fluid,
        FULL_VOLUME_EXPORT_NAMES["fluid"],
    )
    physical_rib = gmsh_module.model.addPhysicalGroup(3, [rib_volume_tag])
    _assign_entity_and_physical_name(
        gmsh_module,
        3,
        rib_volume_tag,
        physical_rib,
        FULL_VOLUME_EXPORT_NAMES["rib_solid"],
    )
    gmsh_module.model.setEntityName(2, surface_tags["fluid_solid_interface"], "fluid_solid_interface")

    for boundary_name in FULL_BOUNDARY_NAMES:
        if boundary_name == "fluid_solid_interface":
            continue
        physical_surface = gmsh_module.model.addPhysicalGroup(2, [surface_tags[boundary_name]])
        _assign_entity_and_physical_name(
            gmsh_module,
            2,
            surface_tags[boundary_name],
            physical_surface,
            FULL_BOUNDARY_EXPORT_NAMES[boundary_name],
        )
