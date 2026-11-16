from __future__ import annotations

from .compat import legacy_extrude, legacy_mesh2d, legacy_topology
from .config import MeshGeneratorConfig

CrossSectionPoint = legacy_topology.Point2D
CrossSectionPatch = legacy_topology.Patch2D
SurfacePatchGrid = legacy_mesh2d.PatchGrid2D
SurfaceMesh = legacy_mesh2d.Mesh2D
VolumeElement = legacy_extrude.HexElement
VolumeMesh = legacy_extrude.Mesh3D


def _require_helix_geometry(config: MeshGeneratorConfig) -> None:
    if config.geometry_type != "helix_pipe":
        raise NotImplementedError(f"Geometry type {config.geometry_type!r} is not implemented yet.")


def build_cross_section_topology(config: MeshGeneratorConfig) -> tuple[dict[str, CrossSectionPoint], list[CrossSectionPatch]]:
    _require_helix_geometry(config)
    return legacy_topology.build_topology(config)


def summarize_topology(patches: list[CrossSectionPatch]) -> str:
    return legacy_topology.topology_summary(patches)


def plot_cross_section_topology(
    points: dict[str, CrossSectionPoint],
    patches: list[CrossSectionPatch],
    config: MeshGeneratorConfig,
    outfile: str = "debug_cross_section.png",
) -> None:
    _require_helix_geometry(config)
    legacy_topology.plot_topology(points, patches, config, outfile=outfile)


def build_surface_mesh(
    points: dict[str, CrossSectionPoint],
    patches: list[CrossSectionPatch],
) -> SurfaceMesh:
    return legacy_mesh2d.build_mesh2d(points, patches)


def plot_surface_mesh(
    mesh: SurfaceMesh,
    points: dict[str, CrossSectionPoint],
    patches: list[CrossSectionPatch],
    config: MeshGeneratorConfig,
    outfile: str = "debug_cross_section_mesh.png",
) -> None:
    _require_helix_geometry(config)
    legacy_mesh2d.plot_mesh2d(mesh, points, patches, config, outfile=outfile)


def build_volume_mesh(
    surface_mesh: SurfaceMesh,
    config: MeshGeneratorConfig,
) -> VolumeMesh:
    _require_helix_geometry(config)
    return legacy_extrude.build_mesh3d(surface_mesh, config)


def plot_volume_preview(
    volume_mesh: VolumeMesh,
    config: MeshGeneratorConfig,
    outfile: str = "debug_3d_preview.png",
) -> None:
    _require_helix_geometry(config)
    legacy_extrude.plot_3d_preview(volume_mesh, config, outfile=outfile)
