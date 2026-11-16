from .config import DEFAULT_CONFIG_PATH, MeshGeneratorConfig

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "MeshGeneratorConfig",
    "CrossSectionPatch",
    "CrossSectionPoint",
    "SurfacePatchGrid",
    "SurfaceMesh",
    "VolumeElement",
    "VolumeMesh",
    "build_cross_section_topology",
    "build_surface_mesh",
    "build_volume_mesh",
    "plot_cross_section_topology",
    "plot_surface_mesh",
    "plot_volume_preview",
    "summarize_topology",
    "export_to_gmsh",
    "export_to_avl_cgns",
]


def __getattr__(name: str):
    if name in {
        "CrossSectionPatch",
        "CrossSectionPoint",
        "SurfacePatchGrid",
        "SurfaceMesh",
        "VolumeElement",
        "VolumeMesh",
        "build_cross_section_topology",
        "build_surface_mesh",
        "build_volume_mesh",
        "plot_cross_section_topology",
        "plot_surface_mesh",
        "plot_volume_preview",
        "summarize_topology",
    }:
        from . import generator

        return getattr(generator, name)
    if name in {"export_to_gmsh", "export_to_avl_cgns"}:
        from . import export

        return getattr(export, name)
    raise AttributeError(name)
