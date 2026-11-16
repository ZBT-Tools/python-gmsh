from __future__ import annotations

from .compat import legacy_export_avl, legacy_export_gmsh
from .config import MeshGeneratorConfig
from .generator import VolumeMesh


def export_to_gmsh(
    volume_mesh: VolumeMesh,
    config: MeshGeneratorConfig,
    outfile: str,
    fluid_only: bool = True,
) -> None:
    legacy_export_gmsh.export_gmsh_discrete(volume_mesh, config, outfile=outfile, fluid_only=fluid_only)


def export_to_avl_cgns(
    volume_mesh: VolumeMesh,
    config: MeshGeneratorConfig,
    outfile: str,
) -> None:
    legacy_export_avl.export_avl_cgns_via_gmsh(
        volume_mesh,
        config,
        outfile=outfile,
        gmsh_exe=config.gmsh_exe,
        keep_intermediate=config.keep_intermediate,
        intermediate_msh=config.intermediate_msh,
        export_solids=not config.fluid_only,
        save_all=config.save_all,
        hex_order=config.hex_order,
        strip_cgns_prefixes=config.strip_cgns_prefixes,
    )
