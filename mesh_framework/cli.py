from __future__ import annotations

import argparse

from .config import DEFAULT_CONFIG_PATH, MeshGeneratorConfig
from .export import export_to_avl_cgns, export_to_gmsh
from .generator import (
    build_cross_section_topology,
    build_surface_mesh,
    build_volume_mesh,
    plot_cross_section_topology,
    plot_surface_mesh,
    plot_volume_preview,
    summarize_topology,
)

DEFAULT_OUTFILES = {
    "plot_2d": "debug_cross_section.png",
    "plot_2d_mesh": "debug_cross_section_mesh.png",
    "plot_3d_preview": "debug_3d_preview.png",
    "export_gmsh": "structured_geometry.msh",
    "export_gmsh_full": "structured_geometry_full.msh",
    "export_avl_cgns": "geometry_multidomain.cgns",
}


def _resolve_output_mode(args: argparse.Namespace, config: MeshGeneratorConfig) -> str:
    selections = {
        "plot_2d": args.plot_2d,
        "plot_2d_mesh": args.plot_2d_mesh,
        "plot_3d_preview": args.plot_3d_preview,
        "export_gmsh": args.export_gmsh is not None,
        "export_gmsh_full": args.export_gmsh_full is not None,
        "export_avl_cgns": args.export_avl_cgns is not None,
    }
    chosen = [mode for mode, enabled in selections.items() if enabled]
    if len(chosen) > 1:
        raise ValueError(
            "Choose exactly one of --plot-2d, --plot-2d-mesh, --plot-3d-preview, "
            "--export-gmsh, --export-gmsh-full, or --export-avl-cgns."
        )
    if len(chosen) == 1:
        return chosen[0]
    return config.output_mode


def _resolve_outfile(mode: str, args: argparse.Namespace, config: MeshGeneratorConfig) -> str:
    if mode == "export_gmsh" and args.export_gmsh is not None:
        return args.export_gmsh
    if mode == "export_gmsh_full" and args.export_gmsh_full is not None:
        return args.export_gmsh_full
    if mode == "export_avl_cgns" and args.export_avl_cgns is not None:
        return args.export_avl_cgns
    if args.outfile:
        return args.outfile
    if config.outfile:
        return config.outfile
    return DEFAULT_OUTFILES[mode]


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured geometry mesh utility.")
    parser.add_argument("--config", "--params", dest="config_path", default=str(DEFAULT_CONFIG_PATH), help="Path to the JSON configuration file.")
    parser.add_argument("--plot-2d", action="store_true", help="Write the 2D topology debug plot.")
    parser.add_argument("--plot-2d-mesh", action="store_true", help="Write the structured 2D mesh debug plot.")
    parser.add_argument("--plot-3d-preview", action="store_true", help="Write the twisted 3D mesh preview plot.")
    parser.add_argument("--export-gmsh", metavar="OUTFILE", default=None, help="Write the fluid-only discrete Gmsh mesh.")
    parser.add_argument("--export-gmsh-full", metavar="OUTFILE", default=None, help="Write the full fluid+solid discrete Gmsh mesh.")
    parser.add_argument("--export-avl-cgns", metavar="OUTFILE", default=None, help="Write the AVL-oriented CGNS multi-domain mesh.")
    parser.add_argument("--outfile", default=None, help="Generic output path override. If omitted, the value from the JSON configuration is used.")
    parser.add_argument("--geometry-type", default=None, help="Geometry type selector. Currently supported: helix_pipe.")
    parser.add_argument("--fluid-only", action="store_true", help="For AVL export only: omit the solid volume and export the interface as a wall.")
    parser.add_argument("--gmsh-exe", default=None, help="Path to the Gmsh executable used for CGNS conversion fallback.")
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep the intermediate Gmsh .msh file during AVL export.")
    parser.add_argument("--intermediate-msh", default=None, help="Explicit path for the intermediate Gmsh .msh file during AVL export.")
    parser.add_argument("--save-all", action="store_true", help="Pass -save_all to the Gmsh CGNS conversion step.")
    parser.add_argument("--hex-order", choices=("default", "flip_i", "flip_k", "reverse"), default=None, help="Export-time hex node ordering used for AVL CGNS conversion.")
    parser.add_argument("--strip-cgns-prefixes", action="store_true", help="Experimental CGNS post-processing. Disabled by default.")
    parser.add_argument("--length", type=float, default=None, help="Override pipe length.")
    parser.add_argument("--nz", type=int, default=None, help="Override the number of axial layers.")
    parser.add_argument("--pitch-angle-deg", type=float, default=None, help="Override the helix pitch angle in degrees.")
    parser.add_argument("--core-mode", choices=["center_point", "annulus_tube"], default=None, help="Choose the inner topology mode.")
    parser.add_argument("--r-inner-tube", type=float, default=None, help="Override the free inner tube radius for annulus_tube mode.")
    parser.add_argument("--r-pipe", type=float, default=None, help="Override the pipe radius.")
    parser.add_argument("--rib-height", type=float, default=None, help="Override the rib height.")
    parser.add_argument("--rib-width-outer", type=float, default=None, help="Override the outer rib width distance in meters.")
    parser.add_argument("--rib-width-inner", type=float, default=None, help="Override the inner rib width distance in meters.")
    args = parser.parse_args()

    config = MeshGeneratorConfig.from_json(args.config_path)
    if args.geometry_type is not None:
        config.geometry_type = args.geometry_type
    if args.length is not None:
        config.length = args.length
    if args.nz is not None:
        config.n_z = args.nz
    if args.pitch_angle_deg is not None:
        config.pitch_angle_deg = args.pitch_angle_deg
    if args.core_mode is not None:
        config.core_mode = args.core_mode
    if args.r_inner_tube is not None:
        config.r_inner_tube = args.r_inner_tube
    if args.r_pipe is not None:
        config.r_pipe = args.r_pipe
    if args.rib_height is not None:
        config.rib_height = args.rib_height
    if args.rib_width_outer is not None:
        config.rib_width_outer = args.rib_width_outer
    if args.rib_width_inner is not None:
        config.rib_width_inner = args.rib_width_inner
    if args.gmsh_exe is not None:
        config.gmsh_exe = args.gmsh_exe
    if args.keep_intermediate:
        config.keep_intermediate = True
    if args.intermediate_msh is not None:
        config.intermediate_msh = args.intermediate_msh
    if args.save_all:
        config.save_all = True
    if args.hex_order is not None:
        config.hex_order = args.hex_order
    if args.fluid_only:
        config.fluid_only = True
    if args.strip_cgns_prefixes:
        config.strip_cgns_prefixes = True

    try:
        mode = _resolve_output_mode(args, config)
    except ValueError as exc:
        parser.error(str(exc))
    config.validate()

    points, patches = build_cross_section_topology(config)
    print(summarize_topology(patches))

    if mode == "plot_2d":
        plot_cross_section_topology(points, patches, config, outfile=_resolve_outfile(mode, args, config))
        return

    surface_mesh = build_surface_mesh(points, patches)
    if mode == "plot_2d_mesh":
        plot_surface_mesh(surface_mesh, points, patches, config, outfile=_resolve_outfile(mode, args, config))
        return

    volume_mesh = build_volume_mesh(surface_mesh, config)
    if mode == "plot_3d_preview":
        plot_volume_preview(volume_mesh, config, outfile=_resolve_outfile(mode, args, config))
    elif mode == "export_gmsh":
        export_to_gmsh(volume_mesh, config, outfile=_resolve_outfile(mode, args, config), fluid_only=True)
    elif mode == "export_gmsh_full":
        export_to_gmsh(volume_mesh, config, outfile=_resolve_outfile(mode, args, config), fluid_only=False)
    else:
        export_to_avl_cgns(volume_mesh, config, outfile=_resolve_outfile(mode, args, config))
