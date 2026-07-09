from __future__ import annotations

import argparse
from pathlib import Path

from extrude import build_mesh3d, plot_3d_preview
from export_avl import export_avl_cgns_via_gmsh
from export_gmsh import HEX_ORDER_MODES, export_gmsh_discrete
from params import HelixMeshParams
from mesh2d import build_mesh2d, plot_mesh2d
from topology import build_topology, plot_topology, topology_summary


DEFAULT_PARAMS_PATH = Path(__file__).resolve().with_name("parameters.json")
DEFAULT_OUTFILES = {
    "plot_2d": "debug_cross_section.png",
    "plot_2d_mesh": "debug_cross_section_mesh.png",
    "plot_3d_preview": "debug_3d_preview.png",
    "export_gmsh": "helix_pipe_structured.msh",
    "export_gmsh_full": "helix_pipe_structured_full.msh",
    "export_avl_cgns": "helix_pipe_multidomain.cgns",
}


def _resolve_output_mode(args: argparse.Namespace, params: HelixMeshParams) -> str:
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
    return params.output_mode


def _resolve_outfile(mode: str, args: argparse.Namespace, params: HelixMeshParams) -> str:
    if mode == "export_gmsh" and args.export_gmsh is not None:
        return args.export_gmsh
    if mode == "export_gmsh_full" and args.export_gmsh_full is not None:
        return args.export_gmsh_full
    if mode == "export_avl_cgns" and args.export_avl_cgns is not None:
        return args.export_avl_cgns
    if args.outfile:
        return args.outfile
    if params.outfile:
        return params.outfile
    return DEFAULT_OUTFILES[mode]


def main() -> None:
    parser = argparse.ArgumentParser(description="Helix mesh debug utility.")
    parser.add_argument("--params", default=str(DEFAULT_PARAMS_PATH), help="Path to the JSON parameter file.")
    parser.add_argument("--plot-2d", action="store_true", help="Write the 2D topology debug plot.")
    parser.add_argument("--plot-2d-mesh", action="store_true", help="Write the structured 2D mesh debug plot.")
    parser.add_argument("--plot-3d-preview", action="store_true", help="Write the twisted 3D mesh preview plot.")
    parser.add_argument("--export-gmsh", metavar="OUTFILE", default=None, help="Write the fluid-only discrete Gmsh mesh.")
    parser.add_argument("--export-gmsh-full", metavar="OUTFILE", default=None, help="Write the full fluid+rib_solid discrete Gmsh mesh.")
    parser.add_argument("--export-avl-cgns", metavar="OUTFILE", default=None, help="Write the AVL-oriented CGNS multi-domain mesh.")
    parser.add_argument("--fluid-only", action="store_true", help="For AVL export only: omit rib_solid volume and export the interface as rib_wall.")
    parser.add_argument("--gmsh-exe", default=None, help="Path to the Gmsh executable used for CGNS conversion.")
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep the intermediate Gmsh .msh file during AVL export.")
    parser.add_argument("--intermediate-msh", default=None, help="Explicit path for the intermediate Gmsh .msh file during AVL export.")
    parser.add_argument("--save-all", action="store_true", help="Pass -save_all to the Gmsh CGNS conversion step.")
    parser.add_argument("--hex-order", choices=HEX_ORDER_MODES, default=None, help="Hex node ordering variant used only for export-time hexahedra. Default for AVL export is reverse.")
    parser.add_argument(
        "--strip-cgns-prefixes",
        action="store_true",
        help="Experimental: strip Gmsh-generated S_/V_ prefixes from CGNS names after export. Disabled by default because some importers reject the rewritten CGNS.",
    )
    parser.add_argument("--length", type=float, default=None, help="Override pipe length.")
    parser.add_argument("--nz", type=int, default=None, help="Override the number of axial layers.")
    parser.add_argument("--pitch-angle-deg", type=float, default=None, help="Override the helix pitch angle in degrees.")
    parser.add_argument("--core-mode", choices=["center_point", "annulus_tube"], default=None, help="Choose the inner topology mode.")
    parser.add_argument("--r-inner-tube", type=float, default=None, help="Override the free inner tube radius for annulus_tube mode.")
    parser.add_argument("--r-pipe", type=float, default=None, help="Override the pipe radius.")
    parser.add_argument("--rib-height", type=float, default=None, help="Override the rib height.")
    parser.add_argument("--rib-width-outer", type=float, default=None, help="Override the outer rib width distance in meters.")
    parser.add_argument("--rib-width-inner", type=float, default=None, help="Override the inner rib width distance in meters.")
    parser.add_argument(
        "--outfile",
        default=None,
        help="Generic output path override. If omitted, the value from parameters.json is used.",
    )
    args = parser.parse_args()

    params = HelixMeshParams.from_json(args.params)
    if args.length is not None:
        params.length = args.length
    if args.nz is not None:
        params.n_z = args.nz
    if args.pitch_angle_deg is not None:
        params.pitch_angle_deg = args.pitch_angle_deg
    if args.core_mode is not None:
        params.core_mode = args.core_mode
    if args.r_inner_tube is not None:
        params.r_inner_tube = args.r_inner_tube
    if args.r_pipe is not None:
        params.r_pipe = args.r_pipe
    if args.rib_height is not None:
        params.rib_height = args.rib_height
    if args.rib_width_outer is not None:
        params.rib_width_outer = args.rib_width_outer
    if args.rib_width_inner is not None:
        params.rib_width_inner = args.rib_width_inner
    if args.gmsh_exe is not None:
        params.gmsh_exe = args.gmsh_exe
    if args.keep_intermediate:
        params.keep_intermediate = True
    if args.intermediate_msh is not None:
        params.intermediate_msh = args.intermediate_msh
    if args.save_all:
        params.save_all = True
    if args.hex_order is not None:
        params.hex_order = args.hex_order
    if args.fluid_only:
        params.fluid_only = True
    if args.strip_cgns_prefixes:
        params.strip_cgns_prefixes = True

    try:
        mode = _resolve_output_mode(args, params)
    except ValueError as exc:
        parser.error(str(exc))
    params.validate()

    points, patches = build_topology(params)
    print(topology_summary(patches))

    if mode == "plot_2d":
        outfile = _resolve_outfile(mode, args, params)
        plot_topology(points, patches, params, outfile=outfile)
    elif mode == "plot_2d_mesh":
        outfile = _resolve_outfile(mode, args, params)
        mesh = build_mesh2d(points, patches)
        plot_mesh2d(mesh, points, patches, params, outfile=outfile)
    elif mode == "plot_3d_preview":
        outfile = _resolve_outfile(mode, args, params)
        mesh2d = build_mesh2d(points, patches)
        mesh3d = build_mesh3d(mesh2d, params)
        plot_3d_preview(mesh3d, params, outfile=outfile)
    elif mode == "export_gmsh":
        outfile = _resolve_outfile(mode, args, params)
        mesh2d = build_mesh2d(points, patches)
        mesh3d = build_mesh3d(mesh2d, params)
        export_gmsh_discrete(mesh3d, params, outfile=outfile)
    elif mode == "export_gmsh_full":
        outfile = _resolve_outfile(mode, args, params)
        mesh2d = build_mesh2d(points, patches)
        mesh3d = build_mesh3d(mesh2d, params)
        export_gmsh_discrete(mesh3d, params, outfile=outfile, fluid_only=False)
    else:
        outfile = _resolve_outfile(mode, args, params)
        mesh2d = build_mesh2d(points, patches)
        mesh3d = build_mesh3d(mesh2d, params)
        export_avl_cgns_via_gmsh(
            mesh3d,
            params,
            outfile=outfile,
            gmsh_exe=params.gmsh_exe,
            keep_intermediate=params.keep_intermediate,
            intermediate_msh=params.intermediate_msh,
            export_solids=not params.fluid_only,
            save_all=params.save_all,
            hex_order=params.hex_order,
            strip_cgns_prefixes=params.strip_cgns_prefixes,
        )


if __name__ == "__main__":
    main()
