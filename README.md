# Structured Geometry Mesh Framework

This repository is organized as a geometry-extensible meshing framework.
The primary package is `mesh_framework`, and the currently implemented geometry type is `helix_pipe`.

The legacy `helix_mesh` folder is kept as a compatibility layer around the original helix implementation, but the main public naming and entrypoints now live under the generic framework API.

The implementation currently covers:

- Milestone 1: explicit corrected 2D cross-section topology
- Milestone 2: structured 2D quadrilateral meshing on the 18-patch topology
- Milestone 3: axial extrusion to a twisted 3D hexahedral mesh
- Milestone 4: Gmsh `*.msh` export
- Milestone 5: AVL-oriented CGNS export through Gmsh conversion

## Topology Overview

For the current `helix_pipe` geometry, the corrected cross-section uses 18 quadrilateral patches:

- 6 inner fluid patches
- 6 outer fluid gap patches
- 6 outer `rib_solid` patches

Each rib is trapezoidal and split into two outer `rib_solid` patches along its centerline.
The six inner quadrilateral patches meet directly at the center point, so there is no separate small core polygon in the current corrected topology.

The inner gap midpoints remain Cartesian chord midpoints between neighboring rib-tip corners and do not lie on `r_tip`.

## Twist Definition

The mesh twist is not entered directly as `total_twist_deg`.
Instead, it is derived from `pitch_angle_deg`:

`total_twist_deg = length / (pi * diameter * tan(pitch_angle_deg * pi / 180)) * 360`

with:

- `diameter = 2 * r_pipe`
- `phi(z) = total_twist * z / length`

The rib widths are specified as direct chord distances in meters:

- `rib_width_outer`
- `rib_width_inner`

They are converted internally to angular widths through the chord relation:

- `rib_width_outer_angle = 2 * asin(rib_width_outer / (2 * r_pipe))`
- `rib_width_inner_angle = 2 * asin(rib_width_inner / (2 * r_tip))`

## Output Modes

The script supports six output modes:

- `plot_2d`
- `plot_2d_mesh`
- `plot_3d_preview`
- `export_gmsh`
- `export_gmsh_full`
- `export_avl_cgns`

You can choose the mode either:

- in [parameters.json](/D:/Software/Python/python-gmsh/mesh_framework/parameters.json:1) via `output_mode`
- or via CLI flags, which take precedence

## Current Export Behavior

### `export_gmsh`

Writes a fluid-only discrete Gmsh `*.msh` mesh.
Internal `rib_solid` cells are not exported as volume cells in this mode.
Fluid-facing surfaces next to omitted rib solids are exported as `BND_W_RibWall`.

Typical groups:

- Volume: `DOM_GasVolume`
- Surfaces: `BND_I_Inlet_Gas`, `BND_O_Outlet_Gas`, `BND_W_ReactorWall`, `BND_W_RibWall`, `axis_core`

### `export_gmsh_full`

Writes the full discrete Gmsh mesh with both domains:

- `DOM_GasVolume`
- `DOM_RibVolume`

### `export_avl_cgns`

Writes a CGNS file for AVL import experiments through the path:

`internal mesh -> Gmsh .msh -> Gmsh CGNS conversion`

The implementation first tries the Python Gmsh API and falls back to a `gmsh` executable if needed.

Important distinction:

- `--export-gmsh` and `--export-gmsh-full` create Gmsh MSH files
- AVL's "MSH Gambit mesh (*.msh)" importer is not the same as Gmsh MSH support
- the supported AVL path is the CGNS export, not direct Gmsh-MSH import

Current stable AVL recommendation:

- use raw Gmsh CGNS naming
- keep `strip_cgns_prefixes = false`
- use `hex_order = reverse`

The currently reliable AVL-compatible raw naming looks like:

- `S_1_BND_I_Inlet_Gas`
- `V_1_DOM_GasVolume`

The project keeps the intended engineering names embedded in that scheme, but does not strip the `S_` / `V_` prefixes by default because that post-processing made some importers reject the CGNS file.

## Parameters File

The default configuration file is [parameters.json](/D:/Software/Python/python-gmsh/mesh_framework/parameters.json:1) inside the `mesh_framework` folder.

The framework now uses `parameters.json` not only for geometry and meshing, but also for optional export settings such as:

- output mode
- output file name
- CGNS export settings
- AVL hex-order selection
- geometry type selection

CLI arguments can still override individual values.

### Full JSON Parameter List

#### Output and Export Settings

- `output_mode`
  Supported values:
  `plot_2d`, `plot_2d_mesh`, `plot_3d_preview`, `export_gmsh`, `export_gmsh_full`, `export_avl_cgns`
- `outfile`
  Generic output target path for the selected mode
- `fluid_only`
  AVL export only; if `true`, omit the rib solid volume
- `gmsh_exe`
  Path or command name for the Gmsh executable fallback
- `keep_intermediate`
  AVL export only; keep the intermediate Gmsh `.msh`
- `intermediate_msh`
  Explicit path for the intermediate `.msh`
- `save_all`
  Pass `-save_all` to Gmsh CGNS conversion
- `hex_order`
  Supported values: `reverse`, `flip_i`, `flip_k`, `default`
- `strip_cgns_prefixes`
  Experimental CGNS post-processing flag; currently not recommended for AVL

#### Geometry Selection

- `geometry_type`
  Currently supported values: `helix_pipe`

#### Helix Geometry Parameters

- `r_pipe`
- `length`
- `n_ribs`
- `rib_height`
- `rib_width_outer`
- `rib_width_inner`
- `pitch_angle_deg`

#### Axial and Patch Resolution

- `n_z`
- `n_radial_outer`
- `n_radial_inner`
- `n_tangential_rib_half`
- `n_tangential_gap_half`
- `n_tangential_inner`

#### Core Topology

- `core_mode`
  Supported values: `center_point`, `annulus_tube`
- `core_radius_fraction`
- `r_inner_tube`

#### Miscellaneous

- `msh_version`

### Example `parameters.json`

```json
{
  "geometry_type": "helix_pipe",
  "output_mode": "export_avl_cgns",
  "outfile": "geometry_multidomain.cgns",
  "fluid_only": false,
  "gmsh_exe": "gmsh",
  "keep_intermediate": false,
  "intermediate_msh": null,
  "save_all": false,
  "hex_order": "reverse",
  "strip_cgns_prefixes": false,
  "r_pipe": 0.007,
  "length": 0.391,
  "n_ribs": 3,
  "rib_height": 0.002,
  "rib_width_outer": 0.002,
  "rib_width_inner": 0.001,
  "pitch_angle_deg": 60,
  "n_z": 200,
  "core_mode": "center_point",
  "core_radius_fraction": 0.15,
  "r_inner_tube": 0.0015,
  "n_radial_outer": 4,
  "n_radial_inner": 8,
  "n_tangential_rib_half": 2,
  "n_tangential_gap_half": 8,
  "n_tangential_inner": 8,
  "msh_version": 2.2
}
```

## CLI Options

The CLI supports both output-mode flags and value overrides.

### Output Selection

- `--plot-2d`
- `--plot-2d-mesh`
- `--plot-3d-preview`
- `--export-gmsh OUTFILE`
- `--export-gmsh-full OUTFILE`
- `--export-avl-cgns OUTFILE`

If none of these is passed, the script uses `output_mode` from `parameters.json`.

### Generic Output Override

- `--outfile PATH`

This overrides the JSON `outfile` for the currently selected mode.

### AVL / CGNS Export Options

- `--fluid-only`
- `--gmsh-exe PATH`
- `--keep-intermediate`
- `--intermediate-msh PATH`
- `--save-all`
- `--hex-order reverse|flip_i|flip_k|default`
- `--strip-cgns-prefixes`

### Geometry and Mesh Overrides

- `--length FLOAT`
- `--nz INT`
- `--pitch-angle-deg FLOAT`
- `--core-mode center_point|annulus_tube`
- `--r-inner-tube FLOAT`
- `--r-pipe FLOAT`
- `--rib-height FLOAT`
- `--rib-width-outer FLOAT`
- `--rib-width-inner FLOAT`

## Typical Commands

### Use only `parameters.json`

```powershell
.\.venv\Scripts\python.exe .\generate_mesh.py
```

### Write a fluid-only Gmsh mesh

```powershell
.\.venv\Scripts\python.exe .\generate_mesh.py --export-gmsh structured_geometry.msh
```

### Write the full multi-domain Gmsh mesh

```powershell
.\.venv\Scripts\python.exe .\generate_mesh.py --export-gmsh-full structured_geometry_full.msh
```

### Write the stable AVL CGNS export

```powershell
.\.venv\Scripts\python.exe .\generate_mesh.py --export-avl-cgns geometry_multidomain.cgns
```

### Keep the intermediate MSH for inspection

```powershell
.\.venv\Scripts\python.exe .\generate_mesh.py --export-avl-cgns geometry_multidomain.cgns --keep-intermediate
```

### Run an AVL orientation variant

```powershell
.\.venv\Scripts\python.exe .\generate_mesh.py --export-avl-cgns geometry_flip_i.cgns --hex-order flip_i --keep-intermediate
```

## Physical Names and Domains

The intended engineering names are:

- Fluid volume: `DOM_GasVolume`
- Rib solid volume: `DOM_RibVolume`
- Fluid inlet: `BND_I_Inlet_Gas`
- Rib inlet: `BND_I_Inlet_Rib`
- Fluid outlet: `BND_O_Outlet_Gas`
- Rib outlet: `BND_O_Outlet_Rib`
- Reactor wall: `BND_W_ReactorWall`
- Rib wall: `BND_W_RibWall`

For Gmsh exports, these names are assigned as both physical-group names and entity names.

For AVL CGNS export, the stable raw Gmsh CGNS path currently keeps prefixed family names such as `V_1_DOM_GasVolume`.

## Core Topology Note

In `center_point` mode, `axis_core` is intentionally empty because the inner region closes all the way to the center point.

In `annulus_tube` mode, `axis_core` exists as an actual open inner boundary around the free inner tube / probe region.

## Requirements

Use a local `.venv`.
Before running tests or exports, make sure the environment contains at least:

- `numpy`
- `pytest`
- `matplotlib`
- `gmsh`

For AVL CGNS export, the Python Gmsh module is the primary path.
A callable `gmsh` executable is used only as fallback.

Windows PowerShell example:

```powershell
.\.venv\Scripts\python.exe -c "import numpy, pytest, matplotlib, gmsh; print('requirements available')"
gmsh -version
```

## Manual Checks

Inspect a Gmsh mesh:

```powershell
gmsh structured_geometry.msh
```

OpenFOAM-side check, if available:

```bash
gmshToFoam structured_geometry.msh
checkMesh
```

AVL-side check:

1. Import the CGNS file through the CGNS importer, not through the Gambit-MSH importer.
2. Confirm that the gas and rib domains are present.
3. Confirm that the inlet, outlet, reactor wall, and rib wall groups are present.

## AVL Troubleshooting

- If AVL rejects a direct `*.msh`, check whether it expects Gambit/Fluent MSH instead of Gmsh MSH.
- Prefer the CGNS export path for AVL experiments.
- Start with `strip_cgns_prefixes = false`.
- Use `--keep-intermediate` to inspect the intermediate `.msh` in Gmsh.
- Use `--save-all` only as an experiment if zones appear to be missing.
- If needed, test the fallback conversion manually:
  `gmsh intermediate.msh -format cgns -o test.cgns`
