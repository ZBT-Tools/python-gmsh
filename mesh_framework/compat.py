from __future__ import annotations

import sys
from pathlib import Path


LEGACY_HELIX_DIR = Path(__file__).resolve().parents[1] / "helix_mesh"
if str(LEGACY_HELIX_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_HELIX_DIR))

import export_avl as legacy_export_avl  # noqa: E402
import export_gmsh as legacy_export_gmsh  # noqa: E402
import extrude as legacy_extrude  # noqa: E402
import mesh2d as legacy_mesh2d  # noqa: E402
import params as legacy_params  # noqa: E402
import topology as legacy_topology  # noqa: E402
