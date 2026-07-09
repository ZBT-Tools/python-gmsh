from dataclasses import dataclass, fields
import json
import math
from pathlib import Path


@dataclass
class HelixMeshParams:
    output_mode: str = "export_gmsh"
    outfile: str = "helix_pipe_structured.msh"
    fluid_only: bool = False
    gmsh_exe: str = "gmsh"
    keep_intermediate: bool = False
    intermediate_msh: str | None = None
    save_all: bool = False
    hex_order: str = "reverse"
    strip_cgns_prefixes: bool = False

    r_pipe: float = 7.0e-3
    length: float = 100.0e-3

    n_ribs: int = 3
    rib_height: float = 2.0e-3

    rib_width_outer: float = 0.002
    rib_width_inner: float = 0.001

    # Helix angle measured against the circumferential direction.
    # A larger angle means a larger axial pitch and therefore less total twist.
    pitch_angle_deg: float = 45

    n_z: int = 96

    # Supported center/core topology modes:
    # - center_point: current collapsed-center inner patch topology
    # - annulus_tube: free inner circular tube or probe boundary
    core_mode: str = "center_point"

    # Legacy helper radius used by the center-point mode for debug/reference.
    core_radius_fraction: float = 0.15
    r_inner_tube: float = 1.5e-3

    # Patch-local resolutions
    n_radial_outer: int = 4
    n_radial_inner: int = 8
    n_tangential_rib_half: int = 4
    n_tangential_gap_half: int = 8
    n_tangential_inner: int = 8

    msh_version: float = 2.2

    @classmethod
    def from_json(cls, path: str | Path) -> "HelixMeshParams":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Parameter file must contain a JSON object.")

        valid_fields = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - valid_fields)
        if unknown:
            raise ValueError(f"Unknown parameter keys in JSON file: {', '.join(unknown)}")

        return cls(**data)

    @property
    def r_tip(self) -> float:
        return self.r_pipe - self.rib_height

    @property
    def r_core(self) -> float:
        return self.core_radius_fraction * self.r_tip

    @property
    def rib_width_outer_angle(self) -> float:
        return 2.0 * math.asin(self.rib_width_outer / (2.0 * self.r_pipe))

    @property
    def rib_width_inner_angle(self) -> float:
        return 2.0 * math.asin(self.rib_width_inner / (2.0 * self.r_tip))

    @property
    def diameter(self) -> float:
        return 2.0 * self.r_pipe

    @property
    def pitch_angle(self) -> float:
        return math.radians(self.pitch_angle_deg)

    @property
    def total_twist_deg(self) -> float:
        if math.isclose(self.pitch_angle_deg, 90.0, rel_tol=0.0, abs_tol=1.0e-12):
            return 0.0
        return self.length / (math.pi * self.diameter * math.tan(self.pitch_angle)) * 360.0

    @property
    def total_twist(self) -> float:
        return math.radians(self.total_twist_deg)

    @property
    def inner_boundary_radius(self) -> float:
        if self.core_mode == "annulus_tube":
            return self.r_inner_tube
        return self.r_core

    def validate(self) -> None:
        valid_output_modes = {
            "plot_2d",
            "plot_2d_mesh",
            "plot_3d_preview",
            "export_gmsh",
            "export_gmsh_full",
            "export_avl_cgns",
        }
        valid_hex_orders = {"default", "flip_i", "flip_k", "reverse"}

        if self.output_mode not in valid_output_modes:
            raise ValueError(f"output_mode must be one of: {', '.join(sorted(valid_output_modes))}.")
        if not self.outfile:
            raise ValueError("outfile must not be empty.")
        if self.hex_order not in valid_hex_orders:
            raise ValueError(f"hex_order must be one of: {', '.join(sorted(valid_hex_orders))}.")
        if self.n_ribs != 3:
            raise ValueError("Version 1 supports exactly n_ribs = 3.")
        if self.r_pipe <= 0:
            raise ValueError("r_pipe must be positive.")
        if self.length <= 0:
            raise ValueError("length must be positive.")
        if self.rib_height <= 0:
            raise ValueError("rib_height must be positive.")
        if self.r_tip <= 0:
            raise ValueError("r_tip = r_pipe - rib_height must be positive.")
        if not (0.0 < self.r_core < self.r_tip):
            raise ValueError("r_core must be positive and smaller than r_tip.")
        if self.rib_width_outer <= 0:
            raise ValueError("rib_width_outer must be positive.")
        if self.rib_width_inner <= 0:
            raise ValueError("rib_width_inner must be positive.")
        if self.rib_width_outer >= 2.0 * self.r_pipe:
            raise ValueError("rib_width_outer must be smaller than the outer diameter.")
        if self.rib_width_inner >= 2.0 * self.r_tip:
            raise ValueError("rib_width_inner must be smaller than the tip diameter.")
        if self.rib_width_outer_angle <= self.rib_width_inner_angle:
            raise ValueError("rib_width_outer angle must be larger than rib_width_inner angle.")
        if not (0.0 < self.pitch_angle_deg <= 90.0):
            raise ValueError("pitch_angle_deg must satisfy 0 < pitch_angle_deg <= 90.")
        if self.n_z <= 0:
            raise ValueError("n_z must be positive.")
        if self.core_mode not in {"center_point", "annulus_tube"}:
            raise ValueError("core_mode must be either 'center_point' or 'annulus_tube'.")
        if self.core_mode == "annulus_tube" and not (0.0 < self.r_inner_tube < self.r_tip):
            raise ValueError("r_inner_tube must be positive and smaller than r_tip in annulus_tube mode.")
