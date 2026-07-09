from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from params import HelixMeshParams


def _pitch_angle_for_twist_deg(length: float, diameter: float, twist_deg: float) -> float:
    if math.isclose(twist_deg, 0.0, abs_tol=1.0e-15):
        return 90.0
    return math.degrees(math.atan(length * 360.0 / (math.pi * diameter * twist_deg)))


def test_total_twist_deg_uses_pitch_angle_and_pi_diameter_relation() -> None:
    params = HelixMeshParams(pitch_angle_deg=45.0, length=0.08, r_pipe=0.014)
    expected = params.length / (math.pi * params.diameter * math.tan(math.radians(45.0))) * 360.0
    assert math.isclose(params.total_twist_deg, expected, rel_tol=0.0, abs_tol=1.0e-12)


def test_rib_width_angles_are_derived_from_metric_chord_widths() -> None:
    params = HelixMeshParams(r_pipe=0.014, rib_height=0.002, rib_width_outer=0.007, rib_width_inner=0.0024)
    assert math.isclose(
        params.rib_width_outer_angle,
        2.0 * math.asin(0.007 / (2.0 * 0.014)),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    assert math.isclose(
        params.rib_width_inner_angle,
        2.0 * math.asin(0.0024 / (2.0 * (0.014 - 0.002))),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )


def test_ninety_degree_pitch_angle_gives_zero_twist() -> None:
    params = HelixMeshParams(pitch_angle_deg=90.0)
    assert math.isclose(params.total_twist_deg, 0.0, abs_tol=1.0e-12)


def test_from_json_loads_parameters_file() -> None:
    payload = json.dumps(
        {
            "output_mode": "export_avl_cgns",
            "outfile": "mesh.cgns",
            "gmsh_exe": "gmsh",
            "hex_order": "reverse",
            "r_pipe": 0.012,
            "length": 0.09,
            "pitch_angle_deg": 35.0,
            "core_mode": "annulus_tube",
            "r_inner_tube": 0.001,
        }
    )

    with patch.object(Path, "read_text", return_value=payload):
        json_path = Path("parameters.json")
        params = HelixMeshParams.from_json(json_path)

        assert params.output_mode == "export_avl_cgns"
        assert params.outfile == "mesh.cgns"
        assert params.gmsh_exe == "gmsh"
        assert params.hex_order == "reverse"
        assert math.isclose(params.r_pipe, 0.012)
        assert math.isclose(params.length, 0.09)
        assert math.isclose(params.pitch_angle_deg, 35.0)
        assert params.core_mode == "annulus_tube"
        assert math.isclose(params.r_inner_tube, 0.001)


def test_from_json_rejects_unknown_keys() -> None:
    with patch.object(Path, "read_text", return_value=json.dumps({"does_not_exist": 1})):
        json_path = Path("parameters.json")
        with pytest.raises(ValueError, match="Unknown parameter keys"):
            HelixMeshParams.from_json(json_path)


def test_pitch_angle_validation_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="pitch_angle_deg"):
        HelixMeshParams(pitch_angle_deg=0.0).validate()


def test_pitch_angle_validation_rejects_values_above_ninety() -> None:
    with pytest.raises(ValueError, match="pitch_angle_deg"):
        HelixMeshParams(pitch_angle_deg=91.0).validate()


def test_validation_rejects_non_widening_outer_rib_width() -> None:
    params = HelixMeshParams(r_pipe=0.014, rib_height=0.002, rib_width_outer=0.002, rib_width_inner=0.0024)
    with pytest.raises(ValueError, match="rib_width_outer angle"):
        params.validate()


def test_validation_rejects_widths_larger_than_diameter() -> None:
    with pytest.raises(ValueError, match="outer diameter"):
        HelixMeshParams(r_pipe=0.014, rib_width_outer=0.03).validate()

    with pytest.raises(ValueError, match="tip diameter"):
        HelixMeshParams(r_pipe=0.014, rib_height=0.002, rib_width_inner=0.025).validate()


def test_validation_rejects_unknown_output_mode() -> None:
    with pytest.raises(ValueError, match="output_mode"):
        HelixMeshParams(output_mode="something_else").validate()


def test_validation_rejects_unknown_hex_order() -> None:
    with pytest.raises(ValueError, match="hex_order"):
        HelixMeshParams(hex_order="weird").validate()
