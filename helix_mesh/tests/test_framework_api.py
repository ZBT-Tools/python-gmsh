from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mesh_framework import DEFAULT_CONFIG_PATH, MeshGeneratorConfig, build_cross_section_topology, summarize_topology


def test_default_framework_config_path_exists() -> None:
    assert DEFAULT_CONFIG_PATH.exists()


def test_framework_config_defaults_to_helix_pipe() -> None:
    config = MeshGeneratorConfig.from_json(DEFAULT_CONFIG_PATH)

    assert config.geometry_type == "helix_pipe"


def test_framework_topology_builder_dispatches_helix_geometry() -> None:
    config = MeshGeneratorConfig.from_json(DEFAULT_CONFIG_PATH)

    _, patches = build_cross_section_topology(config)
    summary = summarize_topology(patches)

    assert "Total patches: 18" in summary
    assert "Outer rib_solid patches: 6" in summary
