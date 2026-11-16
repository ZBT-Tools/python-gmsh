from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .compat import legacy_params


DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("parameters.json")


@dataclass
class MeshGeneratorConfig(legacy_params.HelixMeshParams):
    geometry_type: str = "helix_pipe"

    def validate(self) -> None:
        valid_geometry_types = {"helix_pipe"}
        if self.geometry_type not in valid_geometry_types:
            raise ValueError(
                "geometry_type must be one of: " + ", ".join(sorted(valid_geometry_types)) + "."
            )
        super().validate()
