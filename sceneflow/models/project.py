"""ProjectData — aggregated result of loading a project folder."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from sceneflow.models.scene import SceneData


@dataclass
class ProjectData:
    project_path: Path
    scenes: List[SceneData] = field(default_factory=list)
    script_dict: Dict[int, str] = field(default_factory=dict)
    image_folder: Path = field(default=Path("."))
    voice_folder: Path = field(default=Path("."))
    validation_errors: List[str] = field(default_factory=list)

    @property
    def total_duration(self) -> float:
        return sum(s.duration_seconds for s in self.scenes)

    @property
    def is_valid(self) -> bool:
        return len(self.validation_errors) == 0 and len(self.scenes) > 0
