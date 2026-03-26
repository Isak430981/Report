"""Pipeline interfaces for interchangeable modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Detector(ABC):
    """Responsible for detecting candidate dynamic objects in frames."""

    @abstractmethod
    def detect(self, frame_path: Path) -> Any:
        """Return detector output for one frame."""


class MaskGenerator(ABC):
    """Builds binary masks from detector output and frame data."""

    @abstractmethod
    def build_mask(self, frame_path: Path, detections: Any) -> Any:
        """Return a mask for one frame."""


class Inpainter(ABC):
    """Fills masked regions in frame sequences."""

    @abstractmethod
    def inpaint(self, frame_path: Path, mask: Any, output_path: Path) -> None:
        """Write inpainted result for one frame."""
