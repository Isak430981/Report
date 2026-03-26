"""Stub implementations so the pipeline runs before model integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .interfaces import Detector, Inpainter, MaskGenerator


class YoloStubDetector(Detector):
    def detect(self, frame_path: Path) -> Any:
        # Placeholder detection result structure.
        return {"boxes": [], "classes": []}


class Sam2StubMaskGenerator(MaskGenerator):
    def build_mask(self, frame_path: Path, detections: Any) -> Any:
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read frame: {frame_path}")
        # Empty mask by default; replace with SAM2 prompt/memory logic later.
        return np.zeros(image.shape[:2], dtype=np.uint8)


class ProPainterStubInpainter(Inpainter):
    def inpaint(self, frame_path: Path, mask: Any, output_path: Path) -> None:
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read frame: {frame_path}")

        # OpenCV fallback inpainting for baseline behavior.
        result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)
