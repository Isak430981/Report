"""Factory for creating pluggable pipeline modules."""

from __future__ import annotations

from .interfaces import Detector, Inpainter, MaskGenerator
from .stubs import ProPainterStubInpainter, Sam2StubMaskGenerator, YoloStubDetector


def build_detector(name: str) -> Detector:
    if name == "yolo_stub":
        return YoloStubDetector()
    raise ValueError(f"Unknown detector module: {name}")


def build_masker(name: str) -> MaskGenerator:
    if name == "sam2_stub":
        return Sam2StubMaskGenerator()
    raise ValueError(f"Unknown mask generator module: {name}")


def build_inpainter(name: str) -> Inpainter:
    if name == "propainter_stub":
        return ProPainterStubInpainter()
    raise ValueError(f"Unknown inpainter module: {name}")
