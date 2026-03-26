"""Main pipeline entrypoint for video object removal experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml

from scripts.modules.factory import build_detector, build_inpainter, build_masker


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(config: dict) -> None:
    frames_dir = Path(config["paths"]["frames_dir"])
    masks_dir = Path(config["paths"]["masks_dir"])
    outputs_dir = Path(config["paths"]["outputs_dir"])

    detector = build_detector(config["pipeline"]["detector"])
    masker = build_masker(config["pipeline"]["masker"])
    inpainter = build_inpainter(config["pipeline"]["inpainter"])

    frame_paths = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        print(f"No frames found in {frames_dir}. Add extracted frames and rerun.")
        return

    for frame_path in frame_paths:
        detections = detector.detect(frame_path)
        mask = masker.build_mask(frame_path, detections)

        mask_path = masks_dir / f"{frame_path.stem}_mask.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_path), mask)

        output_path = outputs_dir / frame_path.name
        inpainter.inpaint(frame_path, mask, output_path)
        print(f"Processed: {frame_path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video object removal pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pipeline.example.yaml"),
        help="Path to YAML config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    run_pipeline(cfg)
