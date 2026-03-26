"""Wrapper utilities for calling the removal pipeline from other code."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from scripts.main import load_config, run_pipeline


def run_with_paths(
    *,
    frames_dir: str | Path,
    masks_dir: str | Path,
    outputs_dir: str | Path,
    config_path: str | Path = "configs/pipeline.example.yaml",
    detector: str | None = None,
    masker: str | None = None,
    inpainter: str | None = None,
) -> dict[str, Any]:
    """Run the pipeline programmatically with caller-provided directories.

    Returns the resolved config dictionary that was used for the run so callers
    can log experiment metadata.
    """

    config_file = Path(config_path)
    config = load_config(config_file)
    resolved = copy.deepcopy(config)

    resolved["paths"]["frames_dir"] = str(Path(frames_dir))
    resolved["paths"]["masks_dir"] = str(Path(masks_dir))
    resolved["paths"]["outputs_dir"] = str(Path(outputs_dir))

    if detector:
        resolved["pipeline"]["detector"] = detector
    if masker:
        resolved["pipeline"]["masker"] = masker
    if inpainter:
        resolved["pipeline"]["inpainter"] = inpainter

    run_pipeline(resolved)
    return resolved


if __name__ == "__main__":
    # Example: keep this tiny so external pipeline code can import run_with_paths.
    run_with_paths(
        frames_dir="frames",
        masks_dir="masks",
        outputs_dir="outputs",
    )
