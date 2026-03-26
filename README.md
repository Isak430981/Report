# Video Object Removal Project Scaffold

Clean, modular Python scaffold for a computer vision course project on **video object removal**.

## Project Structure

```
.
├── configs/
│   └── pipeline.example.yaml
├── frames/            # input video frames (png/jpg)
├── masks/             # generated masks
├── outputs/           # inpainted outputs
├── scripts/
│   ├── main.py        # pipeline entrypoint
│   └── modules/
│       ├── factory.py
│       ├── interfaces.py
│       └── stubs.py
├── requirements.txt
└── README.md
```

## Why this structure?

- **Modular interfaces** (`Detector`, `MaskGenerator`, `Inpainter`) let you swap components quickly.
- **Factory pattern** keeps model selection in config instead of hard-coded logic.
- **Stub modules** allow end-to-end testing before integrating heavy models.

This is set up so you can later plug in:
- YOLO / YOLOv8-seg for detection,
- SAM2 (or SAM3) for mask generation,
- ProPainter (or alternatives like E2FGVI/FGVC) for inpainting.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. Put extracted frames into `frames/`.
2. Optionally edit `configs/pipeline.example.yaml`.
3. Run:

```bash
python scripts/main.py --config configs/pipeline.example.yaml
```

Outputs:
- masks in `masks/`
- restored frames in `outputs/`

## Next Integration Steps

1. Replace `YoloStubDetector` with YOLO/SOTA detector wrapper.
2. Replace `Sam2StubMaskGenerator` with SAM2 video mask propagation logic.
3. Replace `ProPainterStubInpainter` with a real ProPainter inference wrapper.
4. Add evaluation scripts for IoU/JR/PSNR/SSIM for course metrics.


## Repository walkthrough (quick answers)

1. **Inference runner**: `scripts/main.py` is the entrypoint; run it with `python scripts/main.py --config <yaml>`.  
2. **Expected inputs**:
   - A YAML config (`configs/pipeline.example.yaml`) with `paths` and `pipeline` module names.
   - Frame images in `paths.frames_dir` (`*.png`/`*.jpg`).
3. **Output locations**:
   - Masks are written to `paths.masks_dir` as `<frame_stem>_mask.png`.
   - Restored images are written to `paths.outputs_dir` using the original frame filename.
4. **Files to modify for batch-processing video folders**:
   - `scripts/main.py`: change frame discovery/iteration logic (e.g., recurse subfolders or add video decoding).
   - `configs/pipeline.example.yaml`: point `paths.frames_dir` / output dirs per run.
   - `scripts/modules/stubs.py` (later your real model wrappers): if batching requires model-specific sequence handling.
5. **Wrapper script**:
   - Use `scripts/pipeline_wrapper.py` and call `run_with_paths(...)` from your own orchestrator.

### Wrapper usage example

```python
from scripts.pipeline_wrapper import run_with_paths

run_with_paths(
    frames_dir="/data/course_videos/video_01_frames",
    masks_dir="/data/course_videos/video_01_masks",
    outputs_dir="/data/course_videos/video_01_outputs",
    config_path="configs/pipeline.example.yaml",
)
```
