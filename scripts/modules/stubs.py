"""Stub implementations so the pipeline runs before model integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from ultralytics import YOLO

import cv2
import numpy as np

from .interfaces import Detector, Inpainter, MaskGenerator

print("[loaded] THIS stubs.py is being used", flush=True)

class YoloStubDetector(Detector):
    def __init__(self):
        print("[init] YoloStubDetector (segmentation)", flush=True)
        self.model = YOLO("yolov8n-seg.pt")

    def detect(self, frame_path: Path) -> Any:
        image = cv2.imread(str(frame_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read frame: {frame_path}")

        h, w = image.shape[:2]
        results = self.model(image)

        masks = []
        classes = []
        boxes = []

        for r in results:
            if r.boxes is None:
                continue

            cls_list = r.boxes.cls.tolist() if r.boxes.cls is not None else []

            # segmentation masks may be None
            mask_data = None
            if r.masks is not None and r.masks.data is not None:
                mask_data = r.masks.data.cpu().numpy()

            box_data = r.boxes.xyxy.cpu().numpy() if r.boxes.xyxy is not None else []

            for i, cls in enumerate(cls_list):
                cls = int(cls)
                classes.append(cls)

                if len(box_data) > i:
                    x1, y1, x2, y2 = map(int, box_data[i].tolist())
                    boxes.append([x1, y1, x2, y2])
                else:
                    boxes.append(None)

                if mask_data is not None and len(mask_data) > i:
                    # YOLO seg mask is usually resized to model shape, so resize back
                    mask = mask_data[i]
                    mask = (mask > 0.5).astype(np.uint8) * 255
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    masks.append(mask)
                else:
                    masks.append(None)

        print(f"{frame_path.name}: detected {len(classes)} objects", flush=True)
        print(f"{frame_path.name}: classes={classes}", flush=True)

        return {
            "boxes": boxes,
            "classes": classes,
            "masks": masks,
        }


class Sam2StubMaskGenerator(MaskGenerator):
    def build_mask(self, frame_path: Path, detections: Any) -> Any:
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read frame: {frame_path}")

        h, w = image.shape[:2]
        final_mask = np.zeros((h, w), dtype=np.uint8)

        classes = detections.get("classes", [])
        masks = detections.get("masks", [])

        for cls, mask in zip(classes, masks):
            # only keep person
            if cls != 0:
                continue
            if mask is None:
                continue

            final_mask = np.maximum(final_mask, mask)

        # very light dilation, just to avoid edge holes
        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.dilate(final_mask, kernel, iterations=1)

        debug_mask_path = Path("masks") / f"mask_{frame_path.name}"
        cv2.imwrite(str(debug_mask_path), final_mask)

        print(f"{frame_path.name}: mask sum = {final_mask.sum()}", flush=True)
        return final_mask


class ProPainterStubInpainter(Inpainter):
    def inpaint(self, frame_path: Path, mask: Any, output_path: Path) -> None:
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read frame: {frame_path}")

        result = image.copy()
        h, w = image.shape[:2]

        # normalize current mask to 2D uint8
        if mask is None:
            mask = np.zeros((h, w), dtype=np.uint8)
        else:
            mask = np.asarray(mask)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(np.uint8)

        stem = frame_path.stem
        try:
            prefix, idx_str = stem.rsplit("_", 1)
            idx = int(idx_str)
        except ValueError:
            fallback = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), fallback)
            return

        current_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        def load_neighbor(neighbor_idx: int):
            neighbor_name = f"{prefix}_{neighbor_idx:04d}{frame_path.suffix}"
            neighbor_path = frame_path.parent / neighbor_name

            if not neighbor_path.exists():
                return None, None, None, neighbor_name

            neighbor_img = cv2.imread(str(neighbor_path), cv2.IMREAD_COLOR)
            if neighbor_img is None:
                return None, None, None, neighbor_name

            if neighbor_img.shape[:2] != (h, w):
                neighbor_img = cv2.resize(neighbor_img, (w, h), interpolation=cv2.INTER_LINEAR)

            neighbor_mask_path = Path("masks") / f"mask_{neighbor_name}"
            if neighbor_mask_path.exists():
                neighbor_mask = cv2.imread(str(neighbor_mask_path), cv2.IMREAD_GRAYSCALE)
                if neighbor_mask is None:
                    neighbor_mask = np.zeros((h, w), dtype=np.uint8)
            else:
                neighbor_mask = np.zeros((h, w), dtype=np.uint8)

            neighbor_mask = np.asarray(neighbor_mask)
            if neighbor_mask.ndim == 3:
                neighbor_mask = neighbor_mask[:, :, 0]
            if neighbor_mask.shape[:2] != (h, w):
                neighbor_mask = cv2.resize(
                    neighbor_mask, (w, h), interpolation=cv2.INTER_NEAREST
                )
            neighbor_mask = neighbor_mask.astype(np.uint8)

            neighbor_gray = cv2.cvtColor(neighbor_img, cv2.COLOR_BGR2GRAY)
            return neighbor_img, neighbor_mask, neighbor_gray, neighbor_name

        def warp_to_current(neighbor_img, neighbor_mask, neighbor_gray):
            # Optical flow from neighbor -> current
            flow = cv2.calcOpticalFlowFarneback(
                neighbor_gray,
                current_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=21,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )

            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (grid_x + flow[..., 0]).astype(np.float32)
            map_y = (grid_y + flow[..., 1]).astype(np.float32)

            warped_img = cv2.remap(
                neighbor_img,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )

            warped_mask = cv2.remap(
                neighbor_mask,
                map_x,
                map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )

            return warped_img, warped_mask

        masked_region = mask > 0

        # use more neighboring frames
        offsets = [-3, -2, -1, 1, 2, 3]
        candidate_stack = []
        used_neighbors = 0

        for offset in offsets:
            neighbor_img, neighbor_mask, neighbor_gray, neighbor_name = load_neighbor(idx + offset)
            if neighbor_img is None:
                continue

            warped_img, warped_mask = warp_to_current(neighbor_img, neighbor_mask, neighbor_gray)

            valid = masked_region & (warped_mask == 0)

            if np.any(valid):
                candidate = np.full((h, w, 3), np.nan, dtype=np.float32)
                candidate[valid] = warped_img[valid].astype(np.float32)
                candidate_stack.append(candidate)
                used_neighbors += 1

        filled = np.zeros((h, w), dtype=bool)

        if candidate_stack:
            stack = np.stack(candidate_stack, axis=0)  # [N, H, W, 3]
            median_pixels = np.nanmedian(stack, axis=0)

            valid_fill = masked_region & np.isfinite(median_pixels).all(axis=2)
            result[valid_fill] = np.clip(median_pixels[valid_fill], 0, 255).astype(np.uint8)
            filled |= valid_fill

        remaining_mask = np.zeros((h, w), dtype=np.uint8)
        remaining_mask[masked_region & (~filled)] = 255

        if remaining_mask.sum() > 0:
            result = cv2.inpaint(
                result,
                remaining_mask,
                inpaintRadius=3,
                flags=cv2.INPAINT_TELEA,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)

        print(
            f"{frame_path.name}: neighbors_used={used_neighbors}, "
            f"temporal_filled={int(filled.sum())} px, "
            f"remaining={int((remaining_mask > 0).sum())} px",
            flush=True
        )