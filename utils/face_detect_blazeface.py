from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2

_BLAZEFACE = None
_ANCHORS = None


def _load_blazeface(models_dir: Path) -> Optional[object]:
    global _BLAZEFACE, _ANCHORS
    if _BLAZEFACE is not None:
        return _BLAZEFACE
    try:
        # Expect vendored file downloaded from hollance/BlazeFace-PyTorch
        from utils.vendor.blazeface import BlazeFace  # type: ignore
        import torch

        anchors_path = models_dir / "blazeface" / "anchors.npy"
        weights_path = models_dir / "blazeface" / "blazeface.pth"
        if not anchors_path.exists() or not weights_path.exists():
            return None

        model = BlazeFace().to("cpu")
        model.load_weights(str(weights_path))
        model.load_anchors(str(anchors_path))
        model.min_score_thresh = 0.1
        model.min_suppression_threshold = 0.3
        model.eval()
        _BLAZEFACE = model
        return _BLAZEFACE
    except Exception:
        return None


def _postprocess(det, H: int, W: int) -> Optional[Tuple[int, int, int, int]]:
    if det is None:
        return None
    if hasattr(det, "numel") and det.numel() == 0:
        return None
    boxes = det[:, :4].cpu().numpy()
    scores = det[:, -1].cpu().numpy()
    if boxes.size == 0:
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idx = int(np.argmax(scores * areas))
    ymin, xmin, ymax, xmax = boxes[idx]
    x1 = int(max(0, xmin * W))
    y1 = int(max(0, ymin * H))
    x2 = int(min(W - 1, xmax * W))
    y2 = int(min(H - 1, ymax * H))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def detect_largest_face_bbox(rgb_image: np.ndarray, project_root: Path) -> Optional[Tuple[int, int, int, int]]:
    models_dir = project_root / "models"
    model = _load_blazeface(models_dir)
    if model is None:
        return None

    try:
        import torch, cv2
        H_full, W_full = rgb_image.shape[:2]

        # center-square → 128x128
        s = min(H_full, W_full)
        y0 = (H_full - s) // 2
        x0 = (W_full - s) // 2
        square = rgb_image[y0:y0 + s, x0:x0 + s]
        resized = cv2.resize(square, (128, 128), interpolation=cv2.INTER_AREA)

        # 1) predict_on_image (RGB)
        det = model.predict_on_image(resized.astype(np.uint8))
        bbox_sq = _postprocess(det, s, s)
        if bbox_sq is not None:
            x1, y1, x2, y2 = bbox_sq
            return (x1 + x0, y1 + y0, x2 + x0, y2 + y0)

        # 2) predict_on_image (BGR)
        det = model.predict_on_image(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR).astype(np.uint8))
        bbox_sq = _postprocess(det, s, s)
        if bbox_sq is not None:
            x1, y1, x2, y2 = bbox_sq
            return (x1 + x0, y1 + y0, x2 + x0, y2 + y0)

        # 3) predict_on_batch NHWC
        img_nhwc = torch.from_numpy(resized).float().unsqueeze(0) / 255.0
        dets = model.predict_on_batch(img_nhwc)
        if isinstance(dets, (list, tuple)) and len(dets) > 0:
            bbox_sq = _postprocess(dets[0], s, s)
            if bbox_sq is not None:
                x1, y1, x2, y2 = bbox_sq
                return (x1 + x0, y1 + y0, x2 + x0, y2 + y0)

        # 4) predict_on_batch NCHW
        img_nchw = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        dets = model.predict_on_batch(img_nchw)
        if isinstance(dets, (list, tuple)) and len(dets) > 0:
            bbox_sq = _postprocess(dets[0], s, s)
            if bbox_sq is not None:
                x1, y1, x2, y2 = bbox_sq
                return (x1 + x0, y1 + y0, x2 + x0, y2 + y0)

        # 5) full-frame fallback
        full = cv2.resize(rgb_image, (128, 128), interpolation=cv2.INTER_AREA)
        det = model.predict_on_image(full.astype(np.uint8))
        bbox = _postprocess(det, H_full, W_full)
        if bbox is not None:
            return bbox

        full_nchw = torch.from_numpy(full).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        dets = model.predict_on_batch(full_nchw)
        if isinstance(dets, (list, tuple)) and len(dets) > 0:
            bbox = _postprocess(dets[0], H_full, W_full)
            if bbox is not None:
                return bbox

        return None
    except Exception:
        return None