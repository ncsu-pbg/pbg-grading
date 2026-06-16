"""
PodTrackingNode — DepthAI v3 ThreadedHostNode for mask-based
morphological measurement and per-frame tracking.

Receives ImgDetections (with on-device seg masks) from DetectionNetwork,
filters by ROI / entry zone / confidence, extracts mask measurements,
and feeds the RealtimeTracker.
"""

from typing import Dict, List, Optional, Tuple

import depthai as dai
import cv2
import numpy as np

from pipeline.tracking import RealtimeTracker, _calculate_iou


# ═══════════════════════════════════════════════════════════════════════
# Letterbox geometry helpers
# ═══════════════════════════════════════════════════════════════════════

def compute_letterbox_params(
    src_w: int, src_h: int, dst_w: int, dst_h: int
) -> dict:
    """Compute letterbox resize parameters (scale, padding, content size)."""
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    pad_x = (dst_w - new_w) // 2
    pad_y = (dst_h - new_h) // 2
    return {
        "scale": scale,
        "new_w": new_w, "new_h": new_h,
        "pad_x": pad_x, "pad_y": pad_y,
        "dst_w": dst_w, "dst_h": dst_h,
    }


def nn_norm_to_crop_px(
    x_norm: float, y_norm: float, lb: dict
) -> Tuple[float, float]:
    """Convert normalised NN coords (0-1 in 640×640) to crop pixel coords."""
    nn_x = x_norm * lb["dst_w"]
    nn_y = y_norm * lb["dst_h"]
    crop_x = (nn_x - lb["pad_x"]) / lb["scale"]
    crop_y = (nn_y - lb["pad_y"]) / lb["scale"]
    return crop_x, crop_y


# ═══════════════════════════════════════════════════════════════════════
# Host node
# ═══════════════════════════════════════════════════════════════════════

class PodTrackingNode(dai.node.ThreadedHostNode):
    """
    On each frame:
      1. Map normalised NN bbox → crop pixel coords (undo letterbox)
      2. ROI centre-point filter
      3. Entry-zone filter
      4. Extract binary mask via getCvSegmentationMaskByIndex()
      5. Feed accepted detections into RealtimeTracker
    """

    def __init__(self):
        super().__init__()
        self.input = self.createInput()

        # Set externally before pipeline.start()
        self.tracker: Optional[RealtimeTracker] = None
        self.cfg: Optional[dict] = None

        # Counters
        self._frames_processed = 0
        self._dets_accepted = 0
        self._first_frame_printed = False

        # Geometry — computed once in _init_geometry()
        self._crop_w = 0
        self._crop_h = 0
        self._lb: Optional[dict] = None
        # Mask → crop pixel scale factors. These get LOCKED IN on the
        # first real mask we see, using the mask's actual shape (which
        # for YOLOv8-seg may be the NN input resolution OR a 1/4-res
        # prototype, depending on how the model archive is wired). The
        # initial values assume mask shape == NN input shape and are
        # only used until we observe an actual mask.
        self._mask_px_to_crop_px_area = 1.0
        self._mask_px_to_crop_px_len = 1.0
        self._mask_geom_locked = False

        # Pipeline config values (set from cfg in _init_geometry)
        self._conf_threshold = 0.70
        self._entry_zone_rel = 0.15
        self._min_mask_area = 200
        # Width-validity gating: a detection's width is excluded from the
        # min-width aggregate when its bbox touches the crop edge (clipped
        # mask) or overlaps another detection (occluded pod underneath).
        self._border_pad_px = 2
        self._overlap_iou_threshold = 0.12

    def _init_geometry(self):
        """Pre-compute all coordinate mapping from config."""
        cfg = self.cfg or {}
        cam = cfg.get("camera", {})
        cap = cfg.get("capture_crop", {})
        nn = cfg.get("nn", {})

        cap_w = cam.get("width", 1280)
        cap_h = cam.get("height", 800)
        nn_size = nn.get("input_size", [640, 640])

        self._conf_threshold = nn.get("confidence_threshold", 0.70)
        # Entry zone is the top fraction of the cropped frame — detections
        # whose bbox top edge lies inside this band are skipped (partial
        # objects entering the field of view). Default 0.15 = top 15%.
        self._entry_zone_rel = cfg.get("entry_zone_threshold_rel", 0.15)
        self._min_mask_area = cfg.get("min_mask_area_px", 200)
        self._border_pad_px = cfg.get("border_pad_px", 2)
        self._overlap_iou_threshold = cfg.get("overlap_iou_threshold", 0.12)

        # Capture crop size in camera pixels
        cx_min = cap.get("x_min", 0.0)
        cx_max = cap.get("x_max", 1.0)
        cy_min = cap.get("y_min", 0.0)
        cy_max = cap.get("y_max", 1.0)
        self._crop_w = int((cx_max - cx_min) * cap_w)
        self._crop_h = int((cy_max - cy_min) * cap_h)

        # Letterbox params: crop → NN input
        self._lb = compute_letterbox_params(
            self._crop_w, self._crop_h, nn_size[0], nn_size[1]
        )

        # Mask → crop pixel scale factors
        self._mask_px_to_crop_px_area = (1.0 / self._lb["scale"]) ** 2
        self._mask_px_to_crop_px_len = 1.0 / self._lb["scale"]

    def run(self):
        self._init_geometry()

        while self.isRunning():
            try:
                msg = self.input.get()
            except Exception:
                break

            self._frames_processed += 1
            n_dets = len(msg.detections)
            if n_dets == 0:
                continue

            # Diagnostics on first frame with detections
            if not self._first_frame_printed:
                self._first_frame_printed = True
                lb = self._lb
                print("\n" + "=" * 60)
                print("FIRST FRAME DIAGNOSTICS")
                print("=" * 60)
                print(f"  Capture crop   : {self._crop_w}x{self._crop_h} px")
                print(f"  Letterbox      : scale={lb['scale']:.4f}, "
                      f"content={lb['new_w']}x{lb['new_h']}, "
                      f"pad=({lb['pad_x']},{lb['pad_y']})")
                print(f"  Entry zone     : top {self._entry_zone_rel*100:.0f}% of crop "
                      f"(skip if bbox y1 < {self._entry_zone_rel * self._crop_h:.0f} px)")
                print(f"  Mask→crop scale: area={self._mask_px_to_crop_px_area:.4f}, "
                      f"len={self._mask_px_to_crop_px_len:.4f}")
                print(f"  Detections     : {n_dets}")

                det0 = msg.detections[0]
                x1, y1 = nn_norm_to_crop_px(det0.xmin, det0.ymin, lb)
                x2, y2 = nn_norm_to_crop_px(det0.xmax, det0.ymax, lb)
                print(f"  det[0] NN norm : ({det0.xmin:.4f},{det0.ymin:.4f})"
                      f"-({det0.xmax:.4f},{det0.ymax:.4f})")
                print(f"  det[0] crop px : ({x1:.1f},{y1:.1f})-({x2:.1f},{y2:.1f})")

                try:
                    cv_mask = msg.getCvSegmentationMask()
                    if cv_mask is not None:
                        arr = np.array(cv_mask)
                        print(f"  Seg mask shape : {arr.shape}, dtype={arr.dtype}")
                        print(f"  Seg mask unique: {np.unique(arr)[:15]}")
                except Exception as e:
                    print(f"  Seg mask error : {e}")
                print("=" * 60 + "\n")

            # ── Process detections (pass 1: filter + measure) ─────────
            lb = self._lb
            survivors: List[dict] = []

            for det_idx, det in enumerate(msg.detections):
                if det.confidence < self._conf_threshold:
                    continue

                x1, y1 = nn_norm_to_crop_px(det.xmin, det.ymin, lb)
                x2, y2 = nn_norm_to_crop_px(det.xmax, det.ymax, lb)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # Entry zone filter: skip detections whose bbox top edge
                # lies in the top entry_zone_rel fraction of the cropped
                # frame (partial objects entering the field of view).
                if self._crop_h > 0:
                    y1_norm = y1 / self._crop_h
                    if y1_norm < self._entry_zone_rel:
                        continue

                # Default to bbox measurements
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                length = max(bbox_w, bbox_h)
                width = min(bbox_w, bbox_h)
                area = bbox_w * bbox_h

                # Attempt mask-based measurements using cv2.minAreaRect
                # for true oriented length/width (not axis-aligned extent)
                try:
                    binary_mask = msg.getCvSegmentationMaskByIndex(det_idx)
                    if binary_mask is not None:
                        mask_arr = np.array(binary_mask, dtype=np.uint8)

                        # Lock in the real mask→crop conversion the FIRST
                        # time we see a mask. The mask shape may be the
                        # NN input resolution or a 1/4-res prototype —
                        # both happen with YOLOv8-seg depending on the
                        # archive — so we measure the actual shape and
                        # derive the scale from there.
                        if not self._mask_geom_locked:
                            mh, mw = mask_arr.shape[:2]
                            if mh > 0 and mw > 0:
                                # Letterbox preserves aspect, so x/y ratios
                                # should match; average for robustness.
                                nn_per_mask = 0.5 * (
                                    lb["dst_w"] / mw + lb["dst_h"] / mh
                                )
                                self._mask_px_to_crop_px_len = (
                                    nn_per_mask / lb["scale"]
                                )
                                self._mask_px_to_crop_px_area = (
                                    self._mask_px_to_crop_px_len ** 2
                                )
                                self._mask_geom_locked = True
                                print(
                                    f"[host_node] mask geometry locked: "
                                    f"shape={mw}x{mh}, "
                                    f"nn_per_mask={nn_per_mask:.4f}, "
                                    f"len_factor={self._mask_px_to_crop_px_len:.4f}, "
                                    f"area_factor={self._mask_px_to_crop_px_area:.4f}"
                                )

                        # Find contours on the binary mask
                        contours, _ = cv2.findContours(
                            mask_arr, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )

                        if contours:
                            # Use the largest contour (by area) if multiple
                            contour = max(contours, key=cv2.contourArea)

                            # Mask area from contour (more accurate than
                            # pixel count for sub-pixel contour edges)
                            mask_area_px = cv2.contourArea(contour)

                            if mask_area_px > 0:
                                # Convert mask px² → crop px²
                                area = mask_area_px * self._mask_px_to_crop_px_area

                                # minAreaRect gives the tightest oriented
                                # bounding rectangle: ((cx,cy), (w,h), angle)
                                # w and h are the two side lengths of the
                                # rotated rect — NOT axis-aligned extents.
                                if len(contour) >= 5:
                                    rect = cv2.minAreaRect(contour)
                                    rect_w, rect_h = rect[1]  # mask pixels

                                    # Convert mask pixels → crop pixels
                                    dim1 = rect_w * self._mask_px_to_crop_px_len
                                    dim2 = rect_h * self._mask_px_to_crop_px_len

                                    length = max(dim1, dim2)
                                    width = min(dim1, dim2)
                                else:
                                    # Fewer than 5 points — fall back to
                                    # axis-aligned extent of the contour
                                    x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
                                    mw_px = float(w_c) * self._mask_px_to_crop_px_len
                                    mh_px = float(h_c) * self._mask_px_to_crop_px_len
                                    length = max(mw_px, mh_px)
                                    width = min(mw_px, mh_px)
                except Exception:
                    pass  # fall back to bbox measurements

                if area < self._min_mask_area:
                    continue

                survivors.append({
                    "bbox": (x1, y1, x2, y2),
                    "cx": int(cx), "cy": int(cy),
                    "length": length, "width": width, "area": area,
                })

            # ── Pass 2: width_valid = (not edge-touching) AND (no other
            # detection's bbox IoU >= overlap threshold).  Edge-clipped or
            # occluded masks yield artificially small min-widths, so we flag
            # them out of the width aggregate while still tracking them
            # (so counting/continuity is unaffected).
            frame_dets: List[dict] = []
            for s in survivors:
                sx1, sy1, sx2, sy2 = s["bbox"]
                touches_edge = (
                    sx1 <= self._border_pad_px
                    or sy1 <= self._border_pad_px
                    or sx2 >= (self._crop_w - self._border_pad_px)
                    or sy2 >= (self._crop_h - self._border_pad_px)
                )
                max_iou_other = 0.0
                for other in survivors:
                    if other is s:
                        continue
                    iou = _calculate_iou(s["bbox"], other["bbox"])
                    if iou > max_iou_other:
                        max_iou_other = iou
                width_valid = not (
                    touches_edge
                    or max_iou_other >= self._overlap_iou_threshold
                )

                frame_dets.append({
                    "cx": s["cx"], "cy": s["cy"],
                    "length": s["length"], "width": s["width"], "area": s["area"],
                    "width_valid": width_valid,
                })

            self._dets_accepted += len(frame_dets)

            if frame_dets and self.tracker is not None:
                self.tracker.update(frame_dets)
