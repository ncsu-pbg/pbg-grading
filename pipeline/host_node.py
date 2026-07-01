"""
PodTrackingNode — DepthAI v3 ThreadedHostNode for mask-based
morphological measurement and per-frame tracking.

Receives ImgDetections (with on-device seg masks) from DetectionNetwork,
filters by ROI / entry zone / confidence, extracts mask measurements,
and feeds the RealtimeTracker.
"""

import time
from pathlib import Path
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

        # Optional measurement-path lens undistortion (off unless
        # undistort_measurements is set AND camera_matrix/dist_coeffs exist in
        # the calibration block). Scales measured dimensions by the local
        # undistortion magnification at each detection's centroid; positions
        # (cx,cy) are left unchanged so tracking/counting are unaffected.
        self._undistort = False
        self._K_crop = None      # 3x3 K with principal point shifted to crop origin
        self._dist = None        # distortion coeffs (k1,k2,p1,p2[,k3,...])

        # Pipeline config values (set from cfg in _init_geometry)
        self._conf_threshold = 0.70
        self._entry_zone_rel = 0.15
        self._min_mask_area = 200
        # Width-validity gating: a detection's width is excluded from the
        # min-width aggregate when its bbox touches the crop edge (clipped
        # mask) or overlaps another detection (occluded pod underneath).
        self._border_pad_px = 2
        self._overlap_iou_threshold = 0.12

        # Optional per-frame calibration log (off unless cfg.calibration_log).
        # Records per-detection crop-pixel measurements so size correction
        # (parallax + mask-fat) can be fit against calipered ground truth.
        self._calib_enabled = False
        self._calib_log = None
        self._plot_name = "run"
        self._output_dir = "./output"

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

        self._calib_enabled = bool(cfg.get("calibration_log", False))
        self._plot_name = str(cfg.get("plot_name", "run"))
        self._output_dir = str(cfg.get("output_dir", "./output"))

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

        # Measurement-path lens undistortion (optional, per-mode). camera_matrix
        # and dist_coeffs are written to the calibration block by the setup
        # dialog (device factory calibration). We shift K's principal point to
        # the crop origin so crop-pixel measurements can be undistorted directly.
        self._undistort = bool(cfg.get("undistort_measurements", False))
        self._K_crop = None
        self._dist = None
        cal = cfg.get("calibration", {}) or {}
        K = cal.get("camera_matrix")
        D = cal.get("dist_coeffs")
        if self._undistort and K is not None and D is not None:
            try:
                Kc = np.array(K, dtype=np.float64)
                Kc[0, 2] -= cx_min * cap_w
                Kc[1, 2] -= cy_min * cap_h
                dist = np.asarray(D, dtype=np.float64).reshape(-1)
                # Validate the model once so a bad coeff count can't crash the
                # per-frame loop.
                cv2.undistortPoints(
                    np.array([[[10.0, 10.0]]], np.float32), Kc, dist, P=Kc
                )
                self._K_crop = Kc
                self._dist = dist
                print("[host_node] measurement undistortion ON "
                      "(device K/dist loaded, principal point shifted to crop)")
            except Exception as e:
                print(f"[host_node] undistortion setup failed ({e}); measuring raw")
                self._K_crop = None
                self._dist = None
        elif self._undistort:
            print("[host_node] undistort_measurements set but camera_matrix/"
                  "dist_coeffs missing — measuring raw (run setup Save first)")

    def _undistort_scale(self, cx: float, cy: float, d: float = 5.0):
        """Local linear magnification of the lens-undistortion map at crop
        pixel (cx, cy). ~1.0 near the optical centre, departs from 1 toward the
        distorted FOV edges. A kernel/pod is small vs the FOV, so a single scale
        at its centroid corrects the whole silhouette to well within our error
        budget. Returns (mean_scale, x_scale, y_scale)."""
        pts = np.array(
            [[[cx, cy]], [[cx + d, cy]], [[cx, cy + d]]], dtype=np.float32
        )
        und = cv2.undistortPoints(
            pts, self._K_crop, self._dist, P=self._K_crop
        ).reshape(-1, 2)
        ax = float(np.hypot(*(und[1] - und[0])) / d)
        ay = float(np.hypot(*(und[2] - und[0])) / d)
        return 0.5 * (ax + ay), ax, ay

    # ── Calibration log (per-frame crop-pixel measurements) ───────────────
    def _open_calib_log(self):
        if not self._calib_enabled:
            return
        try:
            Path(self._output_dir).mkdir(parents=True, exist_ok=True)
            safe = "".join(
                c if c.isalnum() or c in "_-" else "_" for c in self._plot_name
            )
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = Path(self._output_dir) / f"calib_{safe}_{ts}.csv"
            self._calib_log = open(path, "w", encoding="utf-8")
            # All measurements are in CROP PIXELS (pre-mm). crop_w/crop_h let
            # the analysis normalise cx for the distortion-vs-position check.
            self._calib_log.write(f"# crop_w={self._crop_w} crop_h={self._crop_h}\n")
            self._calib_log.write(
                "frame_idx,det_idx,cx,cy,length_px,width_px,area_px,"
                "mask_meas,width_valid,conf\n"
            )
            self._calib_log.flush()
            print(f"[host_node] calibration log -> {path}")
        except Exception as e:
            print(f"[host_node] calibration log open failed: {e}")
            self._calib_log = None

    def _close_calib_log(self):
        if self._calib_log is not None:
            try:
                self._calib_log.flush()
                self._calib_log.close()
            except Exception:
                pass
            self._calib_log = None

    def run(self):
        self._init_geometry()
        self._open_calib_log()

        while self.isRunning():
            try:
                msg = self.input.get()
            except Exception:
                break

            self._frames_processed += 1
            n_dets = len(msg.detections)
            if n_dets == 0:
                # Still tick the tracker: an empty update advances track
                # ageing so stale tracks die out (and coasting tracks keep
                # moving) across detection-free frames, instead of
                # lingering at their last position waiting to steal a
                # future detection.
                if self.tracker is not None:
                    self.tracker.update([])
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
                # for true oriented length/width (not axis-aligned extent).
                # mask_ok records whether the mask path succeeded — bbox
                # fallback measurements are kept for tracking but flagged
                # so they can't pollute the per-track aggregates.
                mask_ok = False
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

                            # Mask area = pixel count of the largest blob
                            # (holes filled).  cv2.contourArea integrates
                            # the polygon through pixel centres and
                            # under-counts small blobs by ~perimeter/2 px².
                            blob = np.zeros_like(mask_arr)
                            cv2.drawContours(
                                blob, [contour], -1, 1, thickness=cv2.FILLED
                            )
                            mask_area_px = float(cv2.countNonZero(blob))

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
                                mask_ok = True
                except Exception:
                    pass  # fall back to bbox measurements

                # Optional lens-distortion correction: scale measured
                # dimensions by the local undistortion magnification at the
                # detection centroid. Positions (cx,cy) are left unchanged so
                # tracking/counting behaviour is identical. No-op unless enabled.
                if self._undistort and self._K_crop is not None:
                    try:
                        mscale, ax, ay = self._undistort_scale(cx, cy)
                        length *= mscale
                        width *= mscale
                        area *= ax * ay
                    except Exception:
                        pass

                if area < self._min_mask_area:
                    continue

                survivors.append({
                    "bbox": (x1, y1, x2, y2),
                    "cx": int(cx), "cy": int(cy),
                    "length": length, "width": width, "area": area,
                    "mask_ok": mask_ok,
                    "conf": float(det.confidence),
                })

            # ── Pass 2: width_valid = (not edge-touching) AND (no other
            # detection's bbox IoU >= overlap threshold).  Edge-clipped or
            # occluded masks yield artificially small min-widths, so we flag
            # them out of the width aggregate while still tracking them
            # (so counting/continuity is unaffected).
            frame_dets: List[dict] = []
            for det_idx, s in enumerate(survivors):
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
                # Bbox-fallback widths are axis-aligned extents, not
                # oriented widths — exclude them from the min-width
                # aggregate alongside edge-clipped/occluded masks.
                width_valid = s["mask_ok"] and not (
                    touches_edge
                    or max_iou_other >= self._overlap_iou_threshold
                )

                frame_dets.append({
                    "cx": s["cx"], "cy": s["cy"],
                    "length": s["length"], "width": s["width"], "area": s["area"],
                    "width_valid": width_valid,
                    "mask_meas": s["mask_ok"],
                })

                if self._calib_log is not None:
                    try:
                        self._calib_log.write(
                            f"{self._frames_processed},{det_idx},"
                            f"{s['cx']},{s['cy']},"
                            f"{s['length']:.3f},{s['width']:.3f},{s['area']:.2f},"
                            f"{int(s['mask_ok'])},{int(width_valid)},{s['conf']:.4f}\n"
                        )
                    except Exception as e:
                        # Never let a logging hiccup take down tracking.
                        print(f"[host_node] calib log write failed ({e}); disabling log")
                        self._close_calib_log()

            # Flush once per frame to the OS so a depthai native teardown crash
            # (which skips our clean close) can't lose the whole recording.
            if self._calib_log is not None:
                try:
                    self._calib_log.flush()
                except Exception:
                    pass

            self._dets_accepted += len(frame_dets)

            # Always tick the tracker, even when every detection was
            # filtered out, so track ageing advances on every frame.
            if self.tracker is not None:
                self.tracker.update(frame_dets)

        self._close_calib_log()
