"""
Unified worker for the Initial Camera Setup dialog.

A single OAK device can only host one connection at a time, so this
worker builds a single pipeline that does everything the Initial
Camera Setup dialog needs:

  • IMU accelerometer streaming → pitch / roll readout
  • Camera streaming → live preview, manual-focus control
  • Sharpness measurement (Laplacian variance) on the marker region
    when both ArUco markers are detected, otherwise on a centre crop
  • Focus sweep — when requested, sets focus to each value in a small
    range, lets the lens settle, samples sharpness, and reports the
    peak value

Pitch / roll convention is the same as the legacy IMUWorker:
    pitch = atan2(accel_y, accel_z)   — image-up/down tilt
    roll  = atan2(accel_x, accel_z)   — image-left/right tilt
"""

import time
from collections import deque
from typing import Any, Deque, List, Optional, Tuple

import cv2
import depthai as dai
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage


# ── Tuning constants ───────────────────────────────────────────────────────

IMU_REPORT_RATE_HZ  = 100
IMU_AVG_WINDOW      = 50           # ~0.5 s at 100 Hz
GRAVITY_MAG_MIN     = 7.0
GRAVITY_MAG_MAX     = 12.5

CAM_FPS             = 15           # live preview FPS

# Sweep step pacing — give the lens time to settle before measuring.
SWEEP_SETTLE_FRAMES = 4            # frames to discard after focus change
SWEEP_MEASURE_FRAMES = 3           # frames averaged at each focus value


# ── ArUco detector ─────────────────────────────────────────────────────────

ARUCO_DICT_ID  = cv2.aruco.DICT_4X4_50

def _make_detector() -> Any:
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    p = cv2.aruco.DetectorParameters()
    try:
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    except AttributeError:
        pass
    try:
        return cv2.aruco.ArucoDetector(d, p)
    except AttributeError:
        return (d, p)

def _detect_markers(detector: Any, gray: np.ndarray):
    if isinstance(detector, tuple):
        d, p = detector
        return cv2.aruco.detectMarkers(gray, d, parameters=p)
    return detector.detectMarkers(gray)


# ── Sharpness measurement ──────────────────────────────────────────────────

def _laplacian_variance(gray: np.ndarray) -> float:
    """
    Standard image-sharpness metric: variance of the Laplacian. Higher
    values mean sharper. Computed on whatever region of the gray frame
    is passed in — the dialog wants this on the marker bounding box if
    available, or a centre crop otherwise.
    """
    if gray.size == 0:
        return 0.0
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _sharpness_roi_bgr(
    bgr: np.ndarray,
    marker_pts: List[np.ndarray],
) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
    """
    Pick a sharpness ROI: bounding box of all detected marker corners
    (with a small margin) if any markers were found, otherwise a 25 %
    centre crop. Returns (sharpness, (x1, y1, x2, y2)) or (sharpness,
    None) if the frame was empty.
    """
    h, w = bgr.shape[:2]
    if marker_pts:
        all_pts = np.vstack(marker_pts)
        x1 = max(0, int(all_pts[:, 0].min()) - 8)
        y1 = max(0, int(all_pts[:, 1].min()) - 8)
        x2 = min(w, int(all_pts[:, 0].max()) + 8)
        y2 = min(h, int(all_pts[:, 1].max()) + 8)
    else:
        cx, cy = w // 2, h // 2
        half = min(w, h) // 4
        x1, y1 = cx - half, cy - half
        x2, y2 = cx + half, cy + half
    if x2 <= x1 or y2 <= y1:
        return 0.0, None
    crop = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return _laplacian_variance(crop), (x1, y1, x2, y2)


# ── Frame annotation ───────────────────────────────────────────────────────

_GREEN  = (0, 220,   0)
_ORANGE = (0, 165, 255)
_WHITE  = (255, 255, 255)


def _annotate(
    bgr: np.ndarray,
    marker_pts: List[np.ndarray],
    sharp_roi: Optional[Tuple[int, int, int, int]],
    sharpness: float,
    focus: int,
    sweeping: bool,
) -> np.ndarray:
    out = bgr.copy()
    for pts in marker_pts:
        cv2.polylines(
            out, [pts.astype(np.int32)],
            isClosed=True, color=_GREEN, thickness=2,
        )
    if sharp_roi is not None:
        x1, y1, x2, y2 = sharp_roi
        cv2.rectangle(out, (x1, y1), (x2, y2), _ORANGE, 1)

    def _put(text, org, color, scale=0.6):
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,    1, cv2.LINE_AA)

    _put(f"Focus: {focus}",          (20, 32),  _WHITE)
    _put(f"Sharpness: {sharpness:.0f}", (20, 58),  _WHITE)
    _put(f"Markers: {len(marker_pts)}", (20, 84), _GREEN if marker_pts else _ORANGE)
    if sweeping:
        _put("SWEEPING…", (20, 110), _ORANGE, scale=0.65)
    return out


def _to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()


# ══════════════════════════════════════════════════════════════════════════════
# Worker
# ══════════════════════════════════════════════════════════════════════════════

class InitialSetupWorker(QThread):
    """
    Combined IMU + camera worker for the Initial Camera Setup dialog.

    Signals
    -------
    frame_ready(QImage, float, int)
        annotated_frame, sharpness, current_focus
    imu_ready(pitch_deg, roll_deg, n_samples)
    sweep_progress(int focus, float sharpness)
    sweep_complete(int best_focus, float best_sharpness)
    log_message(str)
    error_occurred(str)
    worker_stopped()

    Cross-thread control (host calls these from the GUI thread)
    -----------------------------------------------------------
    set_focus(int)         — request a focus change; takes effect on the
                              next loop tick.
    request_sweep(low, high) — schedule a focus sweep.
    request_stop()
    """

    frame_ready    = pyqtSignal(QImage, float, int)
    imu_ready      = pyqtSignal(float, float, int)
    sweep_progress = pyqtSignal(int, float)
    sweep_complete = pyqtSignal(int, float)
    log_message    = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    worker_stopped = pyqtSignal()

    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._stop_requested = False
        # Cross-thread control state. Plain attributes are fine — a
        # missed update gets picked up on the next loop tick, which is
        # 15 Hz, so latency is sub-100 ms.
        self._pending_focus: Optional[int] = None
        self._pending_sweep: Optional[Tuple[int, int]] = None
        self._current_focus: int = int(
            (cfg.get("camera") or {}).get("manual_focus", 154)
        )

    # ── External API (called from GUI thread) ─────────────────────────────

    def request_stop(self):
        self._stop_requested = True

    def set_focus(self, value: int):
        v = max(0, min(255, int(value)))
        self._pending_focus = v

    def request_sweep(self, low: int, high: int):
        lo = max(0, min(255, int(low)))
        hi = max(0, min(255, int(high)))
        if hi < lo:
            lo, hi = hi, lo
        self._pending_sweep = (lo, hi)

    # ── Pipeline / loop ───────────────────────────────────────────────────

    def run(self):
        cam_cfg    = self.cfg.get("camera", {}) or {}
        cap_w      = int(cam_cfg.get("width",  1280))
        cap_h      = int(cam_cfg.get("height",  800))
        device_ip  = self.cfg.get("device_ip", "")
        rotate_180 = bool(cam_cfg.get("rotate_180", False))

        try:
            device_info = dai.DeviceInfo(device_ip)
        except Exception as exc:
            self.error_occurred.emit(f"Invalid device IP '{device_ip}': {exc}")
            self.worker_stopped.emit()
            return

        detector = _make_detector()
        imu_buf: Deque[Tuple[float, float, float]] = deque(maxlen=IMU_AVG_WINDOW)

        try:
            with dai.Pipeline(dai.Device(device_info)) as pipeline:
                # ── Camera node ───────────────────────────────────────
                camera = pipeline.create(dai.node.Camera)
                camera.initialControl.setAutoFocusMode(
                    dai.CameraControl.AutoFocusMode.OFF
                )
                camera.initialControl.setManualFocus(self._current_focus)
                camera.initialControl.setManualExposure(
                    int(cam_cfg.get("exposure_us", 700)),
                    int(cam_cfg.get("iso", 1000)),
                )
                camera = camera.build(dai.CameraBoardSocket.CAM_A)
                cam_out = camera.requestOutput(
                    size=(cap_w, cap_h),
                    type=dai.ImgFrame.Type.BGR888i,
                    fps=CAM_FPS,
                )
                frame_q   = cam_out.createOutputQueue(maxSize=2, blocking=False)
                control_q = camera.inputControl.createInputQueue()

                # ── IMU node ─────────────────────────────────────────
                imu = pipeline.create(dai.node.IMU)
                imu.enableIMUSensor(
                    dai.IMUSensor.ACCELEROMETER_RAW, IMU_REPORT_RATE_HZ,
                )
                imu.setBatchReportThreshold(1)
                imu.setMaxBatchReports(10)
                imu_q = imu.out.createOutputQueue(maxSize=20, blocking=False)

                pipeline.start()
                self.log_message.emit(
                    f"Camera + IMU streaming. Initial focus = {self._current_focus}."
                )

                while pipeline.isRunning() and not self._stop_requested:
                    # 1. Apply pending focus change (immediate response)
                    if self._pending_focus is not None:
                        self._apply_focus(control_q, self._pending_focus)
                        self._pending_focus = None

                    # 2. Pump IMU samples
                    self._drain_imu(imu_q, imu_buf)

                    # 3. Pump a frame
                    if frame_q.has():
                        bgr = frame_q.get().getCvFrame()
                        if rotate_180:
                            bgr = cv2.flip(bgr, -1)
                        sharpness, roi, marker_pts = self._analyse_frame(
                            bgr, detector,
                        )
                        annotated = _annotate(
                            bgr, marker_pts, roi,
                            sharpness, self._current_focus,
                            sweeping=False,
                        )
                        self.frame_ready.emit(
                            _to_qimage(annotated), sharpness, self._current_focus,
                        )

                    # 4. Run a sweep if scheduled (blocks the loop until done)
                    if self._pending_sweep is not None:
                        lo, hi = self._pending_sweep
                        self._pending_sweep = None
                        self._run_sweep(
                            lo, hi, frame_q, control_q, detector,
                            rotate_180=rotate_180,
                        )

                    time.sleep(0.005)

        except Exception as exc:
            self.error_occurred.emit(str(exc))
        finally:
            self.worker_stopped.emit()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _apply_focus(self, control_q, value: int):
        ctrl = dai.CameraControl()
        ctrl.setManualFocus(int(value))
        try:
            control_q.send(ctrl)
        except Exception as exc:
            self.log_message.emit(f"Focus change failed: {exc}")
            return
        self._current_focus = int(value)

    def _drain_imu(self, imu_q, buf: Deque[Tuple[float, float, float]]):
        if not imu_q.has():
            return
        data = imu_q.get()
        for packet in getattr(data, "packets", []):
            accel = getattr(packet, "acceleroMeter", None)
            if accel is None:
                continue
            try:
                ax, ay, az = float(accel.x), float(accel.y), float(accel.z)
            except AttributeError:
                continue
            mag = (ax * ax + ay * ay + az * az) ** 0.5
            if mag < GRAVITY_MAG_MIN or mag > GRAVITY_MAG_MAX:
                continue
            buf.append((ax, ay, az))
        if buf:
            arr = np.asarray(buf, dtype=np.float64)
            mx, my, mz = arr.mean(axis=0)
            pitch = float(np.degrees(np.arctan2(my, mz)))
            roll  = float(np.degrees(np.arctan2(mx, mz)))
            self.imu_ready.emit(pitch, roll, len(buf))

    def _analyse_frame(
        self, bgr: np.ndarray, detector: Any,
    ) -> Tuple[float, Optional[Tuple[int, int, int, int]], List[np.ndarray]]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners_all, ids_all, _ = _detect_markers(detector, gray)
        marker_pts: List[np.ndarray] = []
        if ids_all is not None:
            for idx, mid in enumerate(ids_all.flatten()):
                if int(mid) in (0, 1):
                    marker_pts.append(
                        corners_all[idx].reshape(4, 2).astype(np.float64)
                    )
        sharpness, roi = _sharpness_roi_bgr(bgr, marker_pts)
        return sharpness, roi, marker_pts

    def _run_sweep(
        self,
        lo: int, hi: int,
        frame_q, control_q,
        detector: Any,
        rotate_180: bool = False,
    ):
        """
        Step focus from lo to hi inclusive (stride 1). At each value:
          - send CameraControl to apply focus
          - discard SWEEP_SETTLE_FRAMES frames so the lens settles
          - average sharpness across SWEEP_MEASURE_FRAMES frames
          - emit sweep_progress
        Then snap focus to the value with peak averaged sharpness and
        emit sweep_complete.
        """
        self.log_message.emit(f"Focus sweep starting: {lo} → {hi}")
        results: List[Tuple[int, float]] = []

        for f in range(lo, hi + 1):
            if self._stop_requested:
                self.log_message.emit("Sweep cancelled (stop requested).")
                return
            self._apply_focus(control_q, f)

            # Discard settle frames
            discarded = 0
            t0 = time.time()
            while discarded < SWEEP_SETTLE_FRAMES and (time.time() - t0) < 1.0:
                if frame_q.has():
                    _ = frame_q.get()
                    discarded += 1
                else:
                    time.sleep(0.01)

            # Measure
            sums = 0.0
            taken = 0
            t0 = time.time()
            while taken < SWEEP_MEASURE_FRAMES and (time.time() - t0) < 1.0:
                if frame_q.has():
                    bgr = frame_q.get().getCvFrame()
                    if rotate_180:
                        bgr = cv2.flip(bgr, -1)
                    s, roi, marker_pts = self._analyse_frame(bgr, detector)
                    sums += s
                    taken += 1
                    annotated = _annotate(
                        bgr, marker_pts, roi, s, f, sweeping=True,
                    )
                    self.frame_ready.emit(_to_qimage(annotated), s, f)
                else:
                    time.sleep(0.01)
            avg = sums / max(1, taken)
            results.append((f, avg))
            self.sweep_progress.emit(f, avg)

        if not results:
            return
        best_f, best_s = max(results, key=lambda r: r[1])
        self.log_message.emit(
            f"Sweep complete. Peak sharpness {best_s:.0f} at focus {best_f}."
        )
        self._apply_focus(control_q, best_f)
        self.sweep_complete.emit(best_f, best_s)
