"""
OAK 4 Pro D DepthAI v3 pipeline — camera, ImageManip, DetectionNetwork,
and PodTrackingNode, managed as a stoppable background thread.

Optionally saves cropped camera frames to disk via a threaded writer.
Emits Qt signals for status updates so the GUI thread stays responsive.
"""

import cv2
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import depthai as dai
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from pipeline.tracking import RealtimeTracker
from pipeline.host_node import PodTrackingNode


# ═══════════════════════════════════════════════════════════════════════
# Frame writer — runs in its own thread, writes JPEGs from a queue
# ═══════════════════════════════════════════════════════════════════════

JPEG_QUALITY = 92
WRITE_QUEUE_MAX = 120


def _crop_frame(
    frame: np.ndarray,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
) -> np.ndarray:
    """Crop a frame using normalised coordinates."""
    h, w = frame.shape[:2]
    return frame[
        int(y_min * h) : int(y_max * h),
        int(x_min * w) : int(x_max * w),
    ]


def _writer_loop(q: queue.Queue):
    """Drain the write queue, saving each frame as a JPEG."""
    while True:
        item = q.get()
        if item is None:
            break
        save_path, frame = item
        cv2.imwrite(
            str(save_path), frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
        q.task_done()


class OakPipelineThread(QThread):
    """
    Runs the full DepthAI v3 pipeline in a background QThread.

    Signals
    -------
    log_message(str)
        Informational log line for the GUI console.
    status_update(int, int, float)
        (total_unique_tracks, active_tracks, fps) emitted every
        ``report_interval_s`` seconds.
    pipeline_stopped()
        Emitted when the pipeline exits cleanly.
    error_occurred(str)
        Emitted if the pipeline fails to start or crashes.
    """

    log_message = pyqtSignal(str)
    status_update = pyqtSignal(int, int, float)
    pipeline_stopped = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        cfg: dict,
        save_frames: bool = False,
        save_dir: Optional[str] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.cfg = cfg
        self._stop_requested = False

        # Frame saving
        self.save_frames = save_frames
        self.save_dir: Optional[Path] = Path(save_dir) if save_dir else None
        self.frames_saved = 0
        self.frames_dropped = 0

        # The tracker lives here so the main thread can read results
        # after the pipeline stops.
        tracker_cfg = cfg.get("tracker", {})
        self.tracker = RealtimeTracker(
            iou_threshold=tracker_cfg.get("iou_threshold", 0.5),
            distance_threshold=tracker_cfg.get("distance_threshold", 300.0),
            matching=str(tracker_cfg.get("matching", "greedy")),
            max_age=int(tracker_cfg.get("max_age", 0)),
        )

        # Reference to the host node (set during run)
        self._host_node: Optional[PodTrackingNode] = None

    def request_stop(self):
        """Ask the pipeline to shut down gracefully."""
        self._stop_requested = True

    def run(self):
        """Thread entry point — builds and runs the pipeline."""
        cfg = self.cfg
        cam_cfg = cfg.get("camera", {})
        cap_cfg = cfg.get("capture_crop", {})
        nn_cfg = cfg.get("nn", {})

        model_path = Path(cfg.get("model_path", ""))
        if not model_path.exists():
            self.error_occurred.emit(f"Model file not found: {model_path}")
            return

        device_ip = cfg.get("device_ip", "10.80.32.101")
        nn_size = nn_cfg.get("input_size", [640, 640])
        report_interval = cfg.get("report_interval_s", 2.0)

        cap_w = cam_cfg.get("width", 1280)
        cap_h = cam_cfg.get("height", 800)
        crop_w = int((cap_cfg.get("x_max", 0.85) - cap_cfg.get("x_min", 0.25)) * cap_w)
        crop_h = int((cap_cfg.get("y_max", 1.0) - cap_cfg.get("y_min", 0.35)) * cap_h)

        # Capture crop bounds for frame saving
        cx_min = cap_cfg.get("x_min", 0.25)
        cx_max = cap_cfg.get("x_max", 0.85)
        cy_min = cap_cfg.get("y_min", 0.35)
        cy_max = cap_cfg.get("y_max", 1.00)

        self.log_message.emit("=" * 55)
        self.log_message.emit("REAL-TIME POD TRACKING  |  OAK 4 Pro D  |  DepthAI v3")
        self.log_message.emit("=" * 55)
        self.log_message.emit(f"  Device       : {device_ip}")
        self.log_message.emit(f"  Model        : {model_path.name}")
        self.log_message.emit(f"  Camera       : {cap_w}x{cap_h} "
                              f"@ {cam_cfg.get('fps', 30)} fps")
        self.log_message.emit(f"  Capture crop : {crop_w}x{crop_h} px")
        self.log_message.emit(f"  NN input     : {nn_size[0]}x{nn_size[1]} (letterbox)")
        self.log_message.emit(f"  Confidence   : {nn_cfg.get('confidence_threshold', 0.7)}")
        if self.save_frames and self.save_dir:
            self.log_message.emit(f"  Save frames  : ON → {self.save_dir}")
        else:
            self.log_message.emit(f"  Save frames  : OFF")
        self.log_message.emit("=" * 55)

        # ── Set up frame writer if enabled ────────────────────────────
        write_q: Optional[queue.Queue] = None
        writer_thread: Optional[threading.Thread] = None

        if self.save_frames and self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            write_q = queue.Queue(maxsize=WRITE_QUEUE_MAX)
            writer_thread = threading.Thread(
                target=_writer_loop, args=(write_q,), daemon=True,
            )
            writer_thread.start()

        try:
            device_info = dai.DeviceInfo(device_ip)
            self.log_message.emit(f"Connecting to {device_ip}...")

            with dai.Pipeline(dai.Device(device_info)) as pipeline:

                # ── 1. Camera ─────────────────────────────────────────
                camera = pipeline.create(dai.node.Camera)
                camera.initialControl.setAutoFocusMode(
                    dai.CameraControl.AutoFocusMode.OFF
                )
                camera.initialControl.setManualFocus(
                    cam_cfg.get("manual_focus", 154)
                )
                camera.initialControl.setManualExposure(
                    cam_cfg.get("exposure_us", 700),
                    cam_cfg.get("iso", 1000),
                )
                camera = camera.build(dai.CameraBoardSocket.CAM_A)

                cam_out = camera.requestOutput(
                    size=(cap_w, cap_h),
                    type=dai.ImgFrame.Type.BGR888i,
                    fps=cam_cfg.get("fps", 30),
                )

                # ── 2. ImageManip — crop + letterbox + optional 180° flip
                manip = pipeline.create(dai.node.ImageManip)
                manip.initialConfig.addCrop(
                    dai.Rect(
                        dai.Point2f(cx_min, cy_min),
                        dai.Point2f(cx_max, cy_max),
                    ),
                    normalizedCoords=True,
                )
                manip.initialConfig.setOutputSize(
                    nn_size[0], nn_size[1],
                    dai.ImageManipConfig.ResizeMode.LETTERBOX,
                )
                # Upside-down camera mount — flip both axes (= 180°
                # rotation) so the NN sees the chute right-side up.
                # Try the rotate API first, fall back to a flip pair.
                if cam_cfg.get("rotate_180", False):
                    applied = False
                    cfg_obj = manip.initialConfig
                    for name in ("setRotation", "setRotationDeg", "addRotateDeg"):
                        m = getattr(cfg_obj, name, None)
                        if callable(m):
                            try:
                                m(180)
                                applied = True
                                break
                            except Exception:
                                continue
                    if not applied:
                        h_set = getattr(cfg_obj, "setHorizontalFlip", None)
                        v_set = getattr(cfg_obj, "setVerticalFlip", None)
                        if callable(h_set) and callable(v_set):
                            try:
                                h_set(True)
                                v_set(True)
                                applied = True
                            except Exception:
                                pass
                    if not applied:
                        self.log_message.emit(
                            "WARNING: rotate_180 requested but ImageManip "
                            "doesn't expose a flip/rotate API in this "
                            "depthai version. Image will NOT be rotated."
                        )
                manip.setMaxOutputFrameSize(nn_size[0] * nn_size[1] * 3)
                cam_out.link(manip.inputImage)

                # ── 3. DetectionNetwork ───────────────────────────────
                nn_archive = dai.NNArchive(str(model_path))
                nn = pipeline.create(dai.node.DetectionNetwork).build(
                    manip.out, nn_archive
                )
                nn.setConfidenceThreshold(0.5)

                # ── 4. PodTrackingNode ────────────────────────────────
                pod_node = pipeline.create(PodTrackingNode)
                pod_node.cfg = cfg
                pod_node.tracker = self.tracker
                nn.out.link(pod_node.input)
                self._host_node = pod_node

                # ── 5. Camera control queue ───────────────────────────
                ctrl_q = camera.inputControl.createInputQueue()

                # ── 6. Frame saving output queue (optional) ───────────
                frame_q = None
                if write_q is not None:
                    frame_q = cam_out.createOutputQueue(
                        maxSize=4, blocking=False,
                    )

                # ── Start ─────────────────────────────────────────────
                pipeline.start()
                self.log_message.emit("Pipeline started. Locking camera controls...")

                for _ in range(3):
                    ctrl = dai.CameraControl()
                    ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
                    ctrl.setManualFocus(cam_cfg.get("manual_focus", 154))
                    ctrl.setManualExposure(
                        cam_cfg.get("exposure_us", 700),
                        cam_cfg.get("iso", 1000),
                    )
                    ctrl_q.send(ctrl)
                    time.sleep(0.1)

                self.log_message.emit("Pipeline running. Tracking pods...\n")

                last_report = time.time()
                prev_frames = 0

                rotate_180 = bool(cam_cfg.get("rotate_180", False))

                while pipeline.isRunning() and not self._stop_requested:
                    # ── Pull and save camera frames ───────────────────
                    if frame_q is not None:
                        while frame_q.has():
                            raw_frame = frame_q.get().getCvFrame()
                            if rotate_180:
                                raw_frame = cv2.flip(raw_frame, -1)
                            cropped = _crop_frame(
                                raw_frame, cx_min, cx_max, cy_min, cy_max,
                            )
                            save_path = (
                                self.save_dir
                                / f"frame_{self.frames_saved:05d}.jpg"
                            )
                            try:
                                write_q.put_nowait((save_path, cropped))
                                self.frames_saved += 1
                            except queue.Full:
                                self.frames_dropped += 1

                    time.sleep(0.05)

                    now = time.time()
                    if now - last_report >= report_interval:
                        cur_frames = pod_node._frames_processed
                        elapsed = now - last_report
                        fps = (cur_frames - prev_frames) / elapsed if elapsed > 0 else 0
                        n_tracks = self.tracker.total_unique
                        n_active = self.tracker.active_count

                        self.status_update.emit(n_tracks, n_active, fps)

                        save_msg = ""
                        if write_q is not None:
                            save_msg = (
                                f" | saved {self.frames_saved}"
                                f" (q:{write_q.qsize()}"
                                f" drop:{self.frames_dropped})"
                            )

                        self.log_message.emit(
                            f"  {fps:5.1f} fps | "
                            f"frames {cur_frames} | "
                            f"dets {pod_node._dets_accepted} | "
                            f"active {n_active} | "
                            f"unique {n_tracks}"
                            f"{save_msg}"
                        )

                        prev_frames = cur_frames
                        last_report = now

                # Pipeline exiting
                self.log_message.emit("\nPipeline stopped.")

        except Exception as e:
            self.error_occurred.emit(str(e))
            return
        finally:
            # Shut down the writer thread cleanly
            if write_q is not None:
                write_q.put(None)
            if writer_thread is not None:
                writer_thread.join(timeout=10)
            if write_q is not None and self.frames_saved > 0:
                self.log_message.emit(
                    f"Frame saving complete: {self.frames_saved} saved, "
                    f"{self.frames_dropped} dropped → {self.save_dir}"
                )

        self.pipeline_stopped.emit()
