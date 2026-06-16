"""
Target Calibration Dialog — px/mm scale and colour-correction matrix
using the dual-marker ColorChecker target.

This dialog handles the two pieces of calibration that need the printed
target. Camera-mount tilt is verified separately by `MountSetupDialog`
(IMU-based) and is NOT gated here.

What it does
------------
  • streams live camera frames with overlays (markers, ROI, patch dots)
  • computes px/mm from 8 marker side lengths + the 14 mm C-C distance
  • once the marker pose is stable enough to project patch positions,
    samples each patch's mean BGR and fits a linear-camera-RGB → CIE
    XYZ D65 colour-correction matrix (5-frame rolling average)
  • saves px/mm, mm/px, color_matrix, color_bias, capture_crop, and
    tracking_roi into the active mode's per-mode config block

Save criteria
-------------
  • both markers detected
  • px/mm computed
  • colour calibration computed (24-patch fit, RMS reported)

The "tilt < 1°" / "ROI in frame" / "device intrinsics required" gates
that earlier versions of this dialog enforced are gone:
  • Tilt acceptance moved to MountSetupDialog (IMU-based).
  • ROI-in-frame is informational; capture_crop is clamped to frame
    bounds at save time anyway.
  • Approximate intrinsics still produce a usable colour cal — patch
    refinement compensates for the small projection offset, and px/mm
    doesn't depend on intrinsics at all. A warning is shown but save
    is allowed.

Usage (from MainWindow):
    dlg = SetupDialog(active_cfg, full_cfg, config_path, parent=self)
    dlg.calibration_saved.connect(self._on_calibration_saved)
    dlg.exec()
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from calibration import Calibration
from pipeline.camera_setup_worker import (
    COLOR_WINDOW,
    CameraSetupWorker,
)


# ── Colour helpers ─────────────────────────────────────────────────────────────

_STYLE_OK   = "color: #00cc00; font-weight: bold;"
_STYLE_WARN = "color: #ff8800; font-weight: bold;"
_STYLE_BAD  = "color: #cc0000; font-weight: bold;"
_STYLE_GREY = "color: #888888;"
_STYLE_TEXT = "color: #dddddd;"
_STYLE_DIM  = "color: #aaaaaa;"

_VIDEO_W = 800
_VIDEO_H = 500


class _ClickableVideoLabel(QLabel):
    """QLabel that emits clicked(x, y) with widget-local pixel coordinates."""
    clicked = pyqtSignal(int, int)

    def mousePressEvent(self, ev: QMouseEvent):
        if ev.button() == Qt.MouseButton.LeftButton:
            pos = ev.position()
            self.clicked.emit(int(pos.x()), int(pos.y()))
        super().mousePressEvent(ev)


# ══════════════════════════════════════════════════════════════════════════════
# SetupDialog
# ══════════════════════════════════════════════════════════════════════════════

class SetupDialog(QDialog):
    """
    Modal dialog for camera alignment, px/mm calibration, and colour calibration.

    Signals
    -------
    calibration_saved(Calibration)
        Emitted when the user clicks "Save Calibration".
    """

    calibration_saved = pyqtSignal(object)   # Calibration dataclass

    def __init__(
        self,
        cfg: dict,
        full_cfg: Optional[dict] = None,
        config_path: str | Path = "config.yaml",
        parent=None,
    ):
        super().__init__(parent)
        # cfg is the per-mode subdict that this dialog reads from and
        # writes calibration into; full_cfg is the entire loaded config
        # that gets dumped back to disk on save (so both modes round-
        # trip through the file). If full_cfg is None we fall back to
        # writing cfg alone — supports older single-mode usage.
        self.cfg = cfg
        self.full_cfg = full_cfg if full_cfg is not None else cfg
        self.config_path = Path(config_path)
        self._last_metrics: Optional[Dict] = None
        self._last_qimage = None

        # Click-to-set capture_crop state
        self._setting_crop = False
        self._crop_first_pt: Optional[Tuple[float, float]] = None  # normalised
        self._frame_native_size: Tuple[int, int] = (1280, 800)

        self.setWindowTitle("Target Calibration — px/mm Scale & Colour Matrix")
        self.setMinimumSize(1060, 700)
        self.setModal(True)
        self.setStyleSheet("background-color: #2b2b2b; color: #dddddd;")

        self._build_ui()

        self._worker = CameraSetupWorker(cfg, parent=self)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.metrics_ready.connect(self._on_metrics)
        self._worker.log_message.connect(self._set_status)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.worker_stopped.connect(self._on_worker_stopped)
        self._worker.start()

    # ══════════════════════════════════════════════════════════════════════
    # UI construction
    # ══════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(12, 12, 12, 12)

        # ── Left: video feed ───────────────────────────────────────────────
        left = QVBoxLayout()

        self._video_label = _ClickableVideoLabel()
        self._video_label.setFixedSize(_VIDEO_W, _VIDEO_H)
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setStyleSheet(
            "background-color: #111111; border: 1px solid #444;"
        )
        self._video_label.setText("Connecting to camera…")
        self._video_label.clicked.connect(self._on_video_click)
        left.addWidget(self._video_label)

        # Set Capture Crop button under the video — toggles two-click mode
        crop_btn_row = QHBoxLayout()
        self._set_crop_btn = QPushButton("Set Capture Crop")
        self._set_crop_btn.setFixedHeight(30)
        self._set_crop_btn.setCheckable(True)
        self._set_crop_btn.setToolTip(
            "Click to start, then click two corners on the live preview "
            "(top-left, then bottom-right). Click again to cancel."
        )
        self._set_crop_btn.clicked.connect(self._on_set_crop_toggled)
        self._reset_crop_btn = QPushButton("Reset to Full Frame")
        self._reset_crop_btn.setFixedHeight(30)
        self._reset_crop_btn.setToolTip("Reset capture_crop to full frame.")
        self._reset_crop_btn.clicked.connect(self._on_reset_crop)
        crop_btn_row.addWidget(self._set_crop_btn)
        crop_btn_row.addWidget(self._reset_crop_btn)
        crop_btn_row.addStretch()
        left.addLayout(crop_btn_row)

        self._status_label = QLabel("Initialising…")
        self._status_label.setStyleSheet(_STYLE_GREY)
        self._status_label.setFont(QFont("Consolas", 9))
        left.addWidget(self._status_label)

        root.addLayout(left)

        # ── Right: status panel + buttons ─────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(8)

        # Intrinsics source indicator
        self._intrin_label = QLabel("Intrinsics: loading…")
        self._intrin_label.setFont(QFont("Consolas", 8))
        self._intrin_label.setStyleSheet(_STYLE_GREY)
        self._intrin_label.setWordWrap(True)
        right.addWidget(self._intrin_label)

        # Status group
        status_group = QGroupBox("Setup Status")
        status_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #555; "
            "margin-top: 6px; padding-top: 6px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        )
        status_group.setFixedWidth(240)
        sg_layout = QVBoxLayout(status_group)
        sg_layout.setSpacing(6)

        self._row_markers   = _MetricRow("Markers")
        self._row_pxmm      = _MetricRow("px / mm")
        self._row_mmpx      = _MetricRow("mm / px")
        self._row_crop_x    = _MetricRow("crop x")
        self._row_crop_y    = _MetricRow("crop y")
        self._row_color     = _MetricRow("Colour patches")
        self._row_color_rms = _MetricRow("Colour fit RMS")

        for widget in (
            self._row_markers,
            _HSep(),
            self._row_pxmm, self._row_mmpx,
            _HSep(),
            self._row_crop_x, self._row_crop_y,
            _HSep(),
            self._row_color, self._row_color_rms,
        ):
            sg_layout.addWidget(widget)

        sg_layout.addStretch()
        right.addWidget(status_group)

        # Guidance text
        guide = QLabel(
            f"Place the calibration target flat in view.\n\n"
            f"Both ArUco markers (IDs 0 & 1) must be visible.\n\n"
            f"Colour calibration starts automatically\n"
            f"once the marker pose stabilises\n"
            f"({COLOR_WINDOW} colour frames needed).\n\n"
            f"Camera tilt is verified separately\n"
            f"in the Mount Setup dialog (IMU-based)."
        )
        guide.setStyleSheet(_STYLE_GREY)
        guide.setFont(QFont("Consolas", 8))
        guide.setWordWrap(True)
        right.addWidget(guide)

        right.addStretch()

        # Save button
        self._save_btn = QPushButton("Save Calibration")
        self._save_btn.setFixedHeight(38)
        self._save_btn.setEnabled(False)
        self._save_btn.setToolTip(
            "Enabled when both markers found and colour calibration computed."
        )
        self._save_btn.clicked.connect(self._on_save)
        right.addWidget(self._save_btn)

        close_btn = QPushButton("Close")
        close_btn.setFixedHeight(38)
        close_btn.clicked.connect(self.reject)
        right.addWidget(close_btn)

        root.addLayout(right)

    # ══════════════════════════════════════════════════════════════════════
    # Worker signal handlers
    # ══════════════════════════════════════════════════════════════════════

    def _on_frame(self, qimage):
        self._last_qimage = qimage
        # Track native frame size so click-to-set-crop can convert
        # widget-local coordinates back to normalised frame coordinates.
        self._frame_native_size = (qimage.width(), qimage.height())
        pix = QPixmap.fromImage(qimage).scaled(
            _VIDEO_W, _VIDEO_H,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._video_label.setPixmap(pix)

    # ── Capture-crop click-to-set ─────────────────────────────────────────

    def _displayed_pixmap_geometry(self) -> Tuple[int, int, int, int]:
        """
        Return (x_off, y_off, disp_w, disp_h) describing where the scaled
        pixmap is drawn inside the QLabel. With KeepAspectRatio the pixmap
        is centred and may be letterboxed when the aspect ratios differ.
        """
        fw, fh = self._frame_native_size
        if fw <= 0 or fh <= 0:
            return 0, 0, _VIDEO_W, _VIDEO_H
        scale = min(_VIDEO_W / fw, _VIDEO_H / fh)
        disp_w = int(fw * scale)
        disp_h = int(fh * scale)
        x_off = (_VIDEO_W - disp_w) // 2
        y_off = (_VIDEO_H - disp_h) // 2
        return x_off, y_off, disp_w, disp_h

    def _widget_to_norm(self, wx: int, wy: int) -> Optional[Tuple[float, float]]:
        x_off, y_off, disp_w, disp_h = self._displayed_pixmap_geometry()
        if disp_w <= 0 or disp_h <= 0:
            return None
        x = wx - x_off
        y = wy - y_off
        if x < 0 or y < 0 or x > disp_w or y > disp_h:
            return None
        return (x / disp_w, y / disp_h)

    def _on_set_crop_toggled(self, checked: bool):
        if checked:
            self._setting_crop = True
            self._crop_first_pt = None
            self._set_crop_btn.setText("Click TOP-LEFT corner…")
            self._video_label.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self._setting_crop = False
            self._crop_first_pt = None
            self._set_crop_btn.setText("Set Capture Crop")
            self._video_label.unsetCursor()

    def _on_video_click(self, wx: int, wy: int):
        if not self._setting_crop:
            return
        pt = self._widget_to_norm(wx, wy)
        if pt is None:
            return
        if self._crop_first_pt is None:
            self._crop_first_pt = pt
            self._set_crop_btn.setText("Click BOTTOM-RIGHT corner…")
            return
        # Second click — finalise.
        x1, y1 = self._crop_first_pt
        x2, y2 = pt
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        # Reject degenerate rectangles
        if (x_max - x_min) < 0.02 or (y_max - y_min) < 0.02:
            self._set_status("Crop too small — try again with bigger corners.")
            self._on_set_crop_toggled(False)
            self._set_crop_btn.setChecked(False)
            return
        self.cfg["capture_crop"] = {
            "x_min": round(x_min, 6),
            "x_max": round(x_max, 6),
            "y_min": round(y_min, 6),
            "y_max": round(y_max, 6),
        }
        self._set_status(
            f"capture_crop set to x[{x_min:.3f}–{x_max:.3f}] "
            f"y[{y_min:.3f}–{y_max:.3f}] (saved on Save Calibration)"
        )
        self._on_set_crop_toggled(False)
        self._set_crop_btn.setChecked(False)

    def _on_reset_crop(self):
        self.cfg["capture_crop"] = {
            "x_min": 0.0, "x_max": 1.0,
            "y_min": 0.0, "y_max": 1.0,
        }
        self._set_status(
            "capture_crop reset to full frame (saved on Save Calibration)."
        )

    def _on_metrics(self, metrics: Dict):
        self._last_metrics = metrics
        n_markers    = metrics.get("n_markers", 0)
        approx       = metrics.get("approx_intrinsics", True)
        px_per_mm    = metrics.get("px_per_mm")
        color_sampled = metrics.get("color_sampled", False)
        color_rms    = metrics.get("color_rms")
        color_frames = metrics.get("color_frames", 0)

        # ── Intrinsics source ──────────────────────────────────────────────
        if approx:
            self._intrin_label.setText(
                "Intrinsics: APPROXIMATE\n"
                "Colour cal still works (patch refinement compensates)."
            )
            self._intrin_label.setStyleSheet(_STYLE_WARN)
        else:
            self._intrin_label.setText("Intrinsics: device calibration ✓")
            self._intrin_label.setStyleSheet(_STYLE_OK)

        # ── Markers ───────────────────────────────────────────────────────
        if n_markers == 2:
            self._row_markers.set_value("2 / 2 ✓", _STYLE_OK)
        elif n_markers == 1:
            self._row_markers.set_value("1 / 2 !", _STYLE_WARN)
        else:
            self._row_markers.set_value("0 / 2", _STYLE_BAD)

        # ── px/mm ──────────────────────────────────────────────────────────
        if px_per_mm is not None:
            mm_per_px = metrics.get("mm_per_px")
            self._row_pxmm.set_value(f"{px_per_mm:.4f}", _STYLE_TEXT)
            self._row_mmpx.set_value(f"{mm_per_px:.6f}", _STYLE_TEXT)
        else:
            self._row_pxmm.set_value("—", _STYLE_GREY)
            self._row_mmpx.set_value("—", _STYLE_GREY)

        # ── capture_crop readout ──────────────────────────────────────────
        cap = self.cfg.get("capture_crop") or {}
        x_min = float(cap.get("x_min", 0.0))
        x_max = float(cap.get("x_max", 1.0))
        y_min = float(cap.get("y_min", 0.0))
        y_max = float(cap.get("y_max", 1.0))
        is_full = (x_min <= 0.001 and x_max >= 0.999 and
                   y_min <= 0.001 and y_max >= 0.999)
        crop_style = _STYLE_DIM if is_full else _STYLE_TEXT
        self._row_crop_x.set_value(f"[{x_min:.3f}–{x_max:.3f}]", crop_style)
        self._row_crop_y.set_value(f"[{y_min:.3f}–{y_max:.3f}]", crop_style)

        # ── Colour calibration ─────────────────────────────────────────────
        if color_sampled:
            rms_str = f"{color_rms:.2f} %" if color_rms is not None else "?"
            self._row_color.set_value("24 / 24 ✓", _STYLE_OK)
            self._row_color_rms.set_value(rms_str, _STYLE_OK)
        elif color_frames > 0:
            self._row_color.set_value(
                f"Sampling {color_frames}/{COLOR_WINDOW}…", _STYLE_WARN
            )
            self._row_color_rms.set_value("—", _STYLE_GREY)
        else:
            self._row_color.set_value("—", _STYLE_GREY)
            self._row_color_rms.set_value("—", _STYLE_GREY)

        # ── Save button guard ─────────────────────────────────────────────
        # Tilt is no longer gated here — that's MountSetupDialog's job.
        # Approx intrinsics are tolerated: px/mm doesn't depend on them
        # at all, and patch refinement compensates for the small
        # projection offset they introduce in the colour cal step.
        # ROI clipping is informational; capture_crop is clamped to frame
        # bounds at save time.
        can_save = (
            n_markers == 2
            and px_per_mm is not None
            and color_sampled is True
        )
        self._save_btn.setEnabled(can_save)

        if n_markers < 2:
            self._save_btn.setToolTip(
                f"Cannot save: need both markers ({n_markers}/2 found)."
            )
        elif not color_sampled:
            self._save_btn.setToolTip(
                f"Cannot save: waiting for colour calibration "
                f"({color_frames}/{COLOR_WINDOW} frames collected). "
                "Keep target still."
            )
        else:
            self._save_btn.setToolTip("Save target calibration to config.yaml.")

        # ── Status line ───────────────────────────────────────────────────
        if n_markers == 0:
            self._set_status("Waiting for calibration target…")
        elif n_markers == 1:
            self._set_status("Only 1 marker visible — move target so both are in frame.")
        elif not color_sampled:
            self._set_status(
                f"Colour calibration in progress… "
                f"{color_frames}/{COLOR_WINDOW} frames — keep target still."
            )
        elif can_save:
            rms_str  = f"{color_rms:.2f}%" if color_rms is not None else "?"
            self._set_status(
                f"Ready to save — px/mm {px_per_mm:.4f}, colour RMS {rms_str}"
            )
        else:
            self._set_status("Conditions met — click Save Calibration.")

    def _on_error(self, message: str):
        self._set_status(f"Error: {message}")
        QMessageBox.critical(self, "Camera Error", message)

    def _on_worker_stopped(self):
        self._set_status("Camera disconnected.")

    # ══════════════════════════════════════════════════════════════════════
    # Save calibration
    # ══════════════════════════════════════════════════════════════════════

    def _on_save(self):
        m = self._last_metrics
        if m is None or m.get("px_per_mm") is None:
            return

        px_per_mm: float = m["px_per_mm"]
        mm_per_px: float = m["mm_per_px"]

        # ── px/mm calibration ─────────────────────────────────────────────
        if "calibration" not in self.cfg:
            self.cfg["calibration"] = {}
        self.cfg["calibration"]["mm_per_px"] = round(mm_per_px, 6)
        self.cfg["calibration"]["px_per_mm"] = round(px_per_mm, 4)

        # ── Colour correction matrix ──────────────────────────────────────
        color_matrix = self._worker.color_matrix
        color_bias   = self._worker.color_bias
        color_note   = ""
        if color_matrix is not None and color_bias is not None:
            self.cfg["calibration"]["color_matrix"] = color_matrix.tolist()
            self.cfg["calibration"]["color_bias"]   = color_bias.tolist()
            rms = m.get("color_rms") or 0.0
            color_note = f"\n  colour matrix saved  (RMS {rms:.2f}%)"
        else:
            color_note = "\n  colour matrix: NOT SAVED (not yet computed)"

        # capture_crop is set explicitly via the "Set Capture Crop"
        # button (drag two corners on the live preview); we don't
        # auto-compute it here from the markers any more. It's already
        # in self.cfg, so it round-trips through the yaml dump below.
        # tracking_roi has been removed from the runtime — the entry
        # zone is now the top entry_zone_threshold_rel of capture_crop.
        cap = self.cfg.get("capture_crop", {})
        if cap:
            roi_note = (
                f"\n  capture_crop = "
                f"x [{cap.get('x_min', 0):.4f}–{cap.get('x_max', 1):.4f}]  "
                f"y [{cap.get('y_min', 0):.4f}–{cap.get('y_max', 1):.4f}]"
            )
        else:
            roi_note = ""
        # Drop any stale tracking_roi key left over from older configs.
        self.cfg.pop("tracking_roi", None)

        # ── Write config.yaml ─────────────────────────────────────────────
        # self.cfg is a reference into self.full_cfg["modes"][mode], so
        # the in-place edits above are already visible in full_cfg.
        try:
            with open(self.config_path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(
                    self.full_cfg, fh,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
            file_note = f"\nSaved to: {self.config_path}"
        except Exception as exc:
            file_note = f"\n(Could not write config: {exc})"

        # ── Emit signal ───────────────────────────────────────────────────
        cm_tuple = (
            tuple(tuple(float(v) for v in row) for row in color_matrix.tolist())
            if color_matrix is not None else None
        )
        cb_tuple = (
            tuple(float(v) for v in color_bias.tolist())
            if color_bias is not None else None
        )
        new_cal = Calibration(
            mm_per_px=mm_per_px,
            px_per_mm=px_per_mm,
            color_matrix=cm_tuple,
            color_bias=cb_tuple,
        )
        self.calibration_saved.emit(new_cal)

        QMessageBox.information(
            self,
            "Calibration Saved",
            f"Calibration updated:\n\n"
            f"  px/mm  = {px_per_mm:.4f}\n"
            f"  mm/px  = {mm_per_px:.6f}"
            + color_note
            + roi_note
            + file_note,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Cleanup
    # ══════════════════════════════════════════════════════════════════════

    def closeEvent(self, event):
        self._stop_worker()
        event.accept()

    def reject(self):
        self._stop_worker()
        super().reject()

    def _stop_worker(self):
        if self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait(4000)

    def _set_status(self, text: str):
        self._status_label.setText(text)


# ── Small reusable widgets ─────────────────────────────────────────────────────

class _MetricRow(QFrame):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        lbl = QLabel(label + ":")
        lbl.setFont(QFont("Consolas", 9))
        lbl.setStyleSheet(_STYLE_GREY)
        lbl.setFixedWidth(115)
        layout.addWidget(lbl)

        self._val = QLabel("—")
        self._val.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        self._val.setStyleSheet(_STYLE_GREY)
        self._val.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        layout.addWidget(self._val)

    def set_value(self, text: str, style: str = _STYLE_TEXT):
        self._val.setText(text)
        self._val.setStyleSheet(style)


class _HSep(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.setStyleSheet("color: #444;")
