"""
Initial Camera Setup Dialog — IMU-based mount alignment + focus tuning.

Run once per camera by the installer. Three jobs:

  1. IMU-based mount alignment. The accelerometer reads the gravity
     vector in the camera frame; the operator enters the chute slope
     and adjusts the mount until pitch == slope and roll == 0.
  2. IMU bias zero. With a bubble level on the camera body, capture
     the current IMU reading as the zero-point. Subtracted from all
     subsequent readings.
  3. Focus tuning. A 0–255 slider with live preview + a sharpness
     readout drives the camera's manual focus. Once the operator is
     close, the "Auto Sweep ±5" button does a fine sweep around the
     slider's value (step 1) and snaps to the peak-sharpness focus.

Save target (per-mode block of the active mode):
    surface_angle_deg
    imu_bias.pitch_deg / imu_bias.roll_deg
    camera.manual_focus
"""

from pathlib import Path
from typing import Optional

import yaml
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from pipeline.initial_setup_worker import InitialSetupWorker, IMU_AVG_WINDOW


# ── Tuning constants ──────────────────────────────────────────────────────

ALIGNMENT_OK_DEG    = 0.5
ALIGNMENT_WARN_DEG  = 1.0

FOCUS_SWEEP_RADIUS  = 5    # ±N around slider value, step 1

_VIDEO_W = 640
_VIDEO_H = 400

_STYLE_OK   = "color: #00cc00; font-weight: bold;"
_STYLE_WARN = "color: #ff8800; font-weight: bold;"
_STYLE_BAD  = "color: #cc0000; font-weight: bold;"
_STYLE_GREY = "color: #888888;"
_STYLE_TEXT = "color: #dddddd;"


class MountSetupDialog(QDialog):
    """
    Modal dialog for IMU-based camera mount alignment + focus tuning.

    Signals
    -------
    mount_saved()
        Emitted after successful Save. The active mode's per-mode
        config block has been updated in-place and the full config
        dumped to disk.
    """

    mount_saved = pyqtSignal()

    def __init__(
        self,
        cfg: dict,                    # per-mode subdict (read + write target)
        full_cfg: Optional[dict] = None,   # whole config (dumped on save)
        config_path: str | Path = "config.yaml",
        parent=None,
    ):
        super().__init__(parent)
        self.cfg = cfg
        self.full_cfg = full_cfg if full_cfg is not None else cfg
        self.config_path = Path(config_path)

        # Live IMU readings (raw, before bias subtraction)
        self._raw_pitch: Optional[float] = None
        self._raw_roll:  Optional[float] = None
        self._n_samples: int = 0

        # Bias seeded from config; updated by "Zero IMU Bias".
        bias_cfg = cfg.get("imu_bias", {}) or {}
        self._bias_pitch: float = float(bias_cfg.get("pitch_deg", 0.0))
        self._bias_roll:  float = float(bias_cfg.get("roll_deg",  0.0))

        # Focus seeded from config; the slider drives the camera live.
        cam_cfg = cfg.get("camera", {}) or {}
        self._current_focus: int = int(cam_cfg.get("manual_focus", 154))
        self._latest_sharpness: float = 0.0

        # True while a sweep is running — Save and the slider lock down.
        self._sweeping: bool = False

        label = cfg.get("label", "Camera")
        self.setWindowTitle(f"Initial Camera Setup — {label}")
        self.setMinimumSize(1080, 720)
        self.setModal(True)
        self.setStyleSheet("background-color: #2b2b2b; color: #dddddd;")

        self._build_ui()

        # Seed surface angle from config
        self.surface_angle_input.setValue(
            float(cfg.get("surface_angle_deg", 0.0))
        )

        # Start the unified worker
        self._worker = InitialSetupWorker(cfg, parent=self)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.imu_ready.connect(self._on_imu)
        self._worker.sweep_progress.connect(self._on_sweep_progress)
        self._worker.sweep_complete.connect(self._on_sweep_complete)
        self._worker.log_message.connect(self._set_status)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.worker_stopped.connect(self._on_worker_stopped)
        self._worker.start()

    # ══════════════════════════════════════════════════════════════════
    # UI
    # ══════════════════════════════════════════════════════════════════

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(12, 12, 12, 12)

        # ── Left: video feed + focus controls ───────────────────────
        left = QVBoxLayout()
        left.setSpacing(8)

        self._video_label = QLabel()
        self._video_label.setFixedSize(_VIDEO_W, _VIDEO_H)
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setStyleSheet(
            "background-color: #111111; border: 1px solid #444;"
        )
        self._video_label.setText("Connecting to camera…")
        left.addWidget(self._video_label)

        # Focus controls
        focus_group = QGroupBox("Focus")
        fg_layout = QVBoxLayout(focus_group)
        fg_layout.setSpacing(6)

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("0"))
        self.focus_slider = QSlider(Qt.Orientation.Horizontal)
        self.focus_slider.setRange(0, 255)
        self.focus_slider.setSingleStep(1)
        self.focus_slider.setPageStep(5)
        self.focus_slider.setValue(self._current_focus)
        self.focus_slider.valueChanged.connect(self._on_focus_slider)
        slider_row.addWidget(self.focus_slider, stretch=1)
        slider_row.addWidget(QLabel("255"))
        self.focus_value_label = QLabel(f"{self._current_focus}")
        self.focus_value_label.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        self.focus_value_label.setFixedWidth(40)
        self.focus_value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        slider_row.addWidget(self.focus_value_label)
        fg_layout.addLayout(slider_row)

        ctrl_row = QHBoxLayout()
        self.sweep_btn = QPushButton(f"Auto Sweep ±{FOCUS_SWEEP_RADIUS}")
        self.sweep_btn.setFixedHeight(30)
        self.sweep_btn.setToolTip(
            f"Sweep focus from slider−{FOCUS_SWEEP_RADIUS} to "
            f"slider+{FOCUS_SWEEP_RADIUS} (step 1) and pick the value "
            "with peak Laplacian-variance sharpness on the marker region."
        )
        self.sweep_btn.clicked.connect(self._on_sweep)
        self.sharpness_label = QLabel("Sharpness: —")
        self.sharpness_label.setFont(QFont("Consolas", 10))
        ctrl_row.addWidget(self.sweep_btn)
        ctrl_row.addSpacing(12)
        ctrl_row.addWidget(self.sharpness_label)
        ctrl_row.addStretch()
        fg_layout.addLayout(ctrl_row)

        self.sweep_progress = QProgressBar()
        self.sweep_progress.setFixedHeight(14)
        self.sweep_progress.setRange(0, 1)
        self.sweep_progress.setValue(0)
        self.sweep_progress.setTextVisible(False)
        self.sweep_progress.hide()
        fg_layout.addWidget(self.sweep_progress)

        left.addWidget(focus_group)

        self.status_label = QLabel("Connecting…")
        self.status_label.setStyleSheet(_STYLE_GREY)
        self.status_label.setFont(QFont("Consolas", 9))
        left.addWidget(self.status_label)

        left.addStretch()
        root.addLayout(left)

        # ── Right: surface angle + IMU + bias + save ───────────────
        right = QVBoxLayout()
        right.setSpacing(10)

        # Surface angle
        sa_group = QGroupBox("Chute slope")
        sa_layout = QHBoxLayout(sa_group)
        sa_label = QLabel("Surface angle:")
        sa_label.setFixedWidth(120)
        self.surface_angle_input = QDoubleSpinBox()
        self.surface_angle_input.setRange(-180.0, 180.0)
        self.surface_angle_input.setDecimals(2)
        self.surface_angle_input.setSingleStep(0.1)
        self.surface_angle_input.setSuffix("  °")
        self.surface_angle_input.setValue(0.0)
        self.surface_angle_input.setFixedWidth(120)
        self.surface_angle_input.valueChanged.connect(self._refresh_imu_display)
        sa_hint = QLabel(
            "Measured with a digital level on the chute itself.\n"
            "Use signed values up to ±180° for inverted mounts."
        )
        sa_hint.setStyleSheet(_STYLE_GREY)
        sa_layout.addWidget(sa_label)
        sa_layout.addWidget(self.surface_angle_input)
        sa_layout.addSpacing(12)
        sa_layout.addWidget(sa_hint)
        sa_layout.addStretch()
        right.addWidget(sa_group)

        # IMU readings — only pitch is shown / gated. Roll is captured
        # silently inside Zero IMU Bias so the pitch reading stays
        # properly bias-subtracted, but is not displayed and does not
        # block save (the user explicitly asked for pitch-only).
        imu_group = QGroupBox("IMU readings")
        imu_layout = QVBoxLayout(imu_group)
        imu_layout.setSpacing(6)
        self.pitch_row    = _AngleRow("Pitch")
        self.samples_row  = _ResultRow("Samples")
        imu_layout.addWidget(self.pitch_row)
        imu_layout.addWidget(_HSep())
        imu_layout.addWidget(self.samples_row)
        right.addWidget(imu_group)

        # Bias
        bias_group = QGroupBox("IMU bias zero-point")
        bias_layout = QVBoxLayout(bias_group)
        bias_layout.setSpacing(6)
        bias_hint = QLabel(
            "Place a bubble level on the camera body and physically hold the\n"
            "camera level. Click \"Zero IMU Bias\" — the current IMU readings\n"
            "are recorded as the zero-point for all future measurements."
        )
        bias_hint.setStyleSheet(_STYLE_GREY)
        bias_hint.setFont(QFont("Consolas", 9))
        bias_layout.addWidget(bias_hint)

        bias_btn_row = QHBoxLayout()
        self.zero_bias_btn = QPushButton("Zero IMU Bias")
        self.zero_bias_btn.setFixedHeight(32)
        self.zero_bias_btn.clicked.connect(self._on_zero_bias)
        self.bias_label = QLabel(
            f"Bias:  pitch {self._bias_pitch:+.2f}°"
        )
        self.bias_label.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        bias_btn_row.addWidget(self.zero_bias_btn)
        bias_btn_row.addSpacing(12)
        bias_btn_row.addWidget(self.bias_label)
        bias_btn_row.addStretch()
        bias_layout.addLayout(bias_btn_row)
        right.addWidget(bias_group)

        right.addStretch()

        # Save / Close
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.save_btn = QPushButton("Save Initial Camera Setup")
        self.save_btn.setFixedHeight(38)
        self.save_btn.setEnabled(False)
        self.save_btn.setToolTip(
            f"Enabled when pitch error is below ±{ALIGNMENT_OK_DEG:.1f}°."
        )
        self.save_btn.clicked.connect(self._on_save)
        close_btn = QPushButton("Close")
        close_btn.setFixedHeight(38)
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.save_btn)
        btn_row.addWidget(close_btn)
        right.addLayout(btn_row)

        root.addLayout(right)

    # ══════════════════════════════════════════════════════════════════
    # Worker callbacks
    # ══════════════════════════════════════════════════════════════════

    def _on_frame(self, qimage, sharpness: float, focus: int):
        self._latest_sharpness = sharpness
        self.sharpness_label.setText(f"Sharpness: {sharpness:.0f}")
        # Reflect any focus the worker actually applied (e.g. after a sweep)
        if focus != self._current_focus:
            self._current_focus = focus
            # Block signals so we don't ping-pong the value back to the worker
            self.focus_slider.blockSignals(True)
            self.focus_slider.setValue(focus)
            self.focus_slider.blockSignals(False)
            self.focus_value_label.setText(f"{focus}")
        pix = QPixmap.fromImage(qimage).scaled(
            _VIDEO_W, _VIDEO_H,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._video_label.setPixmap(pix)

    def _on_imu(self, raw_pitch: float, raw_roll: float, n_samples: int):
        self._raw_pitch = raw_pitch
        self._raw_roll  = raw_roll
        self._n_samples = n_samples
        self._refresh_imu_display()

    def _refresh_imu_display(self):
        if self._raw_pitch is None or self._raw_roll is None:
            return
        pitch_corr = self._raw_pitch - self._bias_pitch
        target_pitch = float(self.surface_angle_input.value())
        d_pitch = pitch_corr - target_pitch

        self.pitch_row.set(pitch_corr, target_pitch, d_pitch)

        full_window = self._n_samples >= IMU_AVG_WINDOW
        self.samples_row.set_value(
            f"{self._n_samples} / {IMU_AVG_WINDOW}",
            _STYLE_OK if full_window else _STYLE_GREY,
        )

        can_save = (
            full_window
            and abs(d_pitch) < ALIGNMENT_OK_DEG
            and not self._sweeping
        )
        self.save_btn.setEnabled(can_save)

    # ── Focus controls ────────────────────────────────────────────────

    def _on_focus_slider(self, value: int):
        if self._sweeping:
            return
        self._current_focus = int(value)
        self.focus_value_label.setText(f"{self._current_focus}")
        self._worker.set_focus(self._current_focus)

    def _on_sweep(self):
        if self._sweeping:
            return
        lo = max(0,   self._current_focus - FOCUS_SWEEP_RADIUS)
        hi = min(255, self._current_focus + FOCUS_SWEEP_RADIUS)
        self._sweeping = True
        self.focus_slider.setEnabled(False)
        self.sweep_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.sweep_progress.setRange(0, hi - lo)
        self.sweep_progress.setValue(0)
        self.sweep_progress.show()
        self._set_status(f"Sweeping focus {lo} → {hi}…")
        self._worker.request_sweep(lo, hi)

    def _on_sweep_progress(self, focus: int, sharpness: float):
        # progress bar shows index (0..N) of current step
        # — easiest is to track what we've seen so far via slider value.
        cur = self.sweep_progress.value() + 1
        self.sweep_progress.setValue(min(cur, self.sweep_progress.maximum()))

    def _on_sweep_complete(self, best_focus: int, best_sharpness: float):
        self._sweeping = False
        self.focus_slider.setEnabled(True)
        self.sweep_btn.setEnabled(True)
        self.sweep_progress.hide()
        self._current_focus = best_focus
        self.focus_slider.blockSignals(True)
        self.focus_slider.setValue(best_focus)
        self.focus_slider.blockSignals(False)
        self.focus_value_label.setText(f"{best_focus}")
        self._set_status(
            f"Sweep done. Best focus {best_focus} (sharpness {best_sharpness:.0f})."
        )
        # Save button re-enables on next IMU update via _refresh_imu_display.
        self._refresh_imu_display()

    # ── Bias ──────────────────────────────────────────────────────────

    def _on_zero_bias(self):
        if self._raw_pitch is None or self._raw_roll is None:
            QMessageBox.information(
                self, "IMU Not Ready",
                "No IMU readings yet — wait for the rolling window to fill.",
            )
            return
        if self._n_samples < IMU_AVG_WINDOW:
            reply = QMessageBox.question(
                self, "Window Not Full",
                f"Rolling window has only {self._n_samples}/"
                f"{IMU_AVG_WINDOW} samples — bias will be noisier than "
                f"if you wait. Capture anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self._bias_pitch = self._raw_pitch
        self._bias_roll  = self._raw_roll   # captured silently, not displayed
        self.bias_label.setText(
            f"Bias:  pitch {self._bias_pitch:+.2f}°"
        )
        self._refresh_imu_display()

    # ══════════════════════════════════════════════════════════════════
    # Save
    # ══════════════════════════════════════════════════════════════════

    def _on_save(self):
        sa = float(self.surface_angle_input.value())
        self.cfg["surface_angle_deg"] = round(sa, 3)
        self.cfg.setdefault("imu_bias", {})
        self.cfg["imu_bias"]["pitch_deg"] = round(self._bias_pitch, 4)
        self.cfg["imu_bias"]["roll_deg"]  = round(self._bias_roll,  4)
        self.cfg.setdefault("camera", {})
        self.cfg["camera"]["manual_focus"] = int(self._current_focus)

        try:
            with open(self.config_path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(
                    self.full_cfg, fh,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
        except Exception as exc:
            QMessageBox.critical(
                self, "Save Failed",
                f"Could not write {self.config_path}:\n{exc}",
            )
            return

        self.mount_saved.emit()
        QMessageBox.information(
            self, "Initial Camera Setup Saved",
            f"Saved for {self.cfg.get('label', 'camera')}:\n\n"
            f"  surface_angle  = {sa:.2f}°\n"
            f"  imu_bias.pitch = {self._bias_pitch:+.3f}°\n"
            f"  manual_focus   = {self._current_focus}\n\n"
            f"Saved to: {self.config_path}",
        )
        self.accept()

    # ══════════════════════════════════════════════════════════════════
    # Cleanup / errors
    # ══════════════════════════════════════════════════════════════════

    def _on_error(self, message: str):
        self._set_status(f"Error: {message}")
        QMessageBox.critical(self, "Initial Setup Error", message)

    def _on_worker_stopped(self):
        self._set_status("Camera + IMU stream stopped.")

    def _set_status(self, text: str):
        self.status_label.setText(text)

    def closeEvent(self, event):
        self._stop_worker()
        event.accept()

    def reject(self):
        self._stop_worker()
        super().reject()

    def accept(self):
        self._stop_worker()
        super().accept()

    def _stop_worker(self):
        if self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait(4000)


# ─────────────────────────────────────────────────────────────────────────
# Reusable widgets
# ─────────────────────────────────────────────────────────────────────────

class _AngleRow(QFrame):
    """Row showing an angle, its target, and the delta with status colour."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        lbl = QLabel(label + ":")
        lbl.setFont(QFont("Consolas", 10))
        lbl.setStyleSheet(_STYLE_GREY)
        lbl.setFixedWidth(80)
        layout.addWidget(lbl)

        self._value_lbl = QLabel("—")
        self._value_lbl.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        self._value_lbl.setFixedWidth(110)
        self._value_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._value_lbl)

        target_caption = QLabel("target")
        target_caption.setFont(QFont("Consolas", 9))
        target_caption.setStyleSheet(_STYLE_GREY)
        layout.addWidget(target_caption)

        self._target_lbl = QLabel("—")
        self._target_lbl.setFont(QFont("Consolas", 10))
        self._target_lbl.setFixedWidth(80)
        self._target_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._target_lbl)

        delta_caption = QLabel("Δ")
        delta_caption.setFont(QFont("Consolas", 9))
        delta_caption.setStyleSheet(_STYLE_GREY)
        layout.addWidget(delta_caption)

        self._delta_lbl = QLabel("—")
        self._delta_lbl.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        self._delta_lbl.setFixedWidth(110)
        self._delta_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._delta_lbl)

        layout.addStretch()

    def set(self, value: float, target: float, delta: float):
        abs_d = abs(delta)
        if abs_d < ALIGNMENT_OK_DEG:
            style, mark = _STYLE_OK, "✓"
        elif abs_d < ALIGNMENT_WARN_DEG:
            style, mark = _STYLE_WARN, "!"
        else:
            style, mark = _STYLE_BAD, "✗"
        self._value_lbl.setText(f"{value:+.2f}°")
        self._value_lbl.setStyleSheet(_STYLE_TEXT)
        self._target_lbl.setText(f"{target:+.2f}°")
        self._target_lbl.setStyleSheet(_STYLE_GREY)
        self._delta_lbl.setText(f"{delta:+.2f}°  {mark}")
        self._delta_lbl.setStyleSheet(style)


class _ResultRow(QFrame):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        lbl = QLabel(label + ":")
        lbl.setFont(QFont("Consolas", 10))
        lbl.setStyleSheet(_STYLE_GREY)
        lbl.setFixedWidth(180)
        layout.addWidget(lbl)
        self._val = QLabel("—")
        self._val.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        layout.addWidget(self._val)
        layout.addStretch()

    def set_value(self, text: str, style: str = _STYLE_TEXT):
        self._val.setText(text)
        self._val.setStyleSheet(style)


class _HSep(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.setStyleSheet("color: #444;")
