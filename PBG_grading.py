#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PBG_grading.py
NC State University - Peanut Breeding & Genetics
============================================================================
Web (Dash) front-end for the OAK 4 Pro D pod / seed grading pipeline.

This replaces the PyQt6 desktop GUI (gui/main_window.py) with a browser UI
while reusing the existing back-end unchanged:

    calibration.load_calibration / Calibration
    export.write_results_csv
    pipeline.oak_pipeline.OakPipelineThread   (QThread + pyqtSignal)
    dropbox_team_client.TeamDropboxClientManager

It adds, in one place:
    1. The grading console  -- sample name/weight, mode (pod/seed), live
       stats, log, session history, status bar  (mirrors main_window.py).
    2. Width-distribution visualization for the recorded data, in the NCSU
       house style (mirrors PBG_program.py).
    3. Dropbox export -- "Download to Dropbox" (upload the recording CSV to a
       chosen team-space folder).  "Append to Dropbox File" is present but
       intentionally non-functional for now (needs a summary step first).

------------------------------------------------------------------------------
Qt <-> Dash bridge
------------------------------------------------------------------------------
OakPipelineThread is a QThread that emits pyqtSignals.  A web server has no
Qt event loop, so:
  * a single QCoreApplication is created at import (headless-safe, no display),
    which is all QThread needs to run;
  * the pipeline's signals are connected with DirectConnection, so the slots
    execute in the worker thread and only push into thread-safe state;
  * a dcc.Interval polls that state once a second to update the UI.

------------------------------------------------------------------------------
Deployment
------------------------------------------------------------------------------
Run this ON the machine wired to the OAK cameras (same box that runs the
desktop GUI today) -- it needs the devices on the LAN:

    python PBG_grading.py            # then open http://127.0.0.1:8050

QR-on-bag capture (scan when you click Start) is deferred; the manual
Name / Weight inputs stand in for it.  A `pending_qr` hook is left in the
controller so wiring it later is a small change.
"""

import base64
import collections
import json
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml

import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from scipy import stats

from dotenv import load_dotenv
load_dotenv()

# ── Qt: create the application object BEFORE any QThread is constructed ──────
# QCoreApplication (not QApplication) -> no display / platform plugin needed.
from PyQt6.QtCore import Qt, QCoreApplication
_qt_app = QCoreApplication.instance() or QCoreApplication(sys.argv)

# ── Project back-end (unchanged) ─────────────────────────────────────────────
import depthai as dai
from calibration import Calibration, load_calibration
from export import write_results_csv
from pipeline.oak_pipeline import OakPipelineThread
from dropbox_team_client import TeamDropboxClientManager
from summary import (compute_summary, merge_summary_into_workbook, parse_plot,
                     parse_qr, qr_to_sample_name, session_summary_table)


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(path: Path) -> dict:
    if not path.exists():
        print(f"WARNING: {path} not found; starting with empty config.")
        return {"modes": {}}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


CFG = load_config(CONFIG_PATH)

# Dropbox team-space root.
#   DROPBOX_ROOT == ""  ->  the general top level of the NC State PB&G team
#   space (the "NC State PB&G Dropbox" root). Its contents — e.g.
#   "2026 Peanut Program" and every other team folder — appear here, so the
#   folder browser can now reach anything in the team space.
# DROPBOX_DEFAULT_DEST is just the destination pre-filled in the export box.
DROPBOX_ROOT = ""
DROPBOX_DEFAULT_DEST = "/NC State PB&G Dropbox"

# Team client is optional at boot — the grading console works without it;
# only the Dropbox export needs it.
try:
    team_manager: Optional[TeamDropboxClientManager] = TeamDropboxClientManager()
except Exception as e:  # noqa: BLE001
    print(f"Dropbox client unavailable (export disabled): {e}")
    team_manager = None


# ═════════════════════════════════════════════════════════════════════════════
# GRADING CONTROLLER  — owns the pipeline thread + thread-safe UI state
# ═════════════════════════════════════════════════════════════════════════════

# OAK PoE devices speak XLink over this TCP port. A direct (unicast) connect
# traverses a VPN, unlike the UDP broadcast that getAllAvailableDevices() uses.
OAK_XLINK_TCP_PORT = 11491


def _ip_reachable(ip: str, port: int = OAK_XLINK_TCP_PORT,
                  timeout: float = 1.0) -> bool:
    """Best-effort reachability probe for an OAK camera by IP.

    Used as a fallback when broadcast discovery can't see the device (e.g.
    when connected over VPN, where the actual run still connects fine by
    direct IP). A successful connect — or a 'connection refused', which means
    the host answered — counts as reachable; only a timeout / no-route does
    not. Mirrors the direct-IP path that oak_pipeline.py uses to connect.
    """
    if not ip or ip == "—":
        return False
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except ConnectionRefusedError:
        return True
    except OSError:
        return False


class GradingController:
    """
    Singleton bridge between the Dash UI and OakPipelineThread.

    All public state is guarded by a lock and read via snapshot();
    pipeline signal slots run in the worker thread (DirectConnection)
    and only mutate state under that lock.
    """

    def __init__(self, cfg: dict, config_path: Path):
        self.cfg = cfg
        self.config_path = Path(config_path)
        self._lock = threading.Lock()

        self.mode: str = self._initial_mode()
        self.calibration: Calibration = load_calibration(self._mode_cfg())

        self.thread: Optional[OakPipelineThread] = None
        self.running: bool = False
        self.run_start: Optional[float] = None

        self.logs: collections.deque = collections.deque(maxlen=600)
        self.latest = {"count": 0, "active": 0, "fps": 0.0}
        self.session: List[dict] = []
        self.last_csv_path: Optional[Path] = None
        self.last_error: Optional[str] = None

        # Session persistence: every completed run rewrites this JSON under
        # output/Sessions/, so an accidentally closed app loses nothing and a
        # previous session can be reloaded later. Loading an old session
        # repoints session_path so subsequent runs continue in that file.
        self.sessions_dir = Path(cfg.get("output_dir", "./output")) / "Sessions"
        self.session_path = (self.sessions_dir /
                             f"session_{time.strftime('%Y%m%d_%H%M%S')}.json")

        # Set when a run is started, so _on_pipeline_stopped can label output.
        self._run_plot_name = ""
        self._run_plot_weight = ""
        self._run_mode = self.mode
        self._run_plot = ""

        # Plot registered for the NEXT run (QR scan or Manual Entry's "Set
        # Sample Information"). Consumed once when a run starts; kept separate
        # from the sample name, which is purely the operator's file/folder
        # designation.
        self.pending_plot: Optional[str] = None

        # Calibration: launched as a separate process (calibrate.py) so the
        # PyQt dialog gets its own QApplication / main thread.
        self.calibrating: Optional[str] = None      # "initial" | "target" | None
        self._cal_proc = None

        # Reserved for the future QR-on-bag scan.
        self.pending_qr: Optional[str] = None

    # ── config helpers (mirror main_window.py) ──────────────────────────────
    def _initial_mode(self) -> str:
        m = self.cfg.get("mode")
        modes = self.cfg.get("modes", {})
        if isinstance(m, str) and m in modes:
            return m
        return next(iter(modes), "pod")

    def _mode_cfg(self, mode: Optional[str] = None) -> dict:
        m = mode or self.mode
        modes = self.cfg.get("modes", {})
        if m not in modes:
            modes.setdefault(m, {})
        return modes[m]

    def mode_label(self, mode: Optional[str] = None) -> str:
        mc = self._mode_cfg(mode)
        m = mode or self.mode
        return mc.get("label", m.title())

    def set_mode(self, mode: str) -> None:
        with self._lock:
            if self.running or self.calibrating or mode == self.mode:
                return
            self.mode = mode
            self.calibration = load_calibration(self._mode_cfg())

    # ── calibration (spawns calibrate.py in its own process) ─────────────────
    def launch_calibration(self, step: str, mode: str) -> Optional[str]:
        """Open a calibration dialog (initial|target) for *mode*.

        Returns None on success or an error string. The dialog runs in a
        separate process; when it closes, config.yaml is reloaded.
        """
        import subprocess
        with self._lock:
            if self.running:
                return "Stop the camera before calibrating."
            if self.calibrating:
                return "A calibration window is already open."
            mode = mode or self.mode
            if mode not in self.cfg.get("modes", {}):
                return f"Unknown mode '{mode}'."
            self.mode = mode

        script = Path(__file__).resolve().with_name("calibrate.py")
        if not script.exists():
            return f"Launcher not found next to the app: {script}"

        try:
            proc = subprocess.Popen(
                [sys.executable, str(script),
                 "--step", step, "--mode", mode,
                 "--config", str(self.config_path)],
                cwd=str(Path(__file__).resolve().parent),
            )
        except Exception as e:  # noqa: BLE001
            return f"Could not launch calibration: {e}"

        with self._lock:
            self.calibrating = step
            self._cal_proc = proc
        self._log(f"\nOpened {step} calibration for {self.mode_label(mode)} "
                  f"(mode={mode}). Complete it in the calibration window, then "
                  f"close it — config.yaml will reload automatically.")
        threading.Thread(target=self._watch_calibration, args=(proc,),
                         daemon=True).start()
        return None

    def _watch_calibration(self, proc) -> None:
        try:
            proc.wait()
        except Exception:  # noqa: BLE001
            pass
        self.reload_config()
        with self._lock:
            self.calibrating = None
            self._cal_proc = None
        self._log("\nCalibration window closed — reloaded config.yaml.")

    def reload_config(self) -> None:
        """Re-read config.yaml from disk (after a calibration save)."""
        try:
            new_cfg = load_config(self.config_path)
        except Exception:  # noqa: BLE001
            return
        with self._lock:
            self.cfg = new_cfg
            if self.mode not in new_cfg.get("modes", {}):
                self.mode = self._initial_mode()
            self.calibration = load_calibration(self._mode_cfg())

    def source_status(self) -> tuple[str, str]:
        """Return (text, css_color) describing the active mode's camera.

        Broadcast discovery (getAllAvailableDevices) only works on the same
        subnet; a VPN doesn't forward the broadcast, so off-campus the camera
        won't be "discovered" even though a run connects fine by direct IP.
        We therefore fall back to a direct TCP reachability probe.
        """
        mc = self._mode_cfg()
        ip = mc.get("device_ip", "—")
        label = self.mode_label()
        if ip == "—":
            return f"{label} camera  (no IP configured)", "#aa6600"
        # 1) LAN broadcast discovery (same-subnet only).
        try:
            infos = dai.Device.getAllAvailableDevices()
            ips = {str(getattr(i, "name", "") or getattr(i, "mxid", "")) for i in infos}
        except Exception:  # noqa: BLE001
            ips = set()
        if ip in ips:
            return f"{label} camera  {ip}  ✓ detected", "#00aa00"
        # 2) Direct reachability (works over VPN; mirrors the real connect).
        if _ip_reachable(ip):
            return f"{label} camera  {ip}  ✓ reachable (direct IP)", "#00aa00"
        return f"{label} camera  {ip}  ✗ not found", "#aa0000"

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self, mode: str, plot_name: str, plot_weight: str,
              save_frames: bool, save_dir: Optional[str]) -> Optional[str]:
        """
        Start a grading run. Returns None on success or an error string.
        """
        with self._lock:
            if self.running:
                return "A run is already in progress."

            self.mode = mode
            mc = self._mode_cfg(mode)
            label = self.mode_label(mode)
            self.calibration = load_calibration(mc)

            plot_name = (plot_name or "").strip()
            if not plot_name:
                return "Please enter a sample name before starting."

            # Resolve model path (relative paths are relative to config dir).
            model_path = Path(mc.get("model_path", ""))
            if not model_path.is_absolute():
                model_path = (self.config_path.parent / model_path).resolve()
            if not model_path.exists():
                return (f"Model file for {label} mode not found:\n{model_path}\n"
                        f"Update modes.{mode}.model_path in config.yaml.")

            # When saving frames, nest them per sample AND per mode —
            # <parent>/<Sample Name>/<Pod|Seed> Images — so grading the same
            # sample with both cameras can't overwrite images. If no parent
            # folder is typed in, fall back to the configured output_dir, so
            # the sample name only needs to be entered once. (plot_name is
            # already validated as non-empty above.) The CSV is written to
            # output_dir in _on_stopped regardless.
            effective_save_dir: Optional[str] = None
            if save_frames:
                safe_sample = "".join(
                    c if c.isalnum() or c in "_-" else "_" for c in plot_name
                )
                parent = (save_dir or "").strip() \
                    or self.cfg.get("output_dir", "./output")
                img_sub = "Pod Images" if mode == "pod" else "Seed Images"
                effective_save_dir = str(Path(parent) / safe_sample / img_sub)

            # Build the cfg the pipeline expects (per-mode block + a couple of
            # globals merged in), exactly like main_window._on_start.
            pipeline_cfg = {
                "output_dir":        self.cfg.get("output_dir", "./output"),
                "report_interval_s": self.cfg.get("report_interval_s", 2.0),
                **mc,
		"model_path": str(model_path),
            }

            # reset live state
            self.logs.clear()
            self.latest = {"count": 0, "active": 0, "fps": 0.0}
            self.last_error = None
            self._run_plot_name = plot_name
            self._run_plot_weight = (plot_weight or "").strip()
            self._run_mode = mode
            # Plot for the Dropbox append: an explicitly registered plot wins
            # (Manual Entry / QR scan); otherwise fall back to a plot-suffixed
            # sample name. One-shot — cleared so it can't leak into later runs.
            self._run_plot = self.pending_plot or parse_plot(plot_name) or ""
            self.pending_plot = None
            self._log(f"Mode: {label}  |  Sample: {plot_name}  |  "
                      f"Weight: {self._run_plot_weight or '(none)'}")

            self.thread = OakPipelineThread(
                pipeline_cfg, save_frames=save_frames,
                save_dir=effective_save_dir,
            )
            dc = Qt.ConnectionType.DirectConnection
            self.thread.log_message.connect(self._on_log, dc)
            self.thread.status_update.connect(self._on_status, dc)
            self.thread.pipeline_stopped.connect(self._on_stopped, dc)
            self.thread.error_occurred.connect(self._on_error, dc)

            self.running = True
            self.run_start = time.time()
            self.thread.start()
            return None

    def stop(self) -> None:
        with self._lock:
            t = self.thread
        if t is not None and t.isRunning():
            self._log("\nStopping pipeline...")
            t.request_stop()

    # ── pipeline signal slots (run in the worker thread) ─────────────────────
    def _on_log(self, text: str) -> None:
        with self._lock:
            self.logs.append(text)

    def _on_status(self, total: int, active: int, fps: float) -> None:
        with self._lock:
            self.latest = {"count": int(total), "active": int(active),
                           "fps": float(fps)}

    def _on_stopped(self) -> None:
        # Save results + append a session-history row, then mark stopped.
        with self._lock:
            t = self.thread
            elapsed = self._format_elapsed()
            label = self.mode_label(self._run_mode)
            name = self._run_plot_name or "unnamed"
            weight = self._run_plot_weight

        if t is None:
            with self._lock:
                self.running = False
            return

        tracker = t.tracker
        n_tracks = tracker.total_unique
        csv_name = "(none)"

        if n_tracks > 0:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_dir = Path(self.cfg.get("output_dir", "./output"))
            out_dir.mkdir(parents=True, exist_ok=True)
            safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
            csv_name = f"{self._run_mode}_{safe}_{ts}.csv"
            out_path = out_dir / csv_name
            n_rows = write_results_csv(
                track_dims=tracker.max_dims, output_path=out_path,
                calibration=self.calibration, plot_name=name, plot_weight=weight,
            )
            with self._lock:
                self.last_csv_path = out_path
            self._log(f"\nSaved {n_rows} tracked {label.lower()}s to:\n  {out_path}")
        else:
            self._log(f"No {label.lower()}s tracked — nothing to save.")

        with self._lock:
            self.session.append({
                "n": len(self.session) + 1,
                "mode": label,
                "plot_name": name,
                "plot": self._run_plot,
                "weight": f"{weight}g" if weight else "",
                "count": n_tracks,
                "elapsed": elapsed,
                "csv_file": csv_name,
            })
            self.running = False
            self.run_start = None
        self._persist_session()

    # ── session persistence ──────────────────────────────────────────────────
    def _persist_session(self) -> None:
        """Rewrite the session file after every run (crash/close safeguard)."""
        with self._lock:
            runs = list(self.session)
            path = self.session_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "runs": runs}
            path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        except Exception as e:  # noqa: BLE001 — never let IO kill the run flow
            self._log(f"WARNING: could not save session file: {e}")

    def list_sessions(self) -> List[dict]:
        """Saved sessions, newest first: {path, label, n_runs}."""
        out = []
        for p in sorted(self.sessions_dir.glob("session_*.json"), reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                runs = data.get("runs", [])
                ts = time.strftime("%b %d %Y, %H:%M",
                                   time.strptime(p.stem, "session_%Y%m%d_%H%M%S"))
                out.append({"path": str(p), "n_runs": len(runs),
                            "label": f"{ts} — {len(runs)} run(s)"})
            except Exception:  # noqa: BLE001 — skip unreadable files
                continue
        return out

    def load_session(self, path: str) -> tuple[int, Optional[str]]:
        """Replace the in-memory session with a saved one and resume it.

        Returns (n_runs, error). Safe even mid-testing: the current session is
        already on disk (every run persists), so nothing is lost by switching.
        """
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            runs = data.get("runs", [])
            if not isinstance(runs, list):
                return 0, "Not a session file."
        except Exception as e:  # noqa: BLE001
            return 0, f"Could not read session: {e}"
        with self._lock:
            if self.running:
                return 0, "Stop the camera before loading a session."
            self.session = runs
            self.session_path = Path(path)   # new runs continue in this file
        return len(runs), None

    def _on_error(self, message: str) -> None:
        self._log(f"\nERROR: {message}")
        with self._lock:
            self.last_error = message
            self.running = False
            self.run_start = None

    # ── internals ─────────────────────────────────────────────────────────────
    def _log(self, text: str) -> None:
        # Caller may or may not hold the lock; re-entrant-safe via a flag.
        if self._lock.acquire(blocking=False):
            try:
                self.logs.append(text)
            finally:
                self._lock.release()
        else:
            self.logs.append(text)

    def _format_elapsed(self) -> str:
        if self.run_start is None:
            return "00:00"
        secs = int(time.time() - self.run_start)
        m, s = divmod(secs, 60)
        return f"{m:02d}:{s:02d}"

    # ── snapshot for the UI poller ────────────────────────────────────────────
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "calibrating": self.calibrating,
                "mode": self.mode,
                "label": self.mode_label(),
                "count": self.latest["count"],
                "active": self.latest["active"],
                "fps": self.latest["fps"],
                "elapsed": self._format_elapsed(),
                "log": "\n".join(self.logs),
                "session": list(self.session),
                "last_csv": str(self.last_csv_path) if self.last_csv_path else "",
                "last_sample": self._run_plot_name,
                "last_mode": self._run_mode,
                "last_weight": self._run_plot_weight,
                "error": self.last_error,
            }


CONTROLLER = GradingController(CFG, CONFIG_PATH)


# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATION  (NCSU house style — mirrors PBG_program.py)
# ═════════════════════════════════════════════════════════════════════════════

METRIC_LABELS = {
    "width_mm":  "Width (mm)",
    "length_mm": "Length (mm)",
    "area_mm2":  "Area (mm²)",
    "hunter_l":  "Hunter L",
}

# NCSU grade reference overlays for the WIDTH metric, mirroring the pod/seed
# width distributions in PBG_program.py. Each region is
# (lo, hi, fill_color, opacity); None means "extend to the edge of the data".
# Regions are drawn behind the data and clipped to the visible x-range.
GRADE_REFERENCE = {
    "pod": {
        "regions": [
            (14.6844, None,    "Red",   0.10),   # Jumbo
            (12.7,    14.6844, "White", 0.40),    # Fancy
            (None,    12.7,    "Blue",  0.10),    # No. 1
        ],
        "lines": [14.6844, 12.7],
        "annotations": [(16.5, "Jumbo"), (13.7, "Fancy"), (10.5, "No. 1")],
    },
    "seed": {
        "regions": [
            (9.523, None,  "LightGreen",  0.15),  # SELK
            (8.53,  9.523, "Gold",        0.15),  # ELK
            (7.14,  8.53,  "LightSalmon", 0.15),  # Medium
            (5.93,  7.14,  "OrangeRed",   0.15),  # No. 1
            (None,  5.93,  "Red",         0.15),  # Other
        ],
        "lines": [9.523, 8.53, 7.14, 5.93],
        "annotations": [(10.8, "SELK"), (9.0, "ELK"), (7.8, "Med"),
                        (6.5, "No.1"), (4.5, "Other")],
    },
}


def build_distribution_figure(df: pd.DataFrame, metric: str,
                              mode_label: str) -> go.Figure:
    """Histogram + KDE of a measurement column, NCSU style.

    For the width metric, pod and seed recordings get the NCSU grade
    reference bands (shaded background + threshold lines + grade labels),
    matching the pod/seed width distributions in PBG_program.py. Other
    metrics get plain mean/median reference lines.
    """
    fig = go.Figure()
    if df is None or df.empty or metric not in df.columns:
        fig.update_layout(template="plotly_white",
                           title="No data loaded", height=520)
        return fig

    vals = pd.to_numeric(df[metric], errors="coerce").dropna().values
    if len(vals) == 0:
        fig.update_layout(
            template="plotly_white", height=520,
            title=f"No numeric values for {METRIC_LABELS.get(metric, metric)} "
                  f"(column may be empty)",
        )
        return fig

    x_label = METRIC_LABELS.get(metric, metric)

    # Grade bands only apply to width, and only for pod / seed recordings.
    # Match by prefix: labels can carry the sample name too ("Pod — <sample>"
    # for a selected session row), which must not lose the bands.
    _ml = (mode_label or "").lower()
    grade = (next((v for k, v in GRADE_REFERENCE.items() if _ml.startswith(k)),
                  None)
             if metric == "width_mm" else None)

    # Visible x-range (with a little padding); bands are clipped to this so
    # they never extend into empty space.
    xmin, xmax = float(vals.min()), float(vals.max())
    pad = max(0.5, (xmax - xmin) * 0.04)
    xlo, xhi = xmin - pad, xmax + pad

    # Shaded grade regions (drawn first, behind the data).
    if grade:
        for lo, hi, color, opacity in grade["regions"]:
            x0 = xlo if lo is None else lo
            x1 = xhi if hi is None else hi
            x0, x1 = max(x0, xlo), min(x1, xhi)
            if x0 < x1:
                fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=opacity,
                              layer="below", line_width=0)

    # Histogram (density-normalized so the KDE overlays cleanly)
    nbins = min(40, max(10, int(np.sqrt(len(vals)) * 2)))
    fig.add_trace(go.Histogram(
        x=vals, histnorm="probability density", name="Measured",
        marker_color="rgba(150,150,150,0.45)",
        marker_line=dict(color="rgba(90,90,90,0.9)", width=0.5),
        nbinsx=nbins,
        hovertemplate=f"{x_label}: %{{x:.2f}}<br>density: %{{y:.3f}}<extra></extra>",
    ))

    # Track the tallest feature so we can leave headroom above it (keeps the
    # bars clear of the legend at the top of the plot).
    hist_density, _ = np.histogram(vals, bins=nbins, density=True)
    y_peak = float(hist_density.max()) if len(hist_density) else 0.0

    # KDE overlay (needs >=2 distinct points)
    if len(np.unique(vals)) >= 2:
        kde = stats.gaussian_kde(vals)
        xs = np.linspace(vals.min(), vals.max(), 400)
        kde_y = kde(xs)
        y_peak = max(y_peak, float(kde_y.max()))
        # Percentage of measured values at/below and above each x, shown on
        # hover (mirrors the NCSU pod/seed width plots in PBG_program.py).
        sorted_vals = np.sort(vals)
        pct_below = np.searchsorted(sorted_vals, xs, side="right") / len(vals) * 100.0
        pct_above = 100.0 - pct_below
        fig.add_trace(go.Scatter(
            x=xs, y=kde_y, mode="lines", name="KDE",
            line=dict(color="black", width=2.5),
            customdata=np.column_stack([pct_below, pct_above]),
            hovertemplate=(f"{x_label}: %{{x:.2f}}<br>"
                           f"Below: %{{customdata[0]:.1f}}%<br>"
                           f"Above: %{{customdata[1]:.1f}}%<extra></extra>"),
        ))

    if grade:
        # NCSU grade threshold lines + labels.
        for v in grade["lines"]:
            if xlo <= v <= xhi:
                fig.add_vline(x=v, line_width=0.8, line_color="black")
        for x, txt in grade["annotations"]:
            if xlo <= x <= xhi:
                fig.add_annotation(x=x, yref="paper", y=0.97, yanchor="top",
                                   text=f"<b>{txt}</b>", showarrow=False,
                                   font=dict(size=10, color="#333"))
    else:
        # Generic mean / median reference lines.
        mean_v, med_v = float(np.mean(vals)), float(np.median(vals))
        for v, dash_, txt in ((mean_v, "solid", "mean"), (med_v, "dot", "median")):
            fig.add_vline(x=v, line_width=1, line_dash=dash_, line_color="#222",
                          annotation_text=f"{txt} {v:.2f}",
                          annotation_position="top",
                          annotation_font_size=9)

    fig.update_layout(
        title=dict(text=f"<b>{mode_label} {x_label} Distribution  (n = {len(vals)})</b>",
                   x=0.5, font=dict(size=12)),
        xaxis_title=f"<b>{x_label}</b>", yaxis_title="<b>Density</b>",
        template="plotly_white", font=dict(family="Nunito Sans", size=10),
        legend=dict(font=dict(size=9), orientation="h",
                    yanchor="bottom", y=1.04, xanchor="left", x=0),
        margin=dict(t=80, b=45, l=55, r=20),
        plot_bgcolor="white", height=520, bargap=0.02,
        xaxis_range=[xlo, xhi],
    )
    fig.update_xaxes(mirror=True, ticks="outside", showline=True,
                     linecolor="black", gridcolor="whitesmoke")
    # Leave ~20% headroom above the tallest bar/curve so nothing reaches the
    # legend. Bump the 1.20 factor up for more space, down for less.
    y_range = [0, y_peak * 1.20] if y_peak > 0 else None
    fig.update_yaxes(range=y_range, mirror=True, ticks="outside", showline=True,
                     linecolor="black", gridcolor="whitesmoke")
    return fig


def summary_cards(df: pd.DataFrame, metric: str) -> list:
    if df is None or df.empty or metric not in df.columns:
        return []
    vals = pd.to_numeric(df[metric], errors="coerce").dropna()
    if vals.empty:
        return [dbc.Col(html.Div("No numeric data for this metric.",
                                 className="text-muted small"), width=12)]
    n_fallback = int((df.get("width_quality") == "fallback").sum()) \
        if "width_quality" in df.columns else 0
    stat = [
        ("Count", f"{len(vals)}"),
        ("Mean", f"{vals.mean():.2f}"),
        ("Median", f"{vals.median():.2f}"),
        ("Std Dev", f"{vals.std():.2f}"),
        ("Min / Max", f"{vals.min():.2f} / {vals.max():.2f}"),
        ("Fallback Widths", f"{n_fallback}"),
    ]
    return [
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div(label, className="text-muted small"),
            html.Div(value, style={"fontSize": "18px", "fontWeight": "bold",
                                   "color": "#000"}),
        ], style={"padding": "10px"}), style=CARD_STYLE), width=6, md=2)
        for label, value in stat
    ]


def _mode_label_from_name(name: str) -> str:
    n = (name or "").lower()
    if n.startswith("seed"):
        return "Seed"
    if n.startswith("pod"):
        return "Pod"
    return "Sample"


# ═════════════════════════════════════════════════════════════════════════════
# DASH APP + STYLES  (LUX + NCSU red, matching the other PBG apps)
# ═════════════════════════════════════════════════════════════════════════════

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUX, dbc.icons.BOOTSTRAP],
    meta_tags=[{"name": "viewport",
                "content": "width=device-width, initial-scale=1.0"}],
    suppress_callback_exceptions=True,
    title="PBG Grading",
)
server = app.server

HEADER_STYLE = {"backgroundColor": "#CC0000", "color": "white",
                "fontWeight": "bold", "padding": "8px 12px", "fontSize": "13px"}
CARD_STYLE = {"border": "1px solid #ddd", "borderRadius": "6px",
              "marginBottom": "16px"}
# One color scheme for the whole app: every button charcoal, except
# Start Camera (green) and Stop Camera (red).
BTN_DARK = {"backgroundColor": "#3b4045", "border": "none", "color": "white",
            "fontWeight": "bold"}
BTN_GREEN = {"backgroundColor": "#2f9e44", "border": "none", "color": "white",
             "fontWeight": "bold"}
BTN_RED = {"backgroundColor": "#CC0000", "border": "none", "color": "white",
           "fontWeight": "bold"}
LABEL_STYLE = {"fontWeight": "bold", "fontSize": "12px", "marginBottom": "4px",
               "color": "#333"}

# Logo (optional, same convention as PBG_QRscanner.py)
try:
    _img = base64.b64encode(open("PBG_logo.png", "rb").read()).decode()
    logo_element = html.Img(src=f"data:image/png;base64,{_img}",
                            style={"maxHeight": "90px", "maxWidth": "70%"})
except Exception:  # noqa: BLE001
    logo_element = html.Div()


def _card(title, body, header_style=HEADER_STYLE):
    return dbc.Card([dbc.CardHeader(title, style=header_style),
                     dbc.CardBody(body, style={"padding": "12px"})],
                    style=CARD_STYLE)


# ── Layout ───────────────────────────────────────────────────────────────────
app.layout = dbc.Container([

    dcc.Interval(id="poll", interval=1000, n_intervals=0),
    dcc.Store(id="viz-store", data=None),
    dcc.Store(id="fb-path", data=DROPBOX_ROOT, storage_type="session"),
    dcc.Store(id="fb-selected-folder", data=DROPBOX_DEFAULT_DEST),
    dcc.Store(id="afb-path", data=DROPBOX_ROOT, storage_type="session"),
    dcc.Store(id="afb-selected", data="", storage_type="session"),

    # Header
    dbc.Row(dbc.Col(html.Div([html.Center([
        logo_element, html.P(),
        html.H4("Pod / Seed Grading", className="mb-1"),
        html.P("North Carolina State University — Peanut Breeding & Genetics",
               className="text-muted small"),
    ])]), width=12), className="mb-3 mt-3"),

    dbc.Row([
        # ── LEFT: grading console ─────────────────────────────────────────────
        dbc.Col([
            _card("Source", [
                dbc.Row([
                    dbc.Col(html.Span("Mode:", style=LABEL_STYLE), width="auto",
                            className="pt-1"),
                    dbc.Col(dbc.RadioItems(
                        id="mode-radio", inline=True,
                        options=[{"label": "Pod", "value": "pod"},
                                 {"label": "Seed", "value": "seed"}],
                        value=CONTROLLER.mode,
                    ), width="auto"),
                    dbc.Col(dbc.Button("Refresh", id="refresh-btn", size="sm",
                                       style=BTN_DARK),
                            width="auto", className="ms-auto"),
                ], className="g-2 align-items-center"),
                html.Div(id="source-status", className="small mt-2",
                         style={"fontFamily": "monospace"}),
            ]),

            _card("Sample Information", [
                # QR Dash-style entry tabs: scan a tag, or type the plot data.
                # Both routes just populate the Sample Name field below.
                dbc.Tabs([
                    dbc.Tab([
                        dbc.Row([
                            dbc.Col(html.Span("QR Scan:", style=LABEL_STYLE),
                                    width="auto", className="pt-1"),
                            dbc.Col(dbc.Input(
                                id="qr-scan", size="sm", autofocus=True,
                                debounce=True, autocomplete="off",
                                placeholder="Click here, then scan tag…")),
                        ], className="g-2 mb-1 mt-2 align-items-center"),
                        html.Div(id="qr-scan-result", className="small mb-2",
                                 style={"minHeight": "1em"}),
                    ], label="QR Scan", tab_id="tab-qr"),
                    dbc.Tab([
                        dbc.Row([
                            dbc.Col([html.Div("Year", style=LABEL_STYLE),
                                     dbc.Input(id="man-year", size="sm",
                                               placeholder="2026")], md=6),
                            dbc.Col([html.Div("Location", style=LABEL_STYLE),
                                     dbc.Input(id="man-loc", size="sm",
                                               placeholder="LEW")], md=6),
                        ], className="g-2 mb-2 mt-2"),
                        dbc.Row([
                            dbc.Col([html.Div("NC Accession", style=LABEL_STYLE),
                                     dbc.Input(id="man-acc", size="sm",
                                               placeholder="N19003")], md=6),
                            dbc.Col([html.Div("Name", style=LABEL_STYLE),
                                     dbc.Input(id="man-name", size="sm",
                                               placeholder="Bailey")], md=6),
                        ], className="g-2 mb-2"),
                        dbc.Row([
                            dbc.Col([html.Div("Plot (required)",
                                              style=LABEL_STYLE),
                                     dbc.Input(id="man-plot", size="sm",
                                               placeholder="101")], md=6),
                        ], className="g-2 mb-4"),
                    ], label="Manual Entry", tab_id="tab-manual"),
                ], id="sample-entry-tabs", active_tab="tab-qr",
                   className="mb-2"),
                dbc.Row([
                    dbc.Col(html.Span(["Sample Name",
                                       html.Span(" *", style={"color": "#CC0000"}),
                                       ":"], style=LABEL_STYLE,
                                      title="Required — a run cannot start "
                                            "without a sample name."),
                            width="auto", className="pt-1"),
                    dbc.Col(dbc.Input(id="plot-name", placeholder="e.g. Sample 1",
                                      size="sm")),
                ], className="g-2 mb-2 align-items-center"),
                dbc.Row([
                    dbc.Col(html.Span("Sample Weight:", style=LABEL_STYLE),
                            width="auto", className="pt-1"),
                    dbc.Col(dbc.InputGroup([
                        dbc.Input(id="plot-weight", placeholder="e.g. 100",
                                  type="number", size="sm"),
                        dbc.InputGroupText("g"),
                    ], size="sm")),
                ], className="g-2 align-items-center"),
                dbc.Button("Set Sample Information", id="btn-manual-set",
                           style=BTN_DARK, size="sm",
                           className="w-100 mt-3 mb-1"),
                html.Div(id="manual-result", className="small",
                         style={"minHeight": "1em"}),
            ]),

            _card("Frame Saving", [
                dbc.Checkbox(id="save-frames", label="Save Frames to Disk",
                             value=False, className="mb-2"),
                dbc.Input(id="save-path", size="sm", disabled=True,
                          placeholder="Optional — Default: Output/[Sample Name]/"
                                      "[Pod or Seed] Images"),
            ]),

            _card("Camera Controls", [
                dbc.Row([
                    dbc.Col(dbc.Button([html.I(className="bi bi-aspect-ratio me-1"),
                                        "Initial Camera Setup"],
                                       id="btn-initial", style=BTN_DARK,
                                       size="sm",
                                       className="w-100"), md=6,
                            className="mb-2"),
                    dbc.Col(dbc.Button([html.I(className="bi bi-bullseye me-1"),
                                        "Target Calibration"],
                                       id="btn-target", style=BTN_DARK,
                                       size="sm",
                                       className="w-100"), md=6,
                            className="mb-2"),
                ], className="g-2"),
                dbc.Row([
                    dbc.Col(dbc.Button([html.I(className="bi bi-play-fill me-1"),
                                        "Start Camera"], id="btn-start",
                                       style=BTN_GREEN, className="w-100"), md=4,
                            className="mb-2"),
                    dbc.Col(dbc.Button([html.I(className="bi bi-stop-fill me-1"),
                                        "Stop Camera"], id="btn-stop",
                                       style=BTN_RED, className="w-100",
                                       disabled=True), md=4, className="mb-2"),
                    dbc.Col(dbc.Button([html.I(className="bi bi-arrow-repeat me-1"),
                                        "Next Sample"], id="btn-next",
                                       style=BTN_DARK,
                                       className="w-100"), md=4, className="mb-2"),
                ], className="g-2"),
                html.Div(id="action-alert", className="mt-1"),
            ]),

            # Live stats
            _card("Live Stats", dbc.Row([
                dbc.Col(html.Div([
                    html.Div("0", id="live-count",
                             style={"fontSize": "44px", "fontWeight": "bold",
                                    "textAlign": "center", "fontFamily": "monospace"}),
                    html.Div("Unique Pods", id="live-caption",
                             className="text-muted small text-center"),
                ]), width=4),
                dbc.Col(html.Div([
                    html.Div("00:00", id="live-elapsed",
                             style={"fontSize": "32px", "textAlign": "center",
                                    "fontFamily": "monospace"}),
                    html.Div("Elapsed", className="text-muted small text-center"),
                ]), width=4),
                dbc.Col(html.Div([
                    html.Div("-- fps", id="live-fps",
                             style={"fontSize": "26px", "textAlign": "center",
                                    "fontFamily": "monospace"}),
                    html.Div("Active: --", id="live-active",
                             className="text-muted small text-center"),
                ]), width=4),
            ], className="align-items-center")),

            # Log
            _card("Log", html.Pre(id="log-box", style={
                "height": "180px", "overflowY": "auto", "fontSize": "11px",
                "fontFamily": "monospace", "backgroundColor": "#fafafa",
                "border": "1px solid #eee", "padding": "8px", "margin": 0,
                "whiteSpace": "pre-wrap"})),
        ], lg=5),

        # ── RIGHT: history + visualization + export ───────────────────────────
        dbc.Col([
            _card("Session History", [
                dash_table.DataTable(
                    id="session-table",
                    columns=[{"name": c, "id": i} for c, i in [
                        ("#", "n"), ("Mode", "mode"), ("Sample", "plot_name"),
                        ("Weight", "weight"), ("Count", "count"),
                        ("Duration", "elapsed"), ("CSV File", "csv_file")]],
                    data=[], row_selectable="single", selected_rows=[],
                    page_size=10,
                    style_table={"overflowX": "auto"},
                    style_cell={"fontSize": "11px", "padding": "5px",
                                "textAlign": "center"},
                    style_header={"backgroundColor": "#CC0000", "color": "white",
                                  "fontWeight": "bold"},
                    style_data_conditional=[{"if": {"row_index": "odd"},
                                             "backgroundColor": "#f9f9f9"}],
                ),
                html.Div("Select a row to view that run and use it for "
                         "downloads; deselect for the latest run.",
                         className="text-muted", style={"fontSize": "11px"},
                         id="session-hint"),
                html.Hr(className="my-2"),
                dbc.Row([
                    dbc.Col(dcc.Dropdown(id="session-select", options=[],
                                         placeholder="Previous sessions…",
                                         style={"fontSize": "12px"}), md=7),
                    dbc.Col(dbc.Button("Load Session", id="btn-load-session",
                                       style=BTN_DARK, size="sm",
                                       className="w-100"), md=3),
                    dbc.Col(dbc.Button(html.I(className="bi bi-arrow-clockwise"),
                                       id="btn-refresh-sessions", size="sm",
                                       style=BTN_DARK,
                                       className="w-100",
                                       title="Refresh session list"), md=2),
                ], className="g-2 align-items-center"),
                html.Div(id="session-load-result", className="small mt-1"),
            ]),

            _card("Width Distribution", [
                dbc.Row([
                    dbc.Col([
                        html.Div("Data source", style=LABEL_STYLE),
                        dbc.RadioItems(
                            id="viz-source", inline=True,
                            options=[{"label": "Latest Record", "value": "latest"},
                                     {"label": "Selected Run", "value": "selected"}],
                            value="latest"),
                    ], md=6),
                    dbc.Col([
                        html.Div("Metric", style=LABEL_STYLE),
                        dcc.Dropdown(
                            id="viz-metric",
                            options=[{"label": v, "value": k}
                                     for k, v in METRIC_LABELS.items()],
                            value="width_mm", clearable=False,
                            style={"fontSize": "12px"}),
                    ], md=6),
                ], className="g-2 mb-2 align-items-end"),
                dcc.Graph(id="viz-graph", config={"displaylogo": False}),
                dbc.Row(id="viz-stats", className="g-2 mt-1"),
            ]),

            _card("Download CSV", [
                dbc.Row([
                    dbc.Col([
                        html.Div("File Name", style=LABEL_STYLE),
                        dbc.Input(id="local-filename", size="sm",
                                  placeholder="e.g. Sample 1 - Default: [Sample Name]"),
                    ], md=7),
                    dbc.Col(
                        dbc.Button([html.I(className="bi bi-download me-1"),
                                    "Download to Computer"],
                                   id="btn-local-download",
                                   style=BTN_DARK, className="w-100",
                                   size="sm"),
                        md=5, className="d-flex align-items-end"),
                ], className="g-2 mb-1 align-items-end"),
                dbc.Button([html.I(className="bi bi-table me-1"),
                            "Download Session Summary"],
                           id="btn-summary-download", style=BTN_DARK,
                           className="w-100 mt-2", size="sm",
                           title="One summarized row per plot (same numbers "
                                 "the trial-file append writes) for every run "
                                 "in the session — current or loaded."),
                dcc.Download(id="local-download"),
                html.Div(id="local-result", className="mt-2"),
            ]),

            _card("Export to Dropbox", [
                dbc.Row([
                    dbc.Col([
                        html.Div("Destination Folder", style=LABEL_STYLE),
                        dbc.InputGroup([
                            dbc.Input(id="dropbox-folder", value=DROPBOX_DEFAULT_DEST,
                                      size="sm", readonly=True),
                            dbc.Button("Browse…", id="fb-open", size="sm",
                                       style=BTN_DARK),
                        ]),
                    ], md=7),
                    dbc.Col([
                        html.Div("File Name", style=LABEL_STYLE),
                        dbc.Input(id="dropbox-filename", size="sm",
                                  placeholder="e.g. Sample 1 - Default: [Sample Name]"),
                    ], md=5),
                ], className="g-2 mb-2 align-items-end"),
                dbc.Button([html.I(className="bi bi-cloud-upload me-1"),
                            "Download to Dropbox"],
                           id="btn-dropbox-download",
                           style=BTN_DARK, className="w-100 mb-2",
                           size="sm"),
                dbc.Button([html.I(className="bi bi-table me-1"),
                            "Upload Session Summary"],
                           id="btn-dropbox-summary",
                           style=BTN_DARK, className="w-100 mb-2",
                           size="sm",
                           title="Upload one summarized row per plot for the "
                                 "whole session (current or loaded) as a CSV "
                                 "to the destination folder."),
                html.Hr(className="my-2"),
                dbc.Row([
                    dbc.Col([
                        html.Div("Trial File (.xlsx)", style=LABEL_STYLE),
                        dbc.InputGroup([
                            dbc.Input(id="append-file", size="sm", readonly=True,
                                      placeholder="No file selected"),
                            dbc.Button("Browse…", id="afb-open", size="sm",
                                       style=BTN_DARK),
                        ]),
                    ]),
                ], className="g-2 mb-2 align-items-end"),
                dbc.Button([html.I(className="bi bi-file-earmark-plus me-1"),
                            "Append to Dropbox File"],
                           id="btn-dropbox-append",
                           style=BTN_DARK, className="w-100",
                           size="sm",
                           title="Append every run from this session to the "
                                 "selected trial file — each summary (mean/"
                                 "variance/skewness/kurtosis) fills its own "
                                 "Plot's row"),
                html.Div(id="dropbox-result", className="mt-2"),
            ]),
        ], lg=7),
    ]),

    # Status bar
    dbc.Row(dbc.Col(html.Div("Ready", id="status-bar", style={
        "backgroundColor": "#222", "color": "#eee", "padding": "6px 12px",
        "fontSize": "12px", "fontFamily": "monospace", "borderRadius": "4px",
    }), width=12), className="mt-2 mb-4"),

    # Dropbox folder browser modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Select Dropbox Folder")),
        dbc.ModalBody([
            dbc.ButtonGroup([
                dbc.Button([html.I(className="bi bi-house me-1"), "Root"],
                           id="fb-root", size="sm", style=BTN_DARK),
                dbc.Button([html.I(className="bi bi-arrow-up me-1"), "Up"],
                           id="fb-up", size="sm", style=BTN_DARK),
            ], className="mb-2"),
            html.Div(id="fb-current", className="small text-muted mb-2"),
            html.Div(id="fb-listing",
                     style={"maxHeight": "320px", "overflowY": "auto"}),
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="fb-cancel", style=BTN_DARK, size="sm"),
            dbc.Button("Use This Folder", id="fb-confirm", style=BTN_DARK, size="sm"),
        ]),
    ], id="fb-modal", size="lg", is_open=False),

    # Trial-file browser modal (append target — folders navigate, .xlsx selects)
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Select Trial File (.xlsx)")),
        dbc.ModalBody([
            dbc.ButtonGroup([
                dbc.Button([html.I(className="bi bi-house me-1"), "Root"],
                           id="afb-root", size="sm", style=BTN_DARK),
                dbc.Button([html.I(className="bi bi-arrow-up me-1"), "Up"],
                           id="afb-up", size="sm", style=BTN_DARK),
            ], className="mb-2"),
            html.Div(id="afb-current", className="small text-muted mb-2"),
            html.Div(id="afb-listing",
                     style={"maxHeight": "320px", "overflowY": "auto"}),
        ]),
        dbc.ModalFooter(
            dbc.Button("Cancel", id="afb-cancel", style=BTN_DARK, size="sm")),
    ], id="afb-modal", size="lg", is_open=False),

    html.Footer("© 2026 Peanut Breeding & Genetics",
                className="text-center text-muted small mb-4"),
], fluid=True, className="px-3")


# ═════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ═════════════════════════════════════════════════════════════════════════════

# ── Poll: live stats, log, session table, button states, status bar ──────────
@app.callback(
    Output("live-count", "children"),
    Output("live-caption", "children"),
    Output("live-elapsed", "children"),
    Output("live-fps", "children"),
    Output("live-active", "children"),
    Output("log-box", "children"),
    Output("session-table", "data"),
    Output("status-bar", "children"),
    Output("btn-start", "disabled"),
    Output("btn-stop", "disabled"),
    Output("btn-next", "disabled"),
    Output("btn-initial", "disabled"),
    Output("btn-target", "disabled"),
    Output("plot-name", "disabled"),
    Output("plot-weight", "disabled"),
    Output("mode-radio", "options"),
    Input("poll", "n_intervals"),
)
def poll_state(_n):
    s = CONTROLLER.snapshot()
    running = s["running"]
    calibrating = s["calibrating"]
    busy = running or bool(calibrating)
    caption = f"Unique {s['label']}s"

    if calibrating:
        nice = "Initial Camera Setup" if calibrating == "initial" else "Target Calibration"
        status = f"{nice} open — finish in the calibration window; config reloads on close."
    elif s["error"]:
        status = f"Error — {s['error'].splitlines()[0]}"
    elif running:
        status = (f"Running  |  Unique {s['label']}s: {s['count']}  |  "
                  f"Active: {s['active']}  |  {s['fps']:.1f} fps")
    elif s["session"]:
        status = "Stopped — results saved. Ready for next sample."
    else:
        status = "Ready"

    mode_opts = [
        {"label": "Pod", "value": "pod", "disabled": busy},
        {"label": "Seed", "value": "seed", "disabled": busy},
    ]

    return (
        str(s["count"]), caption, s["elapsed"], f"{s['fps']:.1f} fps",
        f"Active: {s['active']}", s["log"], s["session"], status,
        busy, not running, busy, busy, busy, running, running, mode_opts,
    )


# ── Source status (mode change / refresh) ────────────────────────────────────
@app.callback(
    Output("source-status", "children"),
    Output("source-status", "style"),
    Input("mode-radio", "value"),
    Input("refresh-btn", "n_clicks"),
)
def update_source(mode, _n):
    if mode:
        CONTROLLER.set_mode(mode)
    text, color = CONTROLLER.source_status()
    return text, {"fontFamily": "monospace", "color": color}


# ── Frame-saving path enable ─────────────────────────────────────────────────
@app.callback(
    Output("save-path", "disabled"),
    Input("save-frames", "value"),
    Input("poll", "n_intervals"),
)
def toggle_save_path(checked, _n):
    return (not checked) or CONTROLLER.snapshot()["running"]


# ── Start ─────────────────────────────────────────────────────────────────────
@app.callback(
    Output("action-alert", "children"),
    Input("btn-start", "n_clicks"),
    State("mode-radio", "value"),
    State("plot-name", "value"),
    State("plot-weight", "value"),
    State("save-frames", "value"),
    State("save-path", "value"),
    prevent_initial_call=True,
)
def on_start(_n, mode, name, weight, save_frames, save_path):
    err = CONTROLLER.start(mode, name, str(weight or ""),
                           bool(save_frames), save_path)
    if err:
        return dbc.Alert(err, color="danger", dismissable=True,
                         className="small mb-0", style={"whiteSpace": "pre-wrap"})
    return dbc.Alert("Pipeline starting…", color="success", dismissable=True,
                     className="small mb-0")


# ── Stop ──────────────────────────────────────────────────────────────────────
@app.callback(
    Output("action-alert", "children", allow_duplicate=True),
    Input("btn-stop", "n_clicks"),
    prevent_initial_call=True,
)
def on_stop(_n):
    CONTROLLER.stop()
    return dbc.Alert("Stopping…", color="warning", dismissable=True,
                     className="small mb-0")


# ── Calibration: launch the PyQt dialogs (separate process) ──────────────────
@app.callback(
    Output("action-alert", "children", allow_duplicate=True),
    Input("btn-initial", "n_clicks"),
    State("mode-radio", "value"),
    prevent_initial_call=True,
)
def on_initial_setup(_n, mode):
    err = CONTROLLER.launch_calibration("initial", mode or CONTROLLER.mode)
    if err:
        return dbc.Alert(err, color="danger", dismissable=True,
                         className="small mb-0")
    return dbc.Alert("Initial Camera Setup - Config Reloads When Completed", color="info", dismissable=True,
                     className="small mb-0")


@app.callback(
    Output("action-alert", "children", allow_duplicate=True),
    Input("btn-target", "n_clicks"),
    State("mode-radio", "value"),
    prevent_initial_call=True,
)
def on_target_cal(_n, mode):
    err = CONTROLLER.launch_calibration("target", mode or CONTROLLER.mode)
    if err:
        return dbc.Alert(err, color="danger", dismissable=True,
                         className="small mb-0")
    return dbc.Alert("Target Calibration - Config Reloads When Completed", color="info", dismissable=True,
                     className="small mb-0")


# ── Next sample (auto-increment trailing number) ─────────────────────────────
@app.callback(
    Output("plot-name", "value"),
    Output("plot-weight", "value"),
    Input("btn-next", "n_clicks"),
    State("plot-name", "value"),
    prevent_initial_call=True,
)
def on_next(_n, name):
    import re
    name = (name or "").strip()
    if not name:
        return "", None
    m = re.search(r"(\d+)$", name)
    if m:
        nxt = f"{name[:m.start()]}{int(m.group(1)) + 1}"
    else:
        nxt = f"{name}_2"
    return nxt, None


# ── Sessions: list / load previous sessions ──────────────────────────────────
@app.callback(
    Output("session-select", "options"),
    Input("btn-refresh-sessions", "n_clicks"),
)   # also fires at page load, so the list is ready without clicking refresh
def refresh_sessions(_n):
    return [{"label": s["label"], "value": s["path"]}
            for s in CONTROLLER.list_sessions()]


@app.callback(
    Output("session-load-result", "children"),
    Output("session-table", "selected_rows"),
    Input("btn-load-session", "n_clicks"),
    State("session-select", "value"),
    prevent_initial_call=True,
)
def load_session_cb(_n, path):
    if not path:
        return (html.Span("Choose a session first.", className="text-warning"),
                dash.no_update)
    n, err = CONTROLLER.load_session(path)
    if err:
        return html.Span(err, className="text-danger"), dash.no_update
    return (html.Span(f"✓ Loaded {n} run(s) — new runs will continue in this "
                      "session; select a row to view/export a run.",
                      className="text-success"), [])


# Selecting a history row switches the plot to that run; deselecting returns
# to the latest run.
@app.callback(
    Output("viz-source", "value"),
    Input("session-table", "selected_rows"),
    State("viz-source", "value"),
    prevent_initial_call=True,
)
def on_row_select(rows, src):
    if rows:
        return "selected"
    return "latest" if src == "selected" else dash.no_update


def _resolve_export_run(selected_rows):
    """(csv_path, sample_name) for the selected history row, else latest run.

    Returns (None, None) when there is nothing to export.
    """
    snap = CONTROLLER.snapshot()
    runs = snap["session"]
    if selected_rows:
        i = selected_rows[0]
        if 0 <= i < len(runs):
            r = runs[i]
            csvf = r.get("csv_file")
            if csvf and csvf != "(none)":
                p = Path(CONTROLLER.cfg.get("output_dir", "./output")) / csvf
                if p.exists():
                    return p, (r.get("plot_name") or "").strip()
        return None, None
    last = snap["last_csv"]
    if last and Path(last).exists():
        return Path(last), (snap.get("last_sample") or "").strip()
    return None, None


# ── Visualization: load data into the store ──────────────────────────────────
@app.callback(
    Output("viz-store", "data"),
    Input("viz-source", "value"),
    Input("session-table", "selected_rows"),
    Input("poll", "n_intervals"),
    State("viz-store", "data"),
)
def load_viz_data(src, selected_rows, _n, current):
    if src == "selected" and selected_rows:
        # A history row (current or loaded session) — read its raw CSV.
        runs = CONTROLLER.snapshot()["session"]
        if selected_rows[0] < len(runs):
            r = runs[selected_rows[0]]
            csvf = r.get("csv_file")
            if csvf and csvf != "(none)":
                p = Path(CONTROLLER.cfg.get("output_dir", "./output")) / csvf
                if current and current.get("source") == str(p):
                    return dash.no_update  # already showing; ignore the poll
                try:
                    df = pd.read_csv(p)
                    return {"label": f"{r.get('mode', '')} — "
                                     f"{r.get('plot_name', '')}",
                            "records": df.to_dict("records"),
                            "source": str(p)}
                except Exception:  # noqa: BLE001
                    pass
        # Selected row unusable (no CSV / stale index) → fall back to latest.

    # latest recording (default whenever nothing valid is selected)
    last = CONTROLLER.snapshot()["last_csv"]
    if not last:
        return None if current is not None else dash.no_update
    if current and current.get("source") == last:
        return dash.no_update  # already loaded; avoid re-reading every second
    try:
        df = pd.read_csv(last)
    except Exception:  # noqa: BLE001
        return dash.no_update
    return {"label": _mode_label_from_name(Path(last).name),
            "records": df.to_dict("records"), "source": last}


# ── Visualization: render figure + stats ─────────────────────────────────────
@app.callback(
    Output("viz-graph", "figure"),
    Output("viz-stats", "children"),
    Input("viz-store", "data"),
    Input("viz-metric", "value"),
)
def render_viz(store, metric):
    if not store or not store.get("records"):
        return build_distribution_figure(None, metric, "Sample"), []
    df = pd.DataFrame(store["records"])
    label = store.get("label", "Sample")
    fig = build_distribution_figure(df, metric, label)
    return fig, summary_cards(df, metric)


# ── Dropbox folder browser ───────────────────────────────────────────────────
@app.callback(
    Output("fb-modal", "is_open"),
    Output("dropbox-folder", "value"),
    Output("fb-selected-folder", "data"),
    Input("fb-open", "n_clicks"),
    Input("fb-cancel", "n_clicks"),
    Input("fb-confirm", "n_clicks"),
    State("fb-path", "data"),
    State("dropbox-folder", "value"),
    State("fb-selected-folder", "data"),
    prevent_initial_call=True,
)
def fb_modal(open_c, cancel_c, confirm_c, cur_path, cur_folder, selected):
    trig = dash.ctx.triggered_id
    if trig == "fb-open":
        return True, cur_folder, selected
    if trig == "fb-confirm":
        return False, cur_path or DROPBOX_ROOT, cur_path or DROPBOX_ROOT
    return False, cur_folder, selected  # cancel


@app.callback(
    Output("fb-path", "data"),
    Input("fb-open", "n_clicks"),
    Input("fb-root", "n_clicks"),
    Input("fb-up", "n_clicks"),
    Input({"type": "fb-folder", "path": dash.ALL}, "n_clicks"),
    State("fb-path", "data"),
    State({"type": "fb-folder", "path": dash.ALL}, "id"),
    prevent_initial_call=True,
)
def fb_navigate(open_c, root_c, up_c, folder_clicks, cur_path, folder_ids):
    trig = dash.ctx.triggered_id
    if trig == "fb-open" or trig == "fb-root":
        return DROPBOX_ROOT
    if trig == "fb-up":
        parent = "/".join((cur_path or DROPBOX_ROOT).rstrip("/").split("/")[:-1])
        return parent or DROPBOX_ROOT
    if isinstance(trig, dict) and trig.get("type") == "fb-folder":
        return trig["path"]
    return cur_path or DROPBOX_ROOT


@app.callback(
    Output("fb-listing", "children"),
    Output("fb-current", "children"),
    Input("fb-path", "data"),
    Input("fb-modal", "is_open"),
)
def fb_render(path, is_open):
    if not is_open:
        return [], ""
    path = path or DROPBOX_ROOT
    disp = path if path else "/  (NC State PB&G team-space root)"
    if team_manager is None:
        return html.Div("Dropbox client not configured.",
                        className="text-danger small"), disp
    try:
        items = team_manager.get_folder_contents(path)
    except Exception as e:  # noqa: BLE001
        return html.Div(f"Error: {e}", className="text-danger small"), disp
    folders = sorted([i for i in items if i["type"] == "folder"],
                     key=lambda x: x["name"].lower())
    rows = [
        html.Div([html.I(className="bi bi-folder-fill me-2",
                         style={"color": "#CC0000"}), f["name"]],
                 id={"type": "fb-folder", "path": f["path"]},
                 n_clicks=0, className="fb-item",
                 style={"cursor": "pointer", "padding": "6px 8px"})
        for f in folders
    ]
    if not rows:
        rows = [html.Div("No subfolders.", className="text-muted small p-2")]
    return rows, disp


# ── QR scan: handheld reader types Year|Location|NC_Accession|Name|Plot ──────
# The reader emulates a keyboard into the focused QR box and ends with Enter;
# debounce=True fires this callback on that Enter (or on blur for readers that
# send Tab). The payload becomes the underscore sample name, which drives the
# raw CSV name, the image folder, and the auto-Plot at append time.
@app.callback(
    Output("plot-name", "value", allow_duplicate=True),
    Output("qr-scan", "value"),
    Output("qr-scan-result", "children"),
    Output("man-year", "value"),
    Output("man-loc", "value"),
    Output("man-acc", "value"),
    Output("man-name", "value"),
    Output("man-plot", "value"),
    Input("qr-scan", "value"),
    prevent_initial_call=True,
)
def qr_scanned(text):
    if not (text or "").strip():
        return (dash.no_update,) * 8
    qr = parse_qr(text)
    if qr is None:
        return (dash.no_update, dash.no_update,
                html.Span("Unrecognized QR — expected "
                          "Year|Location|Accession|Name|Plot",
                          className="text-danger"),
                dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update)
    # Mirror the parsed payload into the Manual Entry fields so the operator
    # can see/correct it and "Set Sample Information" is already satisfied.
    fields = (qr["Year"], qr["Location"], qr["NC_Accession"],
              qr["Name"], qr["Plot"])
    name = qr_to_sample_name(text)
    msg = (f"✓ Plot {qr['Plot']} — {qr['Name']} "
           f"({qr['Year']} {qr['Location']}, {qr['NC_Accession']})")
    if not qr["Plot"].isdigit():
        CONTROLLER.pending_plot = None
        return (name, "", html.Span(
            msg + " — warning: Plot is not a number, so Dropbox append "
                  "won't match it automatically.", className="text-warning"),
            *fields)
    # Register the scanned plot for the next run (supersedes any manual set).
    CONTROLLER.pending_plot = qr["Plot"]
    return (name, "", html.Span(msg, className="text-success"), *fields)


# ── Manual entry: register the plot for the next run ─────────────────────────
# The Sample Name stays the operator's own file/folder designation — this
# never overwrites it. Plot (numeric) is what the Dropbox append merges on;
# it's registered here and consumed by the next run. Year/Location/Accession/
# Name are optional context and may be left blank.
@app.callback(
    Output("manual-result", "children"),
    Input("btn-manual-set", "n_clicks"),
    State("man-plot", "value"),
    State("plot-name", "value"),
    prevent_initial_call=True,
)
def manual_set(_n, plot, sample_name):
    plot = str(plot).strip() if plot else ""
    sample_name = (sample_name or "").strip()
    if not plot or not sample_name:
        return html.Span("Plot and Sample Name are both required.",
                         className="text-danger")
    if not plot.isdigit():
        return html.Span("Plot must be a whole number.",
                         className="text-danger")
    CONTROLLER.pending_plot = plot
    return html.Span(f"✓ Set — files named '{sample_name}', data appends to "
                     f"Plot {plot}.", className="text-success")


# ── Trial-file browser (append target) ───────────────────────────────────────
@app.callback(
    Output("afb-modal", "is_open"),
    Output("append-file", "value"),
    Output("afb-selected", "data"),
    Input("afb-open", "n_clicks"),
    Input("afb-cancel", "n_clicks"),
    Input({"type": "afb-file", "path": dash.ALL}, "n_clicks"),
    State("afb-selected", "data"),
    prevent_initial_call=True,
)
def afb_modal(open_c, cancel_c, file_clicks, selected):
    trig = dash.ctx.triggered_id
    if trig == "afb-open":
        return True, dash.no_update, dash.no_update
    if isinstance(trig, dict) and trig.get("type") == "afb-file":
        # Ignore the phantom trigger fired when the listing (re)renders and
        # its pattern components are created with n_clicks=0.
        if not dash.ctx.triggered[0]["value"]:
            raise dash.exceptions.PreventUpdate
        return False, trig["path"], trig["path"]
    return False, dash.no_update, selected  # cancel


@app.callback(
    Output("afb-path", "data"),
    Input("afb-open", "n_clicks"),
    Input("afb-root", "n_clicks"),
    Input("afb-up", "n_clicks"),
    Input({"type": "afb-folder", "path": dash.ALL}, "n_clicks"),
    State("afb-path", "data"),
    prevent_initial_call=True,
)
def afb_navigate(open_c, root_c, up_c, folder_clicks, cur_path):
    trig = dash.ctx.triggered_id
    if trig in ("afb-open", "afb-root"):
        return DROPBOX_ROOT
    if trig == "afb-up":
        parent = "/".join((cur_path or DROPBOX_ROOT).rstrip("/").split("/")[:-1])
        return parent or DROPBOX_ROOT
    if isinstance(trig, dict) and trig.get("type") == "afb-folder":
        if not dash.ctx.triggered[0]["value"]:
            raise dash.exceptions.PreventUpdate
        return trig["path"]
    return cur_path or DROPBOX_ROOT


@app.callback(
    Output("afb-listing", "children"),
    Output("afb-current", "children"),
    Input("afb-path", "data"),
    Input("afb-modal", "is_open"),
)
def afb_render(path, is_open):
    if not is_open:
        return [], ""
    path = path or DROPBOX_ROOT
    disp = path if path else "/  (NC State PB&G team-space root)"
    if team_manager is None:
        return html.Div("Dropbox client not configured.",
                        className="text-danger small"), disp
    try:
        items = team_manager.get_folder_contents(path)
    except Exception as e:  # noqa: BLE001
        return html.Div(f"Error: {e}", className="text-danger small"), disp
    folders = sorted([i for i in items if i["type"] == "folder"],
                     key=lambda x: x["name"].lower())
    files = sorted([i for i in items if i["type"] == "file"
                    and i["name"].lower().endswith(".xlsx")],
                   key=lambda x: x["name"].lower())
    rows = [
        html.Div([html.I(className="bi bi-folder-fill me-2",
                         style={"color": "#CC0000"}), f["name"]],
                 id={"type": "afb-folder", "path": f["path"]},
                 n_clicks=0, className="fb-item",
                 style={"cursor": "pointer", "padding": "6px 8px"})
        for f in folders
    ] + [
        html.Div([html.I(className="bi bi-file-earmark-spreadsheet me-2",
                         style={"color": "#1a7f37"}), f["name"]],
                 id={"type": "afb-file", "path": f["path"]},
                 n_clicks=0, className="fb-item",
                 style={"cursor": "pointer", "padding": "6px 8px"})
        for f in files
    ]
    if not rows:
        rows = [html.Div("No subfolders or .xlsx files.",
                         className="text-muted small p-2")]
    return rows, disp


# ── Dropbox: append session summaries to the selected trial file, by Plot ────
# Appends EVERY completed run from this session in one download → merge-all →
# upload cycle. Each run's Plot comes from its own sample name (QR scan and
# manual entry both guarantee a trailing plot number). Re-appending is
# idempotent — identical values fill the same cells; if a plot was graded
# twice, the later run wins.
@app.callback(
    Output("dropbox-result", "children", allow_duplicate=True),
    Input("btn-dropbox-append", "n_clicks"),
    State("append-file", "value"),
    prevent_initial_call=True,
)
def dropbox_append(_n, file_path):
    def _alert(msg, color):
        return dbc.Alert(msg, color=color, dismissable=True,
                         className="small mb-0")
    if team_manager is None:
        return _alert("Dropbox client not configured (check .env).", "danger")
    if not (file_path or "").strip():
        return _alert("Choose a trial file first (Browse…).", "warning")

    snap = CONTROLLER.snapshot()
    runs = [r for r in snap["session"]
            if r.get("csv_file") and r["csv_file"] != "(none)"]
    if not runs:
        return _alert("No completed runs this session yet.", "warning")

    out_dir = Path(CONTROLLER.cfg.get("output_dir", "./output"))

    # Resolve each run's plot (registered at scan / Set Sample Information,
    # with a plot-suffixed sample name as fallback) + its CSV.
    todo, skipped = [], []
    for r in runs:
        plot = r.get("plot") or parse_plot(r.get("plot_name") or "")
        if not plot:
            skipped.append(f"#{r['n']} '{r.get('plot_name') or 'unnamed'}' (no plot)")
            continue
        csv_path = out_dir / r["csv_file"]
        if not csv_path.exists():
            skipped.append(f"#{r['n']} '{r.get('plot_name')}' (csv missing)")
            continue
        todo.append((r, plot, csv_path, (r.get("weight") or "").rstrip("g")))
    if not todo:
        return _alert("No runs with a Plot to append — scan the tag or use "
                      "Manual Entry's Set Sample Information before each run. "
                      "Skipped: " + "; ".join(skipped), "warning")

    try:
        blob = team_manager.get_file(file_path)
        if not blob:
            return _alert(f"Could not download {file_path}.", "danger")
        appended, failed = [], []
        for r, plot, csv_path, weight in todo:
            try:
                row = compute_summary(csv_path, mode=r.get("mode") or "",
                                      plot_weight=weight, plot=plot)
                blob = merge_summary_into_workbook(blob, row)
                appended.append(f"{plot} ({r.get('mode')})")
            except ValueError as e:
                failed.append(str(e))
        if not appended:
            return _alert("Nothing appended. " + " | ".join(failed), "danger")
        team_manager.upload_file(blob, file_path, mode="overwrite")
    except Exception as e:  # noqa: BLE001
        return _alert(f"Append failed: {e}", "danger")

    parts = [f"Appended {len(appended)} run(s) to {file_path}: "
             + ", ".join(appended)]
    if failed:
        parts.append("Failed: " + " | ".join(failed))
    if skipped:
        parts.append("Skipped: " + "; ".join(skipped))
    color = "success" if not failed and not skipped else "warning"
    return _alert(" — ".join(parts), color)


# ── Local: download the whole session summarized (one row per plot) ──────────
@app.callback(
    Output("local-download", "data", allow_duplicate=True),
    Output("local-result", "children", allow_duplicate=True),
    Input("btn-summary-download", "n_clicks"),
    prevent_initial_call=True,
)
def summary_download(_n):
    snap = CONTROLLER.snapshot()
    df = session_summary_table(snap["session"],
                               CONTROLLER.cfg.get("output_dir", "./output"))
    if df.empty:
        return dash.no_update, dbc.Alert(
            "No completed runs to summarize yet.", color="warning",
            className="small mb-0")
    name = f"{CONTROLLER.session_path.stem}_summary.csv"
    return (dcc.send_string(df.to_csv(index=False), filename=name),
            dbc.Alert(f"Downloading {name} — {len(df)} plot(s) summarized",
                      color="success", dismissable=True, className="small mb-0"))


# ── Dropbox: upload the whole session summarized (one row per plot) ──────────
@app.callback(
    Output("dropbox-result", "children", allow_duplicate=True),
    Input("btn-dropbox-summary", "n_clicks"),
    State("dropbox-folder", "value"),
    prevent_initial_call=True,
)
def dropbox_summary(_n, folder):
    if team_manager is None:
        return dbc.Alert("Dropbox client not configured (check .env).",
                         color="danger", className="small mb-0")
    snap = CONTROLLER.snapshot()
    df = session_summary_table(snap["session"],
                               CONTROLLER.cfg.get("output_dir", "./output"))
    if df.empty:
        return dbc.Alert("No completed runs to summarize yet.",
                         color="warning", className="small mb-0")
    folder = (folder or "").strip()
    if folder == DROPBOX_DEFAULT_DEST:
        folder = DROPBOX_ROOT
    name = f"{CONTROLLER.session_path.stem}_summary.csv"
    dest = f"{folder.rstrip('/')}/{name}"
    try:
        # overwrite: re-uploading the same session updates its summary file
        meta = team_manager.upload_file(df.to_csv(index=False).encode(),
                                        dest, mode="overwrite")
        return dbc.Alert(
            f"Uploaded {len(df)} plot(s) to "
            f"{getattr(meta, 'path_display', dest)}",
            color="success", dismissable=True, className="small mb-0")
    except Exception as e:  # noqa: BLE001
        return dbc.Alert(f"Upload failed: {e}", color="danger",
                         dismissable=True, className="small mb-0")


# ── Dropbox: download (upload latest recording CSV to chosen folder) ─────────
@app.callback(
    Output("dropbox-result", "children"),
    Input("btn-dropbox-download", "n_clicks"),
    State("dropbox-folder", "value"),
    State("dropbox-filename", "value"),
    State("session-table", "selected_rows"),
    prevent_initial_call=True,
)
def dropbox_download(_n, folder, filename, selected_rows):
    if team_manager is None:
        return dbc.Alert("Dropbox client not configured (check .env).",
                         color="danger", className="small mb-0")
    csv_path, sample = _resolve_export_run(selected_rows)
    if csv_path is None:
        return dbc.Alert("No recording available to upload for the selected "
                         "run yet.", color="warning", className="small mb-0")
    # Blank -> the run's sample name (as the placeholder says), then the
    # CSV's own name as a last resort.
    name = (filename or "").strip() or sample or csv_path.name
    if not name.lower().endswith(".csv"):
        name += ".csv"
    # The default destination shown in the box is a display label for the
    # team-space root — map it to the real root path ("") for the API.
    folder = (folder or "").strip()
    if folder == DROPBOX_DEFAULT_DEST:
        folder = DROPBOX_ROOT
    dest = f"{folder.rstrip('/')}/{name}"
    try:
        data = csv_path.read_bytes()
        meta = team_manager.upload_file(data, dest, mode="add")
        return dbc.Alert(f"Uploaded to {getattr(meta, 'path_display', dest)}",
                         color="success", dismissable=True, className="small mb-0")
    except Exception as e:  # noqa: BLE001
        return dbc.Alert(f"Upload failed: {e}", color="danger",
                         dismissable=True, className="small mb-0")


# ── Local: download the latest recording CSV to the operator's computer ──────
# No Dropbox / no .env needed — streams the file straight to the browser, so
# anyone running a station can keep their data.
@app.callback(
    Output("local-download", "data"),
    Output("local-result", "children"),
    Input("btn-local-download", "n_clicks"),
    State("local-filename", "value"),
    State("session-table", "selected_rows"),
    prevent_initial_call=True,
)
def local_download(_n, filename, selected_rows):
    csv_path, sample = _resolve_export_run(selected_rows)
    if csv_path is None:
        return dash.no_update, dbc.Alert(
            "No recording available to download for the selected run yet.",
            color="warning", className="small mb-0")
    # Blank file name -> the run's sample name; fall back to the CSV's name.
    name = (filename or "").strip() or sample or csv_path.name
    if not name.lower().endswith(".csv"):
        name += ".csv"
    return (dcc.send_file(str(csv_path), filename=name),
            dbc.Alert(f"Downloading {name}", color="success",
                      dismissable=True, className="small mb-0"))


if __name__ == "__main__":
    # Per-station overrides (set in the launcher): PBG_HOST=0.0.0.0 serves the
    # app to other machines on the LAN; default 127.0.0.1 is this machine only.
    host = os.environ.get("PBG_HOST", "127.0.0.1")
    port = int(os.environ.get("PBG_PORT", "8050"))
    url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}"
    print(f"PBG Grading — open {url}")
    print("NOTE: run this on the machine connected to the OAK cameras.")
    if os.environ.get("PBG_NO_BROWSER", "") not in ("1", "true", "yes"):
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    app.run(debug=False, host=host, port=port)
