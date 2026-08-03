"""
Backend for PBG Grading: thin controllers over the on-camera pod-counter
HTTP service, plus shared session bookkeeping.

Each OAK 4 camera runs the pod-counter oakapp (deployed from the Oak 4
App repository), which owns the whole DepthAI pipeline — detection,
tracking, measurement, mm conversion, colour — and exposes an HTTP API
on port 8808. This module holds one CameraController per camera:

    * a daemon thread polls GET /status once a second into a cached,
      lock-guarded snapshot the Dash UI reads;
    * start()/stop() map to POST /run/start and POST /run/stop — the
      finished CSV comes back in the stop response and is written to the
      local output dir under the same naming the export flows expect;
    * a run started here keeps counting on the camera even if this app
      dies; on reconnect the poll loop adopts the in-progress run;
    * oakctl wrappers start/stop/restart the app container itself
      (cameras are powered 24/7 but boot-disabled to save energy).

No Dash, Qt, or DepthAI imports here — this module is importable (and
testable) without a browser or a camera.
"""

import collections
import csv
import io
import json
import threading
import time
from pathlib import Path
from typing import List, Optional

import oakctl_tools
from camera_client import CameraClient, CameraClientError, DEFAULT_PORT


def format_elapsed(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    return f"{m:02d}:{s:02d}"


def safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in name)


# ═════════════════════════════════════════════════════════════════════════════
# SESSION LOG  — shared across both cameras
# ═════════════════════════════════════════════════════════════════════════════

class SessionLog:
    """Per-camera session history (one row per completed run) + JSON
    persistence.

    Every completed run rewrites the session file under output/Sessions/,
    so an accidentally closed app loses nothing and a previous session
    can be reloaded later. Loading an old session repoints session_path
    so subsequent runs continue in that file. Pod and seed keep separate
    sessions (separate files, separate history, independent reload).
    """

    def __init__(self, output_dir: str | Path, mode: str):
        self._lock = threading.Lock()
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.sessions_dir = self.output_dir / "Sessions"
        self.session_path = (
            self.sessions_dir /
            f"session_{mode}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        self.runs: List[dict] = []
        self.name: str = ""              # operator-given session name
        self.last_csv_path: Optional[Path] = None
        self.last_csv_time: float = 0.0
        self.last_plot: str = ""

    def set_name(self, name: str) -> None:
        with self._lock:
            self.name = str(name or "").strip()
        self._persist()

    def start_new(self, name: str = "") -> None:
        """Close out the current session and begin a fresh one in the same
        app instance. Nothing is lost — the previous session is already
        persisted under its own timestamped file and can be reloaded."""
        with self._lock:
            self.runs = []
            self.name = str(name or "").strip()
            self.session_path = (
                self.sessions_dir /
                f"session_{self.mode}_{time.strftime('%Y%m%d_%H%M%S')}.json")
            self.last_csv_path = None
            self.last_csv_time = 0.0
            self.last_plot = ""
        self._persist()

    def add_run(self, mode_label: str, plot_name: str, plot: str,
                weight: str, count: int, elapsed: str, csv_file: str,
                csv_path: Optional[Path] = None) -> int:
        with self._lock:
            self.runs.append({
                "n": len(self.runs) + 1,
                "mode": mode_label,
                "plot_name": plot_name,     # full sample name (QR payload)
                "plot": plot,               # trial-file merge key
                "weight": f"{weight}g" if weight else "",
                "count": count,
                "elapsed": elapsed,
                "csv_file": csv_file,
                "appended": "",             # auto-append outcome (✓ / FAILED)
            })
            if csv_path is not None:
                self.last_csv_path = csv_path
                self.last_csv_time = time.time()
                self.last_plot = plot
            row_n = self.runs[-1]["n"]
        self._persist()
        return row_n

    def set_append_status(self, n: int, status: str) -> None:
        with self._lock:
            run = next((r for r in self.runs if r.get("n") == n), None)
            if run is not None:
                run["appended"] = status
        self._persist()

    def set_weight(self, n: int, weight: str) -> None:
        """Late weight entry (seed samples are weighed after shelling —
        i.e. after the run). Updates the session row AND stamps the
        plot_weight column of the run's CSV so downstream summaries and
        the raw file agree."""
        weight = str(weight).strip()
        with self._lock:
            run = next((r for r in self.runs if r.get("n") == n), None)
            if run is None:
                return
            run["weight"] = f"{weight}g" if weight else ""
            csvf = run.get("csv_file")
        if weight and csvf and csvf != "(none)":
            path = self.output_dir / csvf
            try:
                import csv as _csv
                import io as _io
                rows = list(_csv.DictReader(
                    _io.StringIO(path.read_text(encoding="utf-8"))))
                if rows:
                    for r in rows:
                        r["plot_weight"] = weight
                    buf = _io.StringIO()
                    w = _csv.DictWriter(buf, fieldnames=list(rows[0].keys()),
                                        lineterminator="\n")
                    w.writeheader()
                    w.writerows(rows)
                    path.write_text(buf.getvalue(), encoding="utf-8")
            except OSError:
                pass
        self._persist()

    def _persist(self) -> None:
        with self._lock:
            runs = list(self.runs)
            path = self.session_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "name": self.name, "runs": runs}
            path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        except Exception as e:  # noqa: BLE001 — never let IO kill the run flow
            print(f"WARNING: could not save session file: {e}")

    def list_sessions(self) -> List[dict]:
        """This camera's saved sessions, newest first: {path, label, n_runs}."""
        out = []
        pattern = f"session_{self.mode}_*.json"
        for p in sorted(self.sessions_dir.glob(pattern), reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                runs = data.get("runs", [])
                ts = time.strftime(
                    "%b %d %Y, %H:%M",
                    time.strptime(p.stem, f"session_{self.mode}_%Y%m%d_%H%M%S"))
                name = (data.get("name") or "").strip()
                label = f"{name} — {ts}" if name else ts
                out.append({"path": str(p), "n_runs": len(runs),
                            "label": f"{label} — {len(runs)} run(s)"})
            except Exception:  # noqa: BLE001 — skip unreadable files
                continue
        return out

    def load_session(self, path: str) -> tuple[int, Optional[str]]:
        """Replace the in-memory session with a saved one and resume it.

        Returns (n_runs, error). Safe even mid-testing: the current
        session is already on disk (every run persists), so nothing is
        lost by switching.
        """
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            runs = data.get("runs", [])
            if not isinstance(runs, list):
                return 0, "Not a session file."
        except Exception as e:  # noqa: BLE001
            return 0, f"Could not read session: {e}"
        with self._lock:
            self.runs = runs
            self.name = (data.get("name") or "").strip()
            self.session_path = Path(path)   # new runs continue in this file
        return len(runs), None

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "session": list(self.runs),
                "name": self.name,
                "last_csv": str(self.last_csv_path) if self.last_csv_path else "",
                "last_csv_time": self.last_csv_time,
                "last_plot": self.last_plot,
            }


# ═════════════════════════════════════════════════════════════════════════════
# CAMERA CONTROLLER  — one per camera / mode
# ═════════════════════════════════════════════════════════════════════════════

class CameraController:
    """Bridge between the Dash UI and one camera's pod-counter service.

    All public state is guarded by a lock and read via snapshot(). Slow
    operations (stop_run's CSV wait, oakctl invocations) run in worker
    threads so Dash callbacks return immediately; their outcomes land in
    the log and in the polled status.
    """

    def __init__(self, mode: str, label: str, ip: str,
                 port: int, session: SessionLog,
                 weight_after_stop: bool = False):
        self.mode = mode
        self.label = label
        self.ip = ip
        self.port = port
        self.client = CameraClient(ip, port)
        self.session = session
        # Seed samples are weighed after shelling, so weight is collected
        # via a popup AFTER the run stops instead of being required at
        # start (pod keeps weight-before-start).
        self.weight_after_stop = weight_after_stop
        self.pending_weight: Optional[dict] = None

        # Post-run hook — the UI wires this to the trial-file auto-append.
        # Called (controller, run_info) in the stop worker thread once a
        # run is fully complete, i.e. after the post-run weight for
        # weighed-after-shelling modes.
        self.on_run_complete = None
        self._deferred_run: Optional[dict] = None

        self._lock = threading.Lock()
        self.status: Optional[dict] = None     # last /status, None = offline
        self.logs: collections.deque = collections.deque(maxlen=400)
        self.stopping = False                  # a stop_run is in flight
        self.lifecycle_busy: Optional[str] = None   # e.g. "starting app"
        self.last_error: Optional[str] = None

        # The run we started or adopted (camera-side run_name), and the
        # metadata we knew at start time (fallbacks — the authoritative
        # plot_name/plot_weight are baked into the camera's CSV, but a
        # zero-count run has no data rows to read them back from).
        self._attached_run = ""
        self._run_plot = ""
        self._run_plot_name = ""
        self._run_plot_weight = ""
        self._run_meta: dict = {}

        self._auto_start_tried = False
        self._poll_thread: Optional[threading.Thread] = None
        self._stop_polling = threading.Event()

        # Background stored-run downloads. Serialized per camera (one
        # worker) — a 1 GB tarball fetch shouldn't compete with a second
        # one for the camera's CPU/network. Progress is polled by the UI.
        self.downloads: dict = {}          # name -> state dict
        self._dl_thread: Optional[threading.Thread] = None

    # ── polling ──────────────────────────────────────────────────────────────
    def start_polling(self, interval_s: float = 1.0) -> None:
        if self._poll_thread is not None:
            return
        self._poll_thread = threading.Thread(
            target=self._poll_loop, args=(interval_s,), daemon=True,
            name=f"poll-{self.mode}")
        self._poll_thread.start()

    def _poll_loop(self, interval_s: float) -> None:
        while not self._stop_polling.is_set():
            self.poll_once()
            self._stop_polling.wait(interval_s)

    def poll_once(self) -> None:
        """One /status round-trip; updates cached state and handles the
        offline→online, adopt-running-run, and run-ended transitions."""
        try:
            s = self.client.status()
        except CameraClientError:
            s = None

        with self._lock:
            was_online = self.status is not None
            self.status = s

        if s is None:
            if was_online:
                self._log(f"Camera offline ({self.ip}).")
            self._maybe_auto_start()
            return

        if not was_online:
            self._log(f"Camera online — pod-counter "
                      f"v{s.get('app_version', '?')} "
                      f"[{s.get('state', '?')}], staged "
                      f"{s.get('staged_mode', '?')} config "
                      f"({s.get('config_source', '?')}).")
            staged = s.get("staged_mode", "")
            if staged and staged != self.mode:
                self._log(f"WARNING: camera is staged as '{staged}' but this "
                          f"panel is '{self.mode}' — deploy the right config "
                          f"from the Oak 4 App repo.")

        state = s.get("state", "")
        run_name = s.get("run_name", "")

        # A run we didn't start (app survived a host restart, or someone
        # started it elsewhere) — adopt it so Stop works and counts show.
        if state == "running" and run_name and run_name != self._attached_run:
            with self._lock:
                self._attached_run = run_name
                self._run_plot_name = s.get("plot_name", "")
                self._run_plot = ""       # unknown for adopted runs
                self._run_plot_weight = ""
                self._run_meta = {}       # adopted runs never auto-append
            self._log(f"Adopted run already in progress on the camera: "
                      f"{run_name} (sample '{s.get('plot_name', '')}'). "
                      f"Stop works normally; the CSV arrives at stop.")

        # We thought a run was active but the camera is idle and no stop of
        # ours is in flight — it ended without us (app restart, oakctl stop).
        if state == "idle" and self._attached_run and not self.stopping:
            self._log(f"Run {self._attached_run} ended outside this app — "
                      f"its CSV (if any) is still on the camera under "
                      f"Stored Runs….")
            with self._lock:
                self._attached_run = ""

    def _maybe_auto_start(self) -> None:
        """One quiet oakctl start attempt when the camera app is down —
        cameras are powered 24/7 but boot-disabled to save energy."""
        if self._auto_start_tried:
            return
        self._auto_start_tried = True
        if oakctl_tools.find_oakctl() is None:
            self._log("Camera app is down and oakctl is not installed — "
                      "start it manually (oakctl app start) or install "
                      "oakctl for auto-start.")
            return
        self._log("Camera app appears down — trying `oakctl app start`…")

        def _try():
            try:
                oakctl_tools.app_start(self.ip)
                self._log("oakctl start requested — the app takes ~20 s "
                          "to come up.")
            except oakctl_tools.OakctlError as e:
                self._log(f"Auto-start failed: {e}")

        threading.Thread(target=_try, daemon=True).start()

    # ── run lifecycle ────────────────────────────────────────────────────────
    def start(self, plot_name: str, plot: str, plot_weight: str,
              save_frames: bool, require_weight: bool = True,
              meta: Optional[dict] = None) -> Optional[str]:
        """Start a run on the camera. Returns None on success or an error
        string. plot_name is the sample identifier written into the CSV
        (the full QR payload in grading mode; anything in test mode);
        plot is the trial-file merge key. `meta` rides along to the
        on_run_complete hook (e.g. the resolved trial file to append to);
        `require_weight=False` relaxes the weight requirement (test mode)."""
        plot_name = str(plot_name or "").strip()
        plot = str(plot or "").strip()
        plot_weight = str(plot_weight or "").strip()
        if not plot_name:
            return "Sample Name is required before starting."
        if not plot_weight and require_weight and not self.weight_after_stop:
            return "Weight is required before starting."
        if plot_weight:
            try:
                if float(plot_weight) <= 0:
                    return "Weight must be a positive number."
            except ValueError:
                return "Weight must be a number (grams)."

        with self._lock:
            s = self.status
            if self.stopping:
                return "The previous run is still stopping — wait a moment."
        if s is None:
            return (f"{self.label} camera is offline ({self.ip}) — check "
                    f"power/network, or press ▶ App.")
        if s.get("state") == "running":
            return "A run is already in progress on this camera."

        try:
            res = self.client.start_run(plot_name, plot_weight, save_frames)
        except CameraClientError as e:
            return f"Start failed: {e}"

        with self._lock:
            self.last_error = None
            self._attached_run = res.get("run_name", "")
            self._run_plot = plot
            self._run_plot_name = plot_name
            self._run_plot_weight = plot_weight
            self._run_meta = dict(meta or {})
        self._log(f"Run started: {self._attached_run}  |  Plot: {plot or '—'} "
                  f" |  Weight: {plot_weight + 'g' if plot_weight else '—'}  |  "
                  f"frames {'ON (saved on camera)' if save_frames else 'off'}")
        return None

    def stop(self) -> Optional[str]:
        """Stop the active run. Returns None if a stop was dispatched or an
        error string. The actual stop_run runs in a worker thread (frame
        saving can hold the response for a few seconds)."""
        with self._lock:
            if self.stopping:
                return "Already stopping."
            s = self.status
        if s is None:
            return (f"{self.label} camera is offline — the run keeps counting "
                    f"on-camera; reconnect to stop it.")
        if s.get("state") != "running":
            return "No active run on this camera."
        with self._lock:
            self.stopping = True
        self._log("Stopping run…")
        threading.Thread(target=self._do_stop, daemon=True).start()
        return None

    def _do_stop(self) -> None:
        try:
            res = self.client.stop_run()
        except CameraClientError as e:
            self._log(f"ERROR: stop failed: {e}")
            with self._lock:
                self.last_error = f"stop failed: {e}"
                self.stopping = False
            return
        try:
            self._finish_run(res)
        finally:
            with self._lock:
                self.stopping = False
                self._attached_run = ""

    def _finish_run(self, res: dict) -> None:
        """Write the returned CSV locally, add the session row, and clear
        the CSV from the camera (frames stay for Stored Runs…)."""
        run_name = res.get("run_name", "")
        n = int(res.get("n_pods", 0))
        elapsed = format_elapsed(res.get("elapsed_s", 0.0))
        csv_text = res.get("csv", "")

        # Sample name / weight are authoritative in the CSV itself (the
        # camera bakes them into plot_name/plot_weight at /run/start) —
        # read them back so adopted runs get correct metadata too.
        # Zero-count runs have no data rows: fall back to start-time values.
        plot_name, plot_weight = "", ""
        try:
            row = next(csv.DictReader(io.StringIO(csv_text)), None)
            if row:
                plot_name = (row.get("plot_name") or "").strip()
                plot_weight = (row.get("plot_weight") or "").strip()
        except Exception:  # noqa: BLE001
            pass
        with self._lock:
            plot_name = plot_name or self._run_plot_name or "unnamed"
            plot_weight = plot_weight or self._run_plot_weight
            plot = self._run_plot
            meta = dict(self._run_meta)

        csv_name = "(none)"
        csv_path: Optional[Path] = None
        if n > 0 and csv_text:
            out_dir = self.session.output_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            csv_name = f"{self.mode}_{safe_filename(plot_name)}_{ts}.csv"
            csv_path = out_dir / csv_name
            csv_path.write_text(csv_text, encoding="utf-8")
            self._log(f"Saved {n} tracked {self.label.lower()}s to:\n"
                      f"  {csv_path}")
        else:
            self._log(f"No {self.label.lower()}s tracked — nothing to save.")

        frames = int(res.get("frames_saved", 0))
        if frames:
            self._log(f"{frames} frames saved on the camera "
                      f"({int(res.get('frames_dropped', 0))} dropped) — "
                      f"fetch via Stored Runs….")

        row_n = self.session.add_run(
            mode_label=self.label, plot_name=plot_name, plot=plot,
            weight=plot_weight, count=n, elapsed=elapsed,
            csv_file=csv_name, csv_path=csv_path,
        )

        # The CSV is safe locally — clear it from the camera so Stored
        # Runs… only shows what still needs fetching. Frames (and the
        # config snapshot) stay on the camera. Best effort.
        if csv_path is not None and run_name:
            try:
                self.client.delete_csv(run_name)
            except CameraClientError:
                pass

        run_info = {
            "mode": self.mode, "label": self.label, "n": row_n,
            "plot_name": plot_name, "plot": plot, "weight": plot_weight,
            "count": n, "csv_path": csv_path, "meta": meta,
        }

        # Seed flow: the shelled sample is weighed AFTER the run — flag the
        # row so the UI pops the weight request automatically, and hold the
        # post-run hook (auto-append needs the weight) until it's answered.
        if self.weight_after_stop and not plot_weight:
            with self._lock:
                self.pending_weight = {"n": row_n, "plot": plot or plot_name,
                                       "count": n}
                self._deferred_run = run_info
            self._log(f"Weight pending for {plot or plot_name} — enter it "
                      f"in the popup (or choose 'No recorded weight').")
        else:
            self._fire_run_complete(run_info)

    def _fire_run_complete(self, run_info: dict) -> None:
        cb = self.on_run_complete
        if cb is None:
            return
        try:
            cb(self, run_info)
        except Exception as e:  # noqa: BLE001 — never kill the stop worker
            self._log(f"ERROR: post-run handler failed: {e}")

    def set_pending_weight(self, weight: Optional[str]) -> Optional[str]:
        """Answer the post-run weight request. None = 'no recorded weight'
        (row stays blank). Returns an error string for invalid input."""
        with self._lock:
            pending = self.pending_weight
        if pending is None:
            return None
        if weight is not None:
            weight = str(weight).strip()
            if not weight:
                return "Enter a weight, or choose 'No recorded weight'."
            try:
                if float(weight) <= 0:
                    return "Weight must be a positive number."
            except ValueError:
                return "Weight must be a number (grams)."
            self.session.set_weight(pending["n"], weight)
            self._log(f"Weight recorded for {pending['plot']}: {weight}g.")
        else:
            self._log(f"No weight recorded for {pending['plot']}.")
        with self._lock:
            self.pending_weight = None
            deferred = self._deferred_run
            self._deferred_run = None
        # The run is now fully complete (weight answered either way) —
        # release the post-run hook (auto-append) with the final weight.
        if deferred is not None:
            if weight is not None:
                deferred["weight"] = weight
            self._fire_run_complete(deferred)
        return None

    # ── app lifecycle (oakctl / HTTP) ────────────────────────────────────────
    def app_action(self, action: str) -> Optional[str]:
        """'start' | 'restart' | 'stop' the on-camera app container.
        Returns None if dispatched, else an error string."""
        with self._lock:
            if self.lifecycle_busy:
                return f"Busy: {self.lifecycle_busy}."
            s = self.status
        if action == "stop" and s is not None and s.get("state") == "running":
            return "A run is active — stop the run before stopping the app."
        with self._lock:
            self.lifecycle_busy = f"{action}ing app"
        self._log(f"App {action} requested…")

        def _do():
            try:
                if action == "start":
                    oakctl_tools.app_start(self.ip)
                elif action == "restart":
                    # Prefer the app's own re-exec (fast, applies a pushed
                    # config); fall back to the device agent when it's down.
                    try:
                        self.client.restart_app()
                    except CameraClientError:
                        oakctl_tools.app_restart(self.ip)
                elif action == "stop":
                    try:
                        self.client.shutdown_app()
                    except CameraClientError:
                        oakctl_tools.app_stop(self.ip)
                self._log(f"App {action} done (allow ~20 s for the pipeline "
                          f"to settle).")
            except (oakctl_tools.OakctlError, CameraClientError) as e:
                self._log(f"App {action} failed: {e}")
            finally:
                with self._lock:
                    self.lifecycle_busy = None

        threading.Thread(target=_do, daemon=True).start()
        return None

    # ── stored runs (frames/CSVs on the camera) ──────────────────────────────
    def runs_download_dir(self) -> Path:
        return self.session.output_dir / "camera_runs" / self.mode

    def list_runs(self) -> list:
        return self.client.list_runs()

    def download_run(self, name: str) -> Path:
        dest = self.runs_download_dir()
        dest.mkdir(parents=True, exist_ok=True)
        return self.client.download_run(name, dest)

    def download_run_async(self, name: str,
                           total_bytes: int = 0) -> Optional[str]:
        """Queue a stored-run download on the per-camera worker thread.
        Returns an error string only when that run is already in flight.
        Progress lands in snapshot()['downloads'] for the UI to render."""
        with self._lock:
            d = self.downloads.get(name)
            if d and d["state"] in ("queued", "downloading"):
                return f"{name} is already downloading."
            self.downloads[name] = {"name": name, "state": "queued",
                                    "done": 0, "total": int(total_bytes or 0),
                                    "dest": "", "error": ""}
            if self._dl_thread is None or not self._dl_thread.is_alive():
                self._dl_thread = threading.Thread(
                    target=self._download_worker, daemon=True,
                    name=f"dl-{self.mode}")
                self._dl_thread.start()
        return None

    def _download_worker(self) -> None:
        while True:
            with self._lock:
                item = next((d for d in self.downloads.values()
                             if d["state"] == "queued"), None)
                if item is None:
                    return
                item["state"] = "downloading"
            name = item["name"]
            self._log(f"Downloading {name}…")

            def _cb(done: int, _item=item):
                _item["done"] = done          # GIL-safe int store

            try:
                dest_dir = self.runs_download_dir()
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = self.client.download_run(name, dest_dir,
                                                progress_cb=_cb)
                with self._lock:
                    item["state"] = "done"
                    item["dest"] = str(dest)
                self._log(f"✓ Downloaded {name} → {dest}")
            except CameraClientError as e:
                with self._lock:
                    item["state"] = "failed"
                    item["error"] = str(e)
                self._log(f"ERROR: download of {name} failed: {e}")

    def clear_finished_downloads(self) -> None:
        with self._lock:
            self.downloads = {k: v for k, v in self.downloads.items()
                              if v["state"] in ("queued", "downloading")}

    def delete_run(self, name: str) -> None:
        self.client.delete_run(name)

    # ── internals ────────────────────────────────────────────────────────────
    def _log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        with self._lock:
            self.logs.append(f"[{stamp}] {text}")

    def setup_url(self) -> str:
        return f"http://{self.ip}:{self.port}/setup"

    # ── snapshot for the UI poller ───────────────────────────────────────────
    def snapshot(self) -> dict:
        with self._lock:
            s = dict(self.status) if self.status is not None else None
            d = {
                "mode": self.mode,
                "label": self.label,
                "ip": self.ip,
                "online": s is not None,
                "stopping": self.stopping,
                "lifecycle_busy": self.lifecycle_busy,
                "pending_weight": (dict(self.pending_weight)
                                   if self.pending_weight else None),
                "downloads": [dict(v) for v in self.downloads.values()],
                "log": "\n".join(self.logs),
                "error": self.last_error,
            }
        if s is None:
            d.update({
                "state": "offline", "running": False, "count": 0, "active": 0,
                "fps": 0.0, "elapsed": "00:00", "plot_name": "",
                "app_version": "", "staged_mode": "", "staged_at": "",
                "config_source": "", "pending_restart": False,
                "save_frames": False, "frames_saved": 0,
            })
            return d
        state = s.get("state", "idle")
        d.update({
            "state": state,
            "running": state == "running",
            "count": int(s.get("unique", 0)),
            "active": int(s.get("active", 0)),
            "fps": float(s.get("fps", 0.0)),
            "elapsed": format_elapsed(s.get("elapsed_s", 0.0)),
            "plot_name": s.get("plot_name", ""),
            "app_version": s.get("app_version", ""),
            "staged_mode": s.get("staged_mode", ""),
            "staged_at": s.get("staged_at", ""),
            "config_source": s.get("config_source", ""),
            # A pushed config overlay awaits an app restart to take effect.
            "pending_restart": bool(s.get("pending_fingerprint")),
            "save_frames": bool(s.get("save_frames", False)),
            "frames_saved": int(s.get("frames_saved", 0)),
        })
        return d


def build_controllers(cfg: dict) -> dict[str, CameraController]:
    """One controller per configured mode (config order), each with its
    own SessionLog — pod and seed sessions are independent."""
    output_dir = cfg.get("output_dir", "./output")
    out: dict[str, CameraController] = {}
    for mode, mc in (cfg.get("modes") or {}).items():
        out[mode] = CameraController(
            mode=mode,
            label=mc.get("label", mode.title()),
            ip=mc.get("device_ip", ""),
            port=int(mc.get("http_port", DEFAULT_PORT)),
            session=SessionLog(output_dir, mode),
            weight_after_stop=bool(mc.get("weight_after_stop", False)),
        )
    return out
