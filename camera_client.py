"""
HTTP client for the on-camera pod-counter service (port 8808).

VENDORED COPY — single source of truth is the Oak 4 App repository:
    "Oak 4 App/gui/camera_client.py"
Do not fix bugs here first: fix them upstream, then re-copy. This copy
drops the upstream Qt thread helpers (StatusPoller / ClientJob) — the
Dash app polls from a plain daemon thread instead — so this file is
stdlib-only. Vendored 2026-07-21 against on-camera app API v0.5.0.

The camera app listens on http://<device_ip>:<control_port> (default
8808) — see the Oak 4 App's oakapp/pod-counter/main.py for the API.
"""

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

DEFAULT_PORT = 8808
CONTROL_TIMEOUT_S = 4.0


class CameraClientError(Exception):
    """A control-API call failed (network error or API error response)."""


class CameraClient:
    """Synchronous client for one camera's control API."""

    def __init__(self, ip: str, port: int = DEFAULT_PORT):
        self.base = f"http://{ip}:{port}"

    # ── plumbing ──────────────────────────────────────────────────────
    def _request(self, path: str, method: str = "GET",
                 body: Optional[dict] = None,
                 timeout: float = CONTROL_TIMEOUT_S) -> dict | list:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read().decode("utf-8")).get("error", "")
            except Exception:
                msg = ""
            raise CameraClientError(msg or f"HTTP {e.code}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise CameraClientError(f"unreachable: {e}") from e

    # ── API calls ─────────────────────────────────────────────────────
    def status(self) -> dict:
        return self._request("/status")

    def start_run(self, plot_name: str, plot_weight: str,
                  save_frames: bool) -> dict:
        return self._request("/run/start", "POST", {
            "plot_name": plot_name,
            "plot_weight": plot_weight,
            "save_frames": save_frames,
        })

    def stop_run(self) -> dict:
        # Normally sub-second; when frame saving is on, the camera waits
        # (≤15 s) for queued JPEG writes before returning. 30 s is headroom.
        return self._request("/run/stop", "POST", {}, timeout=30.0)

    def list_runs(self) -> list:
        return self._request("/runs", timeout=15.0)

    def get_config(self) -> dict:
        return self._request("/config", timeout=10.0)

    def put_config(self, cfg: dict) -> dict:
        """Push a config overlay to the camera (validated device-side)."""
        return self._request("/config", "PUT", cfg, timeout=15.0)

    def restart_app(self) -> dict:
        """Ask the app to re-exec in place (applies a pushed config)."""
        return self._request("/restart", "POST", {}, timeout=10.0)

    def shutdown_app(self) -> dict:
        """Ask the app to exit (container stops; restart via oakctl)."""
        return self._request("/shutdown", "POST", {}, timeout=6.0)

    def delete_run(self, name: str) -> dict:
        return self._request(f"/runs/{name}", "DELETE")

    def delete_csv(self, name: str) -> dict:
        return self._request(f"/runs/{name}/csv", "DELETE")

    def download_run(self, name: str, dest_dir: str | Path,
                     progress_cb: Optional[Callable[[int], None]] = None,
                     ) -> Path:
        """Stream /runs/<name>.tar.gz to dest_dir. Returns the file path."""
        dest = Path(dest_dir) / f"{name}.tar.gz"
        url = f"{self.base}/runs/{name}.tar.gz"
        try:
            with urllib.request.urlopen(url, timeout=60.0) as resp, \
                    open(dest, "wb") as f:
                done = 0
                while True:
                    chunk = resp.read(256 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if progress_cb:
                        progress_cb(done)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            dest.unlink(missing_ok=True)
            raise CameraClientError(f"download failed: {e}") from e
        return dest
