"""
oakctl wrappers for on-camera app lifecycle control.

VENDORED COPY — single source of truth is the Oak 4 App repository:
    "Oak 4 App/gui/oakctl_tools.py"
Do not fix bugs here first: fix them upstream, then re-copy. This copy
drops the upstream deploy() helper — deploying/staging the on-camera app
stays in the Oak 4 App repo; Grading Dash only starts/stops/restarts an
already-deployed app. Vendored 2026-07-21.

Apps deployed from source with `oakctl app run` get the fixed "develop"
app ID (all zeros), so start/stop/restart can target it directly without
parsing `oakctl app list`. These calls talk to the device's oak-agent —
they work even when the app's own HTTP API is down, which is exactly
when you need them.

All functions are synchronous and raise OakctlError on failure; call
them from a worker thread so the web server never blocks.
"""

import shutil
import subprocess
import sys
from pathlib import Path

# `oakctl app run` registers the source app under this well-known ID.
DEV_APP_ID = "00000000-0000-0000-0000-000000000000"

_WIN_DEFAULT = r"C:\Program Files\oakctl\bin\oakctl.exe"


class OakctlError(Exception):
    """An oakctl invocation failed; str() carries the command output."""


def find_oakctl() -> str | None:
    """Resolve the oakctl binary (PATH first, then the Windows default)."""
    path = shutil.which("oakctl")
    if path:
        return path
    if Path(_WIN_DEFAULT).exists():
        return _WIN_DEFAULT
    return None


def _run(args: list[str], cwd: Path | None = None,
         timeout: float = 120.0) -> str:
    """Run a command; return stdout+stderr text, raise OakctlError on
    failure. stdin is closed so an unexpected interactive prompt (e.g.
    a non-default device password) fails fast instead of hanging."""
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        proc = subprocess.run(
            args, cwd=str(cwd) if cwd else None,
            stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=timeout,
            **kwargs,
        )
    except FileNotFoundError as e:
        raise OakctlError(f"command not found: {args[0]}") from e
    except subprocess.TimeoutExpired as e:
        raise OakctlError(
            f"timed out after {timeout:.0f}s: {' '.join(args[:4])}…"
        ) from e
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        tail = "\n".join(out.strip().splitlines()[-8:])
        raise OakctlError(
            f"exit {proc.returncode}: {' '.join(args[:4])}…\n{tail}"
        )
    return out


def _oakctl(ip: str, *args: str, timeout: float = 120.0) -> str:
    exe = find_oakctl()
    if exe is None:
        raise OakctlError(
            "oakctl not found — install it or add it to PATH "
            "(https://docs.luxonis.com)."
        )
    return _run([exe, *args, "-d", ip], timeout=timeout)


def app_start(ip: str) -> str:
    """Start the (already deployed) pod-counter app."""
    return _oakctl(ip, "app", "start", DEV_APP_ID)


def app_stop(ip: str) -> str:
    """Stop the pod-counter app container."""
    return _oakctl(ip, "app", "stop", DEV_APP_ID)


def app_restart(ip: str) -> str:
    """Restart the pod-counter app (same build, config unchanged)."""
    return _oakctl(ip, "app", "restart", DEV_APP_ID)


def app_logs(ip: str, tail: int = 40) -> str:
    return _oakctl(ip, "app", "logs", DEV_APP_ID,
                   "--tail", str(tail), "--no-follow")
