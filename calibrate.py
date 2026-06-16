#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate.py
NC State University - Peanut Breeding & Genetics
============================================================================
Standalone launcher for the two PyQt calibration dialogs, spawned as a
SEPARATE PROCESS by PBG_grading.py.

Why a separate process?
  The Dash web server runs headless (QCoreApplication) and serves requests
  on worker threads.  A PyQt dialog (QDialog) needs a full QApplication and
  must run on the main thread.  Running it in its own short-lived process
  sidesteps that entirely: this script owns its QApplication, opens one
  dialog modally, and exits when the operator closes it.

Both dialogs persist the active mode's block back into config.yaml on save
(via yaml.safe_dump in their _on_save) — this launcher does not write the
file itself.

Must run on the machine wired to the OAK camera (needs a display).  Run it
from the repo root so `gui.*` / `pipeline.*` import correctly.

Usage:
    python calibrate.py --step initial --mode pod  [--config config.yaml]
    python calibrate.py --step target  --mode seed [--config config.yaml]
"""

import argparse
import sys
from pathlib import Path

import yaml
from PyQt6.QtWidgets import QApplication


def _bring_to_front(dlg) -> None:
    """Nudge a freshly-shown dialog to the foreground.

    When the Dash server spawns this script as a subprocess, the dialog can
    open behind the browser. show()/raise_()/activateWindow() ask the window
    manager to bring it forward. This deliberately does NOT touch
    setWindowFlags(): changing flags recreates the native window, which races
    with the live camera worker thread the dialog has already started and
    aborts the process.
    """
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()


def main() -> int:
    ap = argparse.ArgumentParser(description="PBG calibration launcher")
    ap.add_argument("--step", required=True, choices=["initial", "target"],
                    help="initial = mount/IMU/focus; target = px-mm + colour")
    ap.add_argument("--mode", required=True, help="config mode key, e.g. pod / seed")
    ap.add_argument("--config", default="config.yaml", help="path to config.yaml")
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 2

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    modes = cfg.get("modes", {})
    if args.mode not in modes:
        print(f"Mode '{args.mode}' not in {config_path}; available: {sorted(modes)}")
        return 2
    mode_cfg = modes[args.mode]

    app = QApplication(sys.argv)

    if args.step == "initial":
        # Initial Camera Setup — IMU mount alignment + manual focus tuning.
        from gui.mount_dialog import MountSetupDialog
        dlg = MountSetupDialog(mode_cfg, cfg, config_path)
        dlg.mount_saved.connect(
            lambda: print(f"Initial setup saved to {config_path}"))
    else:
        # Target Calibration — px/mm scale + colour-correction matrix.
        from gui.setup_dialog import SetupDialog
        dlg = SetupDialog(mode_cfg, cfg, config_path)
        dlg.calibration_saved.connect(
            lambda *_: print(f"Target calibration saved to {config_path}"))

    _bring_to_front(dlg)
    dlg.exec()   # modal; dialog writes config.yaml itself on Save
    return 0


if __name__ == "__main__":
    sys.exit(main())
