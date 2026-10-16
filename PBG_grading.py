#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PBG_grading.py
NC State University - Peanut Breeding & Genetics
============================================================================
Web (Dash) front-end for the OAK 4 Pro D pod / seed grading cameras.

Each camera runs the on-camera "pod-counter" application (deployed from
the Oak 4 App repository) which owns the whole DepthAI pipeline and
exposes an HTTP control API on port 8808. This app is a pure front-end
built for GRADER simplicity:

    Grading mode (always the default)
        Scan the bag's QR tag → the sample name, plot, and the Dropbox
        trial file are resolved automatically. Start is blocked unless
        the plot exists in the trial file, has no data for this camera
        yet, and (seed side) already has pod data — you can't shell a
        sample that was never pod-graded. On Stop, the run's summary is
        appended straight into the trial workbook.

    Test mode
        Minimal checks: any sample name, no trial/plot required, nothing
        is appended anywhere. Raw CSVs and a session summary can be
        downloaded locally.

    Admin (button top-right, user/pass admin/admin — a deterrent, not
    security)
        Camera details/logs, on-camera setup pages, stored-runs browser,
        app power controls, and the manual Dropbox export/append tools.

QR tags encode Year|Location|NC_Accession|Trial|Plot. The trial workbook
is found automatically at
    /<Year> Peanut Program/Field Program/<Tests|Increases>/<Location> <Trial>.xlsx

Run this on a machine that can reach the cameras' HTTP port (8808):

    python PBG_grading.py            # then open http://127.0.0.1:8050

Deploying/updating the on-camera app itself stays in the Oak 4 App
repository. The camera API is unauthenticated HTTP — keep the cameras on
the lab network.
"""

import base64
import io
import os
import threading
import time
import webbrowser
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

import dash
from dash import dcc, html, dash_table, Input, Output, State, ALL
import dash_bootstrap_components as dbc

from dotenv import load_dotenv
load_dotenv()

from camera_client import CameraClientError
from dropbox_team_client import TeamDropboxClientManager
from grading_backend import build_controllers, safe_filename
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
OUTPUT_DIR = Path(CFG.get("output_dir", "./output"))

# Admin gate — a deterrent so graders stay out of the tools, not security.
ADMIN_USER = "admin"
ADMIN_PASS = "admin"

# Dropbox team-space root ("" = the NC State PB&G team-space top level).
DROPBOX_ROOT = ""
DROPBOX_DEFAULT_DEST = "/NC State PB&G Dropbox"

# Trial workbooks live here, split across these subfolders.
FIELD_PROGRAM_SUBS = ("Tests", "Increases")

# Team client is optional at boot — the grading console works without it;
# trial lookups/appends and the Dropbox export need it.
try:
    team_manager: Optional[TeamDropboxClientManager] = TeamDropboxClientManager()
except Exception as e:  # noqa: BLE001
    print(f"Dropbox client unavailable (trial append disabled): {e}")
    team_manager = None


# ═════════════════════════════════════════════════════════════════════════════
# BACKEND  — one controller per camera, per-camera session logs
# ═════════════════════════════════════════════════════════════════════════════

CONTROLLERS = build_controllers(CFG)
MODES = list(CONTROLLERS)          # config order, e.g. ["pod", "seed"]
SESSIONS = {m: CONTROLLERS[m].session for m in MODES}


def any_running() -> bool:
    return any(c.snapshot()["running"] for c in CONTROLLERS.values())


def combined_runs() -> list:
    """All completed runs across both cameras' current sessions."""
    runs = []
    for m in MODES:
        runs.extend(SESSIONS[m].snapshot()["session"])
    return runs


# ═════════════════════════════════════════════════════════════════════════════
# TRIAL-FILE RESOLUTION + PRE-START CHECKS + AUTO-APPEND
# ═════════════════════════════════════════════════════════════════════════════

def parse_sample_tokens(name: str) -> Optional[dict]:
    """Split a QR sample name Year_Location_Accession_Trial_Plot."""
    parts = [p.strip() for p in str(name or "").strip().split("_")]
    if len(parts) < 5 or not all(parts[:5]):
        return None
    return {"year": parts[0], "loc": parts[1], "acc": parts[2],
            "trial": parts[3], "plot": parts[4]}


_TRIAL_CACHE: dict = {}          # (year, loc, trial) -> (path, found_at)
_TRIAL_CACHE_TTL_S = 600


def resolve_trial_file(year: str, loc: str, trial: str):
    """Find '<loc> <trial>.xlsx' under /<year> Peanut Program/Field Program/
    (Tests|Increases). Returns (dropbox_path, None) or (None, error)."""
    if team_manager is None:
        return None, "Dropbox is not configured on this station (.env)."
    key = (year, loc.lower(), trial.lower())
    hit = _TRIAL_CACHE.get(key)
    if hit and time.time() - hit[1] < _TRIAL_CACHE_TTL_S:
        return hit[0], None
    target = f"{loc} {trial}.xlsx".lower()
    for sub in FIELD_PROGRAM_SUBS:
        folder = f"/{year} Peanut Program/Field Program/{sub}"
        try:
            items = team_manager.get_folder_contents(folder)
        except Exception:  # noqa: BLE001 — folder may not exist for this year
            continue
        for it in items:
            if it.get("type") == "file" and it["name"].lower() == target:
                _TRIAL_CACHE[key] = (it["path"], time.time())
                return it["path"], None
    return None, (f"No trial file '{loc} {trial}.xlsx' found under "
                  f"/{year} Peanut Program/Field Program/"
                  f"({'|'.join(FIELD_PROGRAM_SUBS)}). Check the tag or add "
                  f"the trial file.")


def get_plot_status(trial_path: str, plot: str) -> dict:
    """Download the trial workbook and report the plot's grading state:
    {exists, has_pod, has_seed}. Raises on unreadable files."""
    blob = team_manager.get_file(trial_path)
    if not blob:
        raise ValueError(f"could not download {trial_path}")
    sheets = pd.read_excel(io.BytesIO(blob), sheet_name=None)
    sheet = "Trial" if "Trial" in sheets else next(iter(sheets))
    df = sheets[sheet]
    if "Plot" not in df.columns:
        raise ValueError(f"sheet '{sheet}' has no Plot column")
    mask = pd.to_numeric(df["Plot"], errors="coerce") == int(plot)
    if not mask.any():
        return {"exists": False, "has_pod": False, "has_seed": False}

    def _filled(col: str) -> bool:
        if col not in df.columns:
            return False
        vals = df.loc[mask, col].dropna()
        return any(str(v).strip() not in ("", "nan") for v in vals)

    return {"exists": True,
            "has_pod": _filled("Pod Count"),
            "has_seed": _filled("Seed Count")}


def verify_before_start(mode_key: str, name: str, plot: str):
    """Grading-mode gate. Returns (trial_path, None) when the run may
    start, else (None, error message). Enforces:
      * the QR name resolves to a real trial workbook;
      * the plot exists in it;
      * this camera has no data for the plot yet (no silent overwrites);
      * seed runs require existing pod data (shelling is irreversible)."""
    tok = parse_sample_tokens(name)
    if tok is None:
        return None, ("Sample Name must be Year_Location_Accession_Trial_"
                      "Plot — scan the bag's QR tag.")
    if not str(plot).strip().isdigit():
        return None, "Plot must be a whole number — scan the bag's QR tag."
    trial_path, err = resolve_trial_file(tok["year"], tok["loc"], tok["trial"])
    if err:
        return None, err
    try:
        st = get_plot_status(trial_path, plot)
    except Exception as e:  # noqa: BLE001
        return None, f"Could not check {Path(trial_path).name}: {e}"
    fname = Path(trial_path).name
    if not st["exists"]:
        return None, f"Plot {plot} is not in {fname} — check the tag."
    # If a re-grade IS intentional, the escape hatch is: run the sample in
    # Test Mode with the same numeric plot, then Admin → Append All Session
    # Runs to the trial file (new values replace old).
    regrade = (f"If re-grading it is intentional: run it in Test Mode with "
               f"plot {plot}, then Admin → Append All Session Runs.")
    if mode_key == "pod":
        if st["has_pod"]:
            return None, (f"Plot {plot} already has POD data in {fname} — "
                          f"grading it again would overwrite. {regrade}")
    else:
        if st["has_seed"]:
            return None, (f"Plot {plot} already has SEED data in {fname} — "
                          f"grading it again would overwrite. {regrade}")
        if not st["has_pod"]:
            return None, (f"Plot {plot} has NO pod data in {fname} yet — "
                          f"grade the pods FIRST. Shelling without pod data "
                          f"is unrecoverable.")
    return trial_path, None


# One append at a time — both cameras can stop near-simultaneously and the
# trial file update is a read-modify-write.
_APPEND_LOCK = threading.Lock()


def _auto_append(ctl, info: dict) -> None:
    """on_run_complete hook: append the finished run's summary into the
    trial workbook resolved at start. Runs in the stop worker thread."""
    trial_path = (info.get("meta") or {}).get("trial_path")
    if not trial_path:
        return                       # test mode / adopted run — no append
    n, plot = info["n"], info["plot"]
    session = SESSIONS[ctl.mode]
    if info["count"] <= 0 or info["csv_path"] is None:
        session.set_append_status(n, "skipped")
        ctl._log(f"Auto-append skipped for plot {plot} — nothing counted.")
        return
    if team_manager is None:
        session.set_append_status(n, "FAILED")
        ctl._log("ERROR: auto-append failed — Dropbox not configured.")
        return
    try:
        row = compute_summary(info["csv_path"], mode=ctl.label,
                              plot_weight=info["weight"], plot=plot)
        with _APPEND_LOCK:
            blob = team_manager.get_file(trial_path)
            if not blob:
                raise RuntimeError(f"could not download {trial_path}")
            new_blob = merge_summary_into_workbook(blob, row)
            team_manager.upload_file(new_blob, trial_path, mode="overwrite")
        session.set_append_status(n, "✓")
        ctl._log(f"✓ Auto-appended plot {plot} "
                 f"({info['count']} {ctl.label.lower()}s) to "
                 f"{Path(trial_path).name}.")
    except Exception as e:  # noqa: BLE001
        session.set_append_status(n, "FAILED")
        ctl._log(f"ERROR: auto-append for plot {plot} failed: {e} — the "
                 f"data is safe in {info['csv_path']}; retry from the "
                 f"Admin export tools.")


# ── Pod → seed handoff ───────────────────────────────────────────────────────
# In practice a plot is graded on the pod camera and then goes straight to
# the seed camera, so the seed panel pre-fills itself from the pod run that
# just finished. Published only AFTER the trial append has run, so a failed
# append can never be mistaken for a seed-side problem. The operator can
# still overwrite either field (or scan a different tag) before starting.
POD_MODE = "pod" if "pod" in MODES else None
SEED_MODE = "seed" if "seed" in MODES else None
_HANDOFF: dict = {"seq": 0, "name": "", "plot": ""}
_HANDOFF_LOCK = threading.Lock()


def _on_run_complete(ctl, info: dict) -> None:
    _auto_append(ctl, info)              # contains its own error handling
    if ctl.mode != POD_MODE or SEED_MODE is None:
        return
    if not info.get("plot_name"):
        return
    with _HANDOFF_LOCK:
        _HANDOFF["seq"] += 1
        _HANDOFF["name"] = info.get("plot_name", "")
        _HANDOFF["plot"] = info.get("plot", "")
    ctl._log(f"Handed {info.get('plot') or info.get('plot_name')} to the "
             f"{SEED_MODE} panel.")


for _c in CONTROLLERS.values():
    _c.on_run_complete = _on_run_complete
    _c.start_polling()


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
BTN_DARK = {"backgroundColor": "#3b4045", "border": "none", "color": "white",
            "fontWeight": "bold"}
BTN_GREEN = {"backgroundColor": "#2f9e44", "border": "none", "color": "white",
             "fontWeight": "bold"}
BTN_RED = {"backgroundColor": "#CC0000", "border": "none", "color": "white",
           "fontWeight": "bold"}
LABEL_STYLE = {"fontWeight": "bold", "fontSize": "12px", "marginBottom": "4px",
               "color": "#333"}
LIGHT_BASE = {"display": "inline-block", "width": "22px", "height": "22px",
              "borderRadius": "50%", "verticalAlign": "middle",
              "border": "2px solid rgba(255,255,255,0.7)"}

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


# ═════════════════════════════════════════════════════════════════════════════
# CAMERA PANEL  — grader-simple: light, inputs, start/stop, stats, restart
# ═════════════════════════════════════════════════════════════════════════════

def build_camera_panel(mode: str) -> dbc.Col:
    ctl = CONTROLLERS[mode]
    if ctl.weight_after_stop:
        weight_col = dbc.Col([
            html.Div("Weight (g)", style=LABEL_STYLE),
            dbc.Input(id=f"weight-{mode}", size="sm", type="number",
                      min=0, style={"display": "none"}),
            html.Div("requested after Stop (shelled weight)",
                     className="text-muted", style={"fontSize": "11px",
                                                    "paddingTop": "6px"}),
        ], md=5)
    else:
        weight_col = dbc.Col([
            html.Div(["Weight (g)",
                      html.Span(" *", style={"color": "#CC0000"})],
                     style=LABEL_STYLE),
            dbc.Input(id=f"weight-{mode}", size="sm", type="number",
                      min=0, placeholder="e.g. 500"),
        ], md=5)

    header = html.Div([
        html.Span(f"{ctl.label} Camera"),
        html.Span(id=f"light-{mode}", title="Green = ready, red = not ready",
                  style={**LIGHT_BASE, "backgroundColor": "#888",
                         "float": "right"}),
        # Lighting monitor pill (camera 0.7.2+): quiet when ok, amber when
        # low/high; empty on older cameras.
        html.Span(id=f"lightpill-{mode}", className="float-end me-2",
                  title="Lighting monitor: median scene brightness reported "
                        "by the camera (expected 60–90)"),
    ])

    lowlight_modal = dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("⚠ Warning: low light")),
        dbc.ModalBody([
            html.Div(["Lighting detection is reporting ",
                      html.B("low light conditions"), " on the ",
                      html.B(f"{ctl.label.lower()} camera"),
                      " — are the lights on?"], className="mb-2"),
            html.Div(id=f"ll-reading-{mode}", className="fw-bold mb-2"),
            html.Div(["Low light blacks out colour and inflates sizes; "
                      "counting still works. Press ", html.B("Continue"),
                      " if this is intentional, or after turning the lights "
                      "on (this reading updates live — give it up to 30 s)."],
                     className="text-muted small"),
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id=f"ll-cancel-{mode}", size="sm",
                       style=BTN_DARK, n_clicks=0),
            dbc.Button("Continue", id=f"ll-continue-{mode}", size="sm",
                       style=BTN_GREEN, n_clicks=0),
        ]),
    ], id=f"ll-modal-{mode}", is_open=False)

    return dbc.Col([
        lowlight_modal,
        dbc.Card([
            dbc.CardHeader(header, style=HEADER_STYLE),
            dbc.CardBody([
                # QR scan — fills Sample Name + Plot and (in Grading mode)
                # verifies the trial file / plot state right away.
                dbc.Row([
                    dbc.Col(html.Span("QR Scan:", style=LABEL_STYLE),
                            width="auto", className="pt-1"),
                    dbc.Col(dbc.Input(
                        id=f"qr-{mode}", size="sm", debounce=True,
                        autocomplete="off",
                        placeholder="Click here, then scan tag…")),
                ], className="g-2 mb-1 align-items-center"),
                html.Div(id=f"qr-result-{mode}", className="small mb-2",
                         style={"minHeight": "1em"}),
                dbc.Row([
                    dbc.Col([
                        html.Div(["Sample Name",
                                  html.Span(" *", style={"color": "#CC0000"})],
                                 style=LABEL_STYLE),
                        dbc.Input(id=f"name-{mode}", size="sm",
                                  placeholder="scan the QR tag"),
                    ], md=4),
                    dbc.Col([
                        html.Div(["Plot",
                                  html.Span(" *", style={"color": "#CC0000"})],
                                 style=LABEL_STYLE),
                        dbc.Input(id=f"plot-{mode}", size="sm",
                                  placeholder="auto"),
                    ], md=3),
                    weight_col,
                ], className="g-2 mb-2"),

                # On by default: an ungraded run can be re-analysed later
                # only if its frames were kept, and the camera now has a
                # 30-day / 50 GB retention window to hold them.
                html.Div(dbc.Checkbox(
                    id=f"saveframes-{mode}", value=True,
                    label="Save frames for this run (stored on the camera "
                          "— fetch via Admin → Stored Runs…)"),
                    id=f"saveframes-wrap-{mode}", className="mb-2"),

                dbc.Row([
                    dbc.Col(dbc.Button([html.I(className="bi bi-play-fill me-1"),
                                        "Start"], id=f"start-{mode}",
                                       style=BTN_GREEN, className="w-100"),
                            width=6),
                    dbc.Col(dbc.Button([html.I(className="bi bi-stop-fill me-1"),
                                        "Stop"], id=f"stop-{mode}",
                                       style=BTN_RED, className="w-100",
                                       disabled=True), width=6),
                ], className="g-2 mb-2"),
                # "material moving but no run started" (camera 0.7.0+) and
                # low/high light (0.7.2+) — informational, never block a run.
                html.Div(id=f"idlewarn-{mode}"),
                html.Div(id=f"lightwarn-{mode}"),
                html.Div(id=f"alert-{mode}", className="mb-1"),

                dbc.Row([
                    dbc.Col(html.Div([
                        html.Div("0", id=f"count-{mode}",
                                 style={"fontSize": "40px", "fontWeight": "bold",
                                        "textAlign": "center",
                                        "fontFamily": "monospace"}),
                        html.Div(f"Unique {ctl.label}s",
                                 className="text-muted small text-center"),
                    ]), width=4),
                    dbc.Col(html.Div([
                        html.Div("00:00", id=f"elapsed-{mode}",
                                 style={"fontSize": "30px",
                                        "textAlign": "center",
                                        "fontFamily": "monospace"}),
                        html.Div("Elapsed",
                                 className="text-muted small text-center"),
                    ]), width=4),
                    dbc.Col(html.Div([
                        html.Div("-- fps", id=f"fps-{mode}",
                                 style={"fontSize": "24px",
                                        "textAlign": "center",
                                        "fontFamily": "monospace"}),
                        html.Div("Active: --", id=f"active-{mode}",
                                 className="text-muted small text-center"),
                    ]), width=4),
                ], className="align-items-center mb-2"),

                dbc.Row([
                    dbc.Col(dbc.Button("⟲ Restart Camera App",
                                       id=f"app-restart-{mode}", size="sm",
                                       style=BTN_DARK, className="w-100"),
                            md=6),
                    dbc.Col(html.Div("If the light is red, try a restart "
                                     "(takes ~20 s).",
                                     className="text-muted",
                                     style={"fontSize": "11px",
                                            "paddingTop": "6px"}), md=6),
                ], className="g-2"),
            ], style={"padding": "12px"}),
        ], style=CARD_STYLE),

        # Per-camera session history (separate pod/seed sessions).
        _card(f"{ctl.label} Session History", [
            html.Div(id=f"session-current-{mode}",
                     className="small fw-bold mb-1"),
            html.Div("Select one or more rows to download those runs "
                     "(Test Mode / Admin); none selected = the latest run.",
                     className="text-muted mb-1", style={"fontSize": "11px"}),
            dash_table.DataTable(
                id=f"session-table-{mode}",
                columns=[{"name": c, "id": i} for c, i in [
                    ("#", "n"), ("Sample", "plot_name"), ("Plot", "plot"),
                    ("Weight", "weight"), ("Count", "count"),
                    ("Duration", "elapsed"), ("Appended", "appended")]],
                data=[], row_selectable="multi", selected_rows=[],
                page_size=8,
                style_table={"overflowX": "auto"},
                style_cell={"fontSize": "11px", "padding": "5px",
                            "textAlign": "center"},
                style_header={"backgroundColor": "#CC0000", "color": "white",
                              "fontWeight": "bold"},
                style_data_conditional=[
                    {"if": {"row_index": "odd"},
                     "backgroundColor": "#f9f9f9"},
                    {"if": {"filter_query": "{appended} = 'FAILED'",
                            "column_id": "appended"},
                     "color": "#CC0000", "fontWeight": "bold"},
                ],
            ),
            html.Hr(className="my-2"),
            dbc.Row([
                dbc.Col(dcc.Dropdown(id=f"session-select-{mode}", options=[],
                                     placeholder="Previous sessions…",
                                     style={"fontSize": "12px"}), md=7),
                dbc.Col(dbc.Button("Load", id=f"btn-load-session-{mode}",
                                   style=BTN_DARK, size="sm",
                                   className="w-100"), md=3),
                dbc.Col(dbc.Button(html.I(className="bi bi-arrow-clockwise"),
                                   id=f"btn-refresh-sessions-{mode}",
                                   size="sm", style=BTN_DARK,
                                   className="w-100",
                                   title="Refresh session list"), md=2),
            ], className="g-2 align-items-center"),
            html.Div(id=f"session-load-result-{mode}", className="small mt-1"),
        ]),

        # Stored-runs modal (opened from the Admin tools).
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle(
                f"Stored Runs — {ctl.label} Camera ({ctl.ip})")),
            dbc.ModalBody([
                html.Div("Runs live on the camera's /data partition. "
                         "Download fetches a .tar.gz (frames + CSV) into "
                         f"output/camera_runs/{mode}/ in the background — "
                         "progress shows below and you can close this "
                         "window meanwhile. Delete removes the run from "
                         "the camera permanently.",
                         className="text-muted small mb-2"),
                html.Div(id=f"runs-progress-{mode}", className="mb-2"),
                html.Div(id=f"runs-result-{mode}", className="small mb-2"),
                html.Div(id=f"runs-list-{mode}", className="small mb-1"),
                # Permanent (initially empty) so callbacks can reference it
                # before the modal has ever been opened — a dynamically
                # created component here breaks the whole callback chain.
                html.Div(dbc.Checklist(
                    id=f"runs-select-{mode}", options=[], value=[],
                    inputStyle={"marginRight": "8px"},
                    labelStyle={"fontFamily": "monospace",
                                "fontSize": "12px", "display": "block",
                                "padding": "3px 0",
                                "borderBottom": "1px solid #f0f0f0"}),
                    style={"maxHeight": "380px", "overflowY": "auto"}),
            ]),
            dbc.ModalFooter([
                dbc.Button("Select all", id=f"runs-selall-{mode}",
                           size="sm", outline=True, color="secondary",
                           n_clicks=0),
                dbc.Button([html.I(className="bi bi-download me-1"),
                            "Download Selected"],
                           id=f"runs-dl-selected-{mode}",
                           size="sm", style=BTN_GREEN, n_clicks=0),
                dbc.Button([html.I(className="bi bi-trash me-1"),
                            "Delete Selected"],
                           id=f"runs-del-selected-{mode}",
                           size="sm", color="danger", outline=True,
                           n_clicks=0,
                           title="Removes the selected runs from the "
                                 "camera permanently."),
                dbc.Button([html.I(className="bi bi-arrow-clockwise me-1"),
                            "Refresh"], id=f"runs-refresh-{mode}",
                           size="sm", style=BTN_DARK),
                dbc.Button("Close", id=f"runs-close-{mode}", size="sm",
                           style=BTN_DARK),
            ]),
        ], id=f"runs-modal-{mode}", size="lg", is_open=False),
    ] + ([
        # Post-run weight request (seed: weighed after shelling).
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle(
                f"{ctl.label} Sample Weight"), close_button=False),
            dbc.ModalBody([
                html.Div(id=f"wm-info-{mode}", className="mb-2"),
                dbc.InputGroup([
                    dbc.Input(id=f"wm-weight-{mode}", type="number", min=0,
                              placeholder="e.g. 200", autofocus=True,
                              n_submit=0),
                    dbc.InputGroupText("g"),
                ]),
                html.Div("Type the weight and press Enter, or click Save.",
                         className="text-muted", style={"fontSize": "11px"}),
                html.Div(id=f"wm-error-{mode}", className="small mt-1",
                         style={"minHeight": "1em"}),
                dcc.Store(id=f"wm-seen-{mode}", data=None),
            ]),
            dbc.ModalFooter([
                dbc.Button("No recorded weight", id=f"wm-skip-{mode}",
                           size="sm", style=BTN_DARK, n_clicks=0,
                           title="Leave the weight blank for this run"),
                dbc.Button("Save Weight", id=f"wm-save-{mode}",
                           size="sm", style=BTN_GREEN, n_clicks=0),
            ]),
        ], id=f"wm-modal-{mode}", is_open=False, backdrop="static",
           keyboard=False),
    ] if ctl.weight_after_stop else []), lg=6)


def build_admin_camera_tools(mode: str):
    ctl = CONTROLLERS[mode]
    return dbc.Col([
        html.Div([html.B(f"{ctl.label} Camera "),
                  html.Span(ctl.ip, style={"fontFamily": "monospace",
                                           "fontSize": "12px"})]),
        html.Div(id=f"admin-status-{mode}", className="small",
                 style={"fontFamily": "monospace", "minHeight": "2.5em"}),
        dbc.Row([
            dbc.Col(dbc.Button(
                [html.I(className="bi bi-sliders me-1"), "Camera Setup"],
                href=ctl.setup_url(), target="_blank", external_link=True,
                size="sm", style=BTN_DARK, className="w-100",
                title="On-camera setup page (mount, focus, calibration) — "
                      "saves to the camera itself."), md=6, className="mb-2"),
            dbc.Col(dbc.Button(
                [html.I(className="bi bi-archive me-1"), "Stored Runs…"],
                id=f"runs-open-{mode}", size="sm", style=BTN_DARK,
                className="w-100"), md=6, className="mb-2"),
        ], className="g-2"),
        dbc.Row([
            dbc.Col(dbc.Button("▶ Start App", id=f"app-start-{mode}",
                               size="sm", style=BTN_DARK, className="w-100"),
                    md=6, className="mb-2"),
            dbc.Col(dbc.Button("⏻ Stop App", id=f"app-stop-{mode}",
                               size="sm", style=BTN_DARK, className="w-100",
                               title="Stop the on-camera app to save power"),
                    md=6, className="mb-2"),
        ], className="g-2"),
        dbc.Button("show log", id=f"log-toggle-{mode}", color="link",
                   size="sm", n_clicks=0, className="p-0",
                   style={"fontSize": "11px", "color": "#888"}),
        dbc.Collapse(html.Pre(id=f"log-{mode}", style={
            "height": "160px", "overflowY": "auto", "fontSize": "11px",
            "fontFamily": "monospace", "backgroundColor": "#fafafa",
            "border": "1px solid #eee", "padding": "8px", "margin": 0,
            "whiteSpace": "pre-wrap"}),
            id=f"log-collapse-{mode}", is_open=False),
    ], lg=6)


def render_runs_list(mode: str):
    """(summary_children, checklist_options, disabled_values) for the
    camera's stored runs."""
    ctl = CONTROLLERS[mode]
    try:
        runs = ctl.list_runs()
    except CameraClientError as e:
        return (html.Div(f"Could not list runs: {e}",
                         className="text-danger small p-2"), [], [])
    if not runs:
        return (html.Div("No stored runs on this camera.",
                         className="text-muted small p-2"), [], [])
    total_mb = sum(r.get("bytes", 0) for r in runs) / 1e6
    # Retention figures come from the camera itself (/status), so this
    # line can't drift out of date when the policy is retuned.
    s = ctl.snapshot()
    days, cap = s.get("runs_retention_days"), s.get("runs_max_gb")
    free, disk = s.get("disk_free_gb"), s.get("disk_total_gb")
    policy = (f"auto-pruned after {days:.0f} days or past {cap:.0f} GB total"
              if days and cap else "auto-pruned periodically")
    disk_txt = (f"  ·  {free:,.0f} GB free of {disk:,.0f} GB on the camera"
                if free and disk else "")
    rows = [html.Div(
        f"{len(runs)} run(s) · {total_mb:,.0f} MB stored — runs are "
        f"{policy} (oldest first), so download anything you want to "
        f"keep.{disk_txt}",
        className="text-muted small mb-2")]
    options, disabled = [], []
    for r in runs:
        name = r.get("name", "")
        size_mb = r.get("bytes", 0) / 1e6
        tags = []
        if r.get("active"):
            tags.append("running")
            disabled.append(name)
        if r.get("has_csv"):
            tags.append("csv")
        suffix = f"  [{', '.join(tags)}]" if tags else ""
        options.append({
            "label": (f"{name}   ·   {r.get('files', 0)} files · "
                      f"{size_mb:,.0f} MB{suffix}"),
            "value": name,
            "disabled": name in disabled,
        })
    return rows, options, disabled


# ═════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

app.layout = dbc.Container([

    dcc.Interval(id="poll", interval=1000, n_intervals=0),
    dcc.Store(id="admin-auth", data=False, storage_type="session"),
    # Which pod run the seed panel has already been auto-populated from
    # (see the pod -> seed handoff below).
    dcc.Store(id="handoff-seen", data=0),
    dcc.Store(id="fb-path", data=DROPBOX_ROOT, storage_type="session"),
    dcc.Store(id="fb-selected-folder", data=DROPBOX_DEFAULT_DEST),
    dcc.Store(id="afb-path", data=DROPBOX_ROOT, storage_type="session"),
    dcc.Store(id="afb-selected", data="", storage_type="session"),

    # Header
    dbc.Row([
        dbc.Col(html.Div([html.Center([
            logo_element, html.P(),
            html.H4("Pod / Seed Grading", className="mb-1"),
            html.P("North Carolina State University — "
                   "Peanut Breeding & Genetics",
                   className="text-muted small"),
        ])]), width=12),
    ], className="mb-2 mt-3"),

    # Mode toggle (Grading is ALWAYS the default on page load) + Admin
    dbc.Row([
        dbc.Col(dbc.RadioItems(
            id="ui-mode",
            options=[{"label": "Grading", "value": "auto"},
                     {"label": "Test Mode", "value": "test"}],
            value="auto", inline=True,
            className="btn-group", inputClassName="btn-check",
            labelClassName="btn btn-outline-dark btn-sm",
            labelCheckedClassName="active",
        ), width="auto"),
        dbc.Col(html.Div(id="mode-hint", className="small text-muted pt-1"),
                width="auto"),
        # Session name — labels BOTH cameras' current sessions (shows in
        # the Previous sessions… lists and the summary downloads).
        dbc.Col(dbc.InputGroup([
            dbc.InputGroupText("Session", style={"fontSize": "12px"}),
            dbc.Input(id="session-name", size="sm", debounce=True,
                      placeholder="name this session…",
                      style={"maxWidth": "220px"}),
            dbc.Button("Set", id="btn-session-name", size="sm",
                       style=BTN_DARK),
            dbc.Button("+ New", id="btn-session-new", size="sm",
                       style=BTN_DARK, n_clicks=0,
                       title="Close this session and start a fresh one "
                             "(both cameras). Nothing is lost — the current "
                             "session stays on disk and can be reloaded."),
        ], size="sm"), width="auto", className="ms-auto"),
        dbc.Col(html.Div(id="session-name-result", className="small",
                         style={"minHeight": "1em"}), width="auto"),
        dbc.Col(dbc.Button([html.I(className="bi bi-gear me-1"), "Admin"],
                           id="admin-btn", size="sm", outline=True,
                           color="secondary"),
                width="auto"),
    ], className="mb-3 align-items-center"),

    # ── Camera panels ─────────────────────────────────────────────────────────
    dbc.Row([build_camera_panel(m) for m in MODES]),

    # ── Test-mode downloads (hidden in Grading mode) ──────────────────────────
    html.Div(dbc.Row(dbc.Col(
        _card("Test Mode — Download Data", [
            html.Div("Test runs are not appended anywhere — download them "
                     "here. 'Run' buttons use the selected history row "
                     "(no selection = the latest run); 'Session' buttons "
                     "cover every run in both cameras' current sessions.",
                     className="text-muted small mb-2"),
            dbc.Row([
                dbc.Col([
                    html.Div("File Name (run downloads)", style=LABEL_STYLE),
                    dbc.Input(id="local-filename", size="sm",
                              placeholder="Default: the CSV's own name"),
                ], md=12),
            ], className="g-2 mb-2 align-items-end"),
            dbc.Row([
                dbc.Col(dbc.Button([html.I(className="bi bi-download me-1"),
                                    "Run — Raw CSV"],
                                   id="btn-local-download",
                                   style=BTN_DARK, className="w-100",
                                   size="sm",
                                   title="Every tracked item of the selected/"
                                         "latest run (mm + colour columns)."),
                        md=6, className="mb-2"),
                dbc.Col(dbc.Button([html.I(className="bi bi-file-text me-1"),
                                    "Run — Summary"],
                                   id="btn-run-summary-download",
                                   style=BTN_DARK, className="w-100",
                                   size="sm",
                                   title="One summarized row for the "
                                         "selected/latest run."),
                        md=6, className="mb-2"),
                dbc.Col(dbc.Button([html.I(className="bi bi-table me-1"),
                                    "Session — All Summaries"],
                                   id="btn-summary-download", style=BTN_DARK,
                                   className="w-100", size="sm",
                                   title="One summarized row per plot/sample "
                                         "for every run in both sessions."),
                        md=6, className="mb-2"),
                dbc.Col(dbc.Button([html.I(className="bi bi-file-zip me-1"),
                                    "Session — All Raw CSVs (zip)"],
                                   id="btn-session-raw-download",
                                   style=BTN_DARK, className="w-100",
                                   size="sm",
                                   title="Zip of every run's raw CSV from "
                                         "both cameras' current sessions."),
                        md=6, className="mb-2"),
            ], className="g-2"),
            dcc.Download(id="local-download"),
            html.Div(id="local-result", className="mt-2"),
        ]), lg=6)), id="test-download-wrap", style={"display": "none"}),

    # ── Admin section (collapsed until login) ─────────────────────────────────
    dbc.Collapse([
        html.Hr(),
        dbc.Row([
            dbc.Col(html.H5("Admin Tools", className="mb-0"), width="auto"),
            dbc.Col(dbc.Button("Log out", id="btn-admin-logout", size="sm",
                               outline=True, color="secondary"),
                    width="auto", className="ms-auto"),
        ], className="mb-2 align-items-center"),

        _card("Camera Tools", dbc.Row(
            [build_admin_camera_tools(m) for m in MODES])),

        dbc.Row([
            dbc.Col(_card("Export to Dropbox (manual)", [
                html.Div("Normally unnecessary — Grading mode appends "
                         "automatically. Use these to retry a FAILED append "
                         "or push copies elsewhere.",
                         className="text-muted small mb-2"),
                html.Div("Upload files to a folder", style=LABEL_STYLE),
                html.Div("Saves new files — the selected run's raw CSV, or "
                         "the session summary CSV — into the Destination "
                         "Folder.", className="text-muted mb-2",
                         style={"fontSize": "11px"}),
                dbc.Row([
                    dbc.Col([
                        html.Div("Destination Folder", style=LABEL_STYLE),
                        dbc.InputGroup([
                            dbc.Input(id="dropbox-folder",
                                      value=DROPBOX_DEFAULT_DEST,
                                      size="sm", readonly=True),
                            dbc.Button("Browse…", id="fb-open", size="sm",
                                       style=BTN_DARK),
                        ]),
                    ], md=7),
                    dbc.Col([
                        html.Div("File Name", style=LABEL_STYLE),
                        dbc.Input(id="dropbox-filename", size="sm",
                                  placeholder="Default: the CSV's own name"),
                    ], md=5),
                ], className="g-2 mb-2 align-items-end"),
                dbc.Button([html.I(className="bi bi-cloud-upload me-1"),
                            "Upload Run CSV"],
                           id="btn-dropbox-download",
                           style=BTN_DARK, className="w-100 mb-2", size="sm"),
                dbc.Button([html.I(className="bi bi-table me-1"),
                            "Upload Session Summary"],
                           id="btn-dropbox-summary",
                           style=BTN_DARK, className="w-100 mb-2", size="sm"),
                html.Hr(className="my-2"),
                html.Div("Append to a trial workbook", style=LABEL_STYLE),
                html.Div("Creates no new files — fills each graded Plot's "
                         "row inside the selected .xlsx (pod + seed columns "
                         "merge on the same row). Re-appending is safe.",
                         className="text-muted mb-2",
                         style={"fontSize": "11px"}),
                dbc.Row([
                    dbc.Col([
                        html.Div("Trial File (.xlsx)", style=LABEL_STYLE),
                        dbc.InputGroup([
                            dbc.Input(id="append-file", size="sm",
                                      readonly=True,
                                      placeholder="No file selected"),
                            dbc.Button("Browse…", id="afb-open", size="sm",
                                       style=BTN_DARK),
                        ]),
                    ]),
                ], className="g-2 mb-2 align-items-end"),
                dbc.Button([html.I(className="bi bi-file-earmark-plus me-1"),
                            "Append All Session Runs"],
                           id="btn-dropbox-append",
                           style=BTN_DARK, className="w-100", size="sm"),
                html.Div(id="dropbox-result", className="mt-2"),
            ]), lg=6),
        ]),
    ], id="admin-section", is_open=False),

    # Status bar
    dbc.Row(dbc.Col(html.Div("Ready", id="status-bar", style={
        "backgroundColor": "#222", "color": "#eee", "padding": "6px 12px",
        "fontSize": "12px", "fontFamily": "monospace", "borderRadius": "4px",
    }), width=12), className="mt-2 mb-4"),

    # New-session confirmation
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Start a New Session?")),
        dbc.ModalBody([
            html.Div(id="new-session-summary", className="mb-2"),
            html.Div("The current session stays saved on disk — you can "
                     "reload it any time from Previous sessions…. New runs "
                     "will go into a fresh session on both cameras.",
                     className="text-muted small"),
            dbc.Row([
                dbc.Col(html.Div("Name the new session (optional)",
                                 style=LABEL_STYLE), width=12),
                dbc.Col(dbc.Input(id="new-session-name", size="sm",
                                  placeholder="e.g. LEW MET afternoon"),
                        width=12),
            ], className="g-2 mt-1"),
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="ns-cancel", size="sm", style=BTN_DARK,
                       n_clicks=0),
            dbc.Button("Start New Session", id="ns-confirm", size="sm",
                       style=BTN_GREEN, n_clicks=0),
        ]),
    ], id="new-session-modal", is_open=False),

    # Admin login modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Admin Login")),
        dbc.ModalBody([
            dbc.Input(id="admin-user", placeholder="Username", size="sm",
                      className="mb-2", autocomplete="off"),
            dbc.Input(id="admin-pass", placeholder="Password", size="sm",
                      type="password", n_submit=0),
            html.Div(id="login-err", className="small text-danger mt-2",
                     style={"minHeight": "1em"}),
        ]),
        dbc.ModalFooter(
            dbc.Button("Log in", id="btn-admin-login", size="sm",
                       style=BTN_DARK)),
    ], id="login-modal", is_open=False),

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
            dbc.Button("Use This Folder", id="fb-confirm", style=BTN_DARK,
                       size="sm"),
        ]),
    ], id="fb-modal", size="lg", is_open=False),

    # Trial-file browser modal (folders navigate, .xlsx selects)
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
# ADMIN + MODE CALLBACKS
# ═════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("admin-auth", "data"),
    Output("login-modal", "is_open"),
    Output("admin-section", "is_open"),
    Output("login-err", "children"),
    Output("admin-pass", "value"),
    Input("admin-btn", "n_clicks"),
    Input("btn-admin-login", "n_clicks"),
    Input("admin-pass", "n_submit"),
    Input("btn-admin-logout", "n_clicks"),
    State("admin-user", "value"),
    State("admin-pass", "value"),
    State("admin-auth", "data"),
    State("admin-section", "is_open"),
    prevent_initial_call=True,
)
def admin_control(_b, _l, _s, _o, user, pw, authed, sec_open):
    trig = dash.ctx.triggered_id
    if trig == "admin-btn":
        if authed:
            return authed, False, not sec_open, "", dash.no_update
        return authed, True, False, "", ""
    if trig in ("btn-admin-login", "admin-pass"):
        if (user or "").strip() == ADMIN_USER and (pw or "") == ADMIN_PASS:
            return True, False, True, "", ""
        return False, True, False, "Wrong username or password.", ""
    if trig == "btn-admin-logout":
        return False, False, False, "", ""
    raise dash.exceptions.PreventUpdate


@app.callback(
    Output("test-download-wrap", "style"),
    Output("mode-hint", "children"),
    Input("ui-mode", "value"),
)
def mode_visibility(ui_mode):
    if ui_mode == "test":
        return ({},
                "Test Mode — any sample name, nothing is appended to trials.")
    return ({"display": "none"},
            "Scan the bag's QR tag — the trial file is found automatically.")


# ── Session naming (labels BOTH cameras' current sessions) ───────────────────
@app.callback(
    Output("session-name-result", "children"),
    Output("session-name", "value"),
    Input("btn-session-name", "n_clicks"),
    Input("session-name", "n_submit"),
    Input("poll", "n_intervals"),
    State("session-name", "value"),
    prevent_initial_call=True,
)
def set_session_name(_n, _s, poll_n, name):
    trig = dash.ctx.triggered_id
    if trig == "poll":
        # Once, shortly after page load: show the current name (e.g. after
        # a browser refresh or a session load).
        if poll_n and poll_n <= 2:
            current = SESSIONS[MODES[0]].snapshot()["name"]
            return dash.no_update, (current or dash.no_update)
        raise dash.exceptions.PreventUpdate
    name = str(name or "").strip()
    for m in MODES:
        SESSIONS[m].set_name(name)
    if not name:
        return html.Span("Session name cleared.",
                         className="text-muted"), dash.no_update
    return html.Span("✓ named", className="text-success"), dash.no_update


# ── New session (both cameras) ───────────────────────────────────────────────
@app.callback(
    Output("new-session-modal", "is_open"),
    Output("new-session-summary", "children"),
    Output("session-name", "value", allow_duplicate=True),
    Output("session-name-result", "children", allow_duplicate=True),
    Input("btn-session-new", "n_clicks"),
    Input("ns-cancel", "n_clicks"),
    Input("ns-confirm", "n_clicks"),
    State("new-session-name", "value"),
    prevent_initial_call=True,
)
def new_session(_open, _cancel, _confirm, new_name):
    trig = dash.ctx.triggered_id
    if trig == "btn-session-new":
        if any_running():
            return (False, dash.no_update, dash.no_update,
                    html.Span("Stop the running camera first.",
                              className="text-danger"))
        counts = ", ".join(
            f"{CONTROLLERS[m].label} {len(SESSIONS[m].snapshot()['session'])}"
            for m in MODES)
        cur = SESSIONS[MODES[0]].snapshot()["name"] or "(unnamed)"
        return (True,
                html.Div([html.B(f"Current session: {cur}"),
                          html.Span(f" — {counts} run(s) recorded.")]),
                dash.no_update, "")
    if trig == "ns-confirm":
        name = str(new_name or "").strip()
        for m in MODES:
            SESSIONS[m].start_new(name)
        label = f"'{name}'" if name else "(unnamed)"
        return (False, dash.no_update, name,
                html.Span(f"✓ new session {label}",
                          className="text-success"))
    return False, dash.no_update, dash.no_update, ""


# ── Pod → seed autofill (consumes the handoff published after the append) ────
if POD_MODE and SEED_MODE:
    @app.callback(
        Output(f"name-{SEED_MODE}", "value", allow_duplicate=True),
        Output(f"plot-{SEED_MODE}", "value", allow_duplicate=True),
        Output(f"qr-result-{SEED_MODE}", "children", allow_duplicate=True),
        Output("handoff-seen", "data"),
        Input("poll", "n_intervals"),
        State("handoff-seen", "data"),
        State("ui-mode", "value"),
        prevent_initial_call=True,
    )
    def seed_autofill(_n, seen, ui_mode):
        with _HANDOFF_LOCK:
            seq, name, plot = (_HANDOFF["seq"], _HANDOFF["name"],
                               _HANDOFF["plot"])
        if not seq or seq == seen:
            raise dash.exceptions.PreventUpdate
        # Never overwrite a run in progress; wait for it to finish.
        if CONTROLLERS[SEED_MODE].snapshot()["running"]:
            raise dash.exceptions.PreventUpdate
        if ui_mode == "test":
            return name, plot, html.Span(
                f"↳ from pod: {name} (test mode)",
                className="text-success"), seq
        # Re-verify for the seed side: the pod append just landed, so this
        # normally confirms the plot is now ready to shell.
        trial_path, err = verify_before_start(SEED_MODE, name, plot)
        if err:
            return name, plot, html.Span(f"↳ from pod — ✗ {err}",
                                         className="text-danger"), seq
        return name, plot, html.Span(
            f"↳ from pod ✓ Plot {plot} → will append to "
            f"{Path(trial_path).name}", className="text-success"), seq


# ═════════════════════════════════════════════════════════════════════════════
# PER-CAMERA CALLBACKS
# ═════════════════════════════════════════════════════════════════════════════

def lighting_widgets(label: str, s: dict):
    """(header pill, panel strip, modal reading line, modal button label)
    from a snapshot's `lighting` block. Older cameras → all empty."""
    lt = s.get("lighting") if s.get("online") else None
    if not lt or not lt.get("status") or lt.get("status") == "unknown":
        return "", None, "", "Continue"
    g = lt.get("median_gray")
    gtxt = f"{g:.0f}" if isinstance(g, (int, float)) else "—"
    st = lt["status"]
    if st == "ok":
        return (dbc.Badge(f"💡 light {gtxt}", color="success",
                          className="fw-normal"),
                None, "", "Continue")
    low = st == "low"
    pill = dbc.Badge(f"💡 {'LOW LIGHT' if low else 'TOO BRIGHT'} · {gtxt}",
                     color="warning", text_color="dark",
                     className="fw-bold")
    strip = dbc.Alert([
        html.B(f"⚠ {'Low light' if low else 'Too much light'} on the "
               f"{label.lower()} camera (reading {gtxt}, warns "
               f"{'below' if low else 'above'} "
               f"{lt.get('warn_below_gray' if low else 'warn_above_gray', '?'):.0f}). "),
        ("Colour will be blacked out and sizes inflated until the lights "
         "are fixed — counting still works. Are the lights on?") if low else
        ("Highlights are clipping — colour will be unreliable. Dim or "
         "move the light."),
    ], color="warning", className="small py-2 mb-2")
    if low:
        reading = html.Span(f"Current reading: {gtxt} (warns below "
                            f"{lt.get('warn_below_gray', 45):.0f}) — still low.",
                            className="text-danger")
        btn = "Continue anyway"
    else:
        reading = html.Span(f"Current reading: {gtxt} — lights look OK now.",
                            className="text-success")
        btn = "Continue"
    return pill, strip, reading, btn


def idle_widget(ctl, s: dict, busy: bool):
    """Red banner when the camera (0.7.0+) sees material flowing while no
    run is active: >= 12 confident frames in its 3 s window, last within
    1.5 s. Same rule as the kit station."""
    ia = s.get("idle_activity") if s.get("online") else None
    flowing = (not busy and ia and ia.get("frames", 0) >= 12
               and ia.get("last_s_ago") is not None
               and ia["last_s_ago"] <= 1.5)
    if not flowing:
        return None
    need = "Sample Name" if ctl.weight_after_stop else "Sample Name and weight"
    return dbc.Alert([
        html.Div(f"⚠ {ctl.label}s are moving but no run is started",
                 className="fw-bold", style={"fontSize": "17px"}),
        html.Div(f"Scan the tag / enter the {need} and press Start — "
                 f"nothing is being counted right now.", className="small"),
    ], color="danger", className="py-2 mb-2")


def register_panel_callbacks(mode: str) -> None:
    ctl = CONTROLLERS[mode]

    # ── Poll: light, stats, session table, admin status, input states ────────
    @app.callback(
        Output(f"count-{mode}", "children"),
        Output(f"elapsed-{mode}", "children"),
        Output(f"fps-{mode}", "children"),
        Output(f"active-{mode}", "children"),
        Output(f"light-{mode}", "style"),
        Output(f"session-table-{mode}", "data"),
        Output(f"session-current-{mode}", "children"),
        Output(f"start-{mode}", "disabled"),
        Output(f"stop-{mode}", "disabled"),
        Output(f"name-{mode}", "disabled"),
        Output(f"plot-{mode}", "disabled"),
        Output(f"weight-{mode}", "disabled"),
        Output(f"saveframes-{mode}", "disabled"),
        Output(f"admin-status-{mode}", "children"),
        Output(f"log-{mode}", "children"),
        Output(f"lightpill-{mode}", "children"),
        Output(f"lightwarn-{mode}", "children"),
        Output(f"idlewarn-{mode}", "children"),
        Output(f"ll-reading-{mode}", "children"),
        Output(f"ll-continue-{mode}", "children"),
        Input("poll", "n_intervals"),
    )
    def poll_panel(_n, _mode=mode):
        s = CONTROLLERS[_mode].snapshot()
        running, stopping = s["running"], s["stopping"]
        ready = s["online"]
        light = {**LIGHT_BASE, "float": "right",
                 "backgroundColor": "#2f9e44" if ready else "#a80000",
                 "boxShadow": f"0 0 6px "
                              f"{'#2f9e44' if ready else '#a80000'}"}

        admin_lines = [html.Div(
            f"{'online' if s['online'] else 'OFFLINE'} · state {s['state']} "
            f"· {s['fps']:.1f} fps")]
        active_dl = [d for d in (s.get("downloads") or [])
                     if d["state"] in ("queued", "downloading")]
        if active_dl:
            admin_lines.append(html.Div(
                f"⬇ {len(active_dl)} download(s) in progress — see "
                f"Stored Runs…"))
        if s["online"]:
            admin_lines.append(html.Div(
                f"pod-counter v{s['app_version']} · staged "
                f"{s['staged_mode'] or '?'} @ {s['staged_at'] or '?'} · "
                f"config: {s['config_source'] or '?'}"))
            if s["staged_mode"] and s["staged_mode"] != _mode:
                admin_lines.append(dbc.Badge("wrong staged mode!",
                                             color="danger"))
            # camera clock (0.7.4+): the OAK 4 has no RTC — the station sets it
            clk = s.get("clock")
            if isinstance(clk, dict) and clk.get("synced"):
                admin_lines.append(html.Div(
                    f"clock: set from this PC · camera now {clk.get('now', '?')}"))
            elif isinstance(clk, dict):
                admin_lines.append(dbc.Badge(
                    "camera clock unset (1970) — being set from this PC…",
                    color="warning"))
            elif s.get("app_version"):
                admin_lines.append(dbc.Badge(
                    f"camera clock not managed (app {s['app_version']} < 0.7.4)",
                    color="warning"))
            if s["pending_restart"]:
                admin_lines.append(dbc.Badge(
                    "config pushed — restart to apply", color="warning"))

        busy = running or stopping
        sess = SESSIONS[_mode].snapshot()
        sess_name = sess["name"] or "(unnamed — set a name at the top)"
        current = f"Session: {sess_name} · {len(sess['session'])} run(s)"
        pill, lightwarn, ll_reading, ll_btn = lighting_widgets(
            CONTROLLERS[_mode].label, s)
        idlewarn = idle_widget(CONTROLLERS[_mode], s, busy)
        return (
            str(s["count"]), s["elapsed"], f"{s['fps']:.1f} fps",
            f"Active: {s['active']}", light,
            sess["session"], current,
            busy or not s["online"],          # start disabled
            (not running) or stopping,        # stop disabled
            busy, busy, busy, busy,           # inputs locked during a run
            admin_lines, s["log"],
            pill, lightwarn, idlewarn, ll_reading, ll_btn,
        )

    # ── Start ─────────────────────────────────────────────────────────────────
    @app.callback(
        Output(f"alert-{mode}", "children"),
        Output(f"ll-modal-{mode}", "is_open"),
        Input(f"start-{mode}", "n_clicks"),
        Input(f"ll-continue-{mode}", "n_clicks"),
        Input(f"ll-cancel-{mode}", "n_clicks"),
        State(f"name-{mode}", "value"),
        State(f"plot-{mode}", "value"),
        State(f"weight-{mode}", "value"),
        State(f"saveframes-{mode}", "value"),
        State("ui-mode", "value"),
        prevent_initial_call=True,
    )
    def on_start(_n, _cont, _cancel, name, plot, weight, save_frames,
                 ui_mode, _mode=mode):
        trig = dash.ctx.triggered_id
        if trig == f"ll-cancel-{_mode}":
            return dash.no_update, False
        if trig == f"start-{_mode}":
            # Low light (camera 0.7.2+): warn before starting — re-asked on
            # EVERY start by design; Continue in the modal goes ahead.
            lt = CONTROLLERS[_mode].snapshot().get("lighting") or {}
            if lt.get("status") == "low":
                return dash.no_update, True
        alert = _do_start(_mode, name, plot, weight, save_frames, ui_mode)
        return alert, False

    def _do_start(_mode, name, plot, weight, save_frames, ui_mode):
        ctlr = CONTROLLERS[_mode]
        name = str(name or "").strip()
        plot = str(plot or "").strip()
        weight = str(weight or "").strip()

        def _err(msg):
            return dbc.Alert(msg, color="danger", dismissable=True,
                             className="small mb-0",
                             style={"whiteSpace": "pre-wrap"})

        if ui_mode == "test":
            # Minimal checks; never appends anywhere.
            err = ctlr.start(name, plot, weight, bool(save_frames),
                             require_weight=False, meta={"test": True})
            if err:
                return _err(err)
            return dbc.Alert("Test run started (no trial append).",
                             color="info", dismissable=True, duration=4000,
                             className="small mb-0")

        # Grading mode — verify the trial file before anything moves.
        if not name:
            return _err("Sample Name is required — scan the bag's QR tag.")
        if not plot:
            return _err("Plot is required — scan the bag's QR tag.")
        if not ctlr.weight_after_stop and not weight:
            return _err("Weight is required before starting.")
        trial_path, err = verify_before_start(_mode, name, plot)
        if err:
            return _err(err)
        err = ctlr.start(name, plot, weight, save_frames=bool(save_frames),
                         meta={"trial_path": trial_path})
        if err:
            return _err(err)
        return dbc.Alert(
            f"Run started — Plot {plot} will append to "
            f"{Path(trial_path).name} on Stop.",
            color="success", dismissable=True, duration=6000,
            className="small mb-0")

    # ── Stop — also clears the sample inputs for the next run ────────────────
    # (The QR box itself is already cleared by every scan, and writing it
    # here would put a duplicate writer on the scan callback's self-loop,
    # which Dash's client-side graph refuses to fire.)
    @app.callback(
        Output(f"alert-{mode}", "children", allow_duplicate=True),
        Output(f"name-{mode}", "value", allow_duplicate=True),
        Output(f"plot-{mode}", "value", allow_duplicate=True),
        Output(f"weight-{mode}", "value"),
        Output(f"saveframes-{mode}", "value"),
        Output(f"qr-result-{mode}", "children", allow_duplicate=True),
        Input(f"stop-{mode}", "n_clicks"),
        prevent_initial_call=True,
    )
    def on_stop(_n, _mode=mode):
        err = CONTROLLERS[_mode].stop()
        if err:
            return (dbc.Alert(err, color="warning", dismissable=True,
                              className="small mb-0"),) + (dash.no_update,) * 5
        # Clear every per-sample input and return frame saving to its
        # default (on), so turning it off for one run can't quietly lose
        # the images for the rest of the day.
        return (dbc.Alert("Stopping — results land in the session table.",
                          color="warning", dismissable=True, duration=4000,
                          className="small mb-0"),
                "", "", None, True, "")

    # ── QR scan → sample name + plot (+ trial verification in Grading) ───────
    @app.callback(
        Output(f"name-{mode}", "value"),
        Output(f"plot-{mode}", "value"),
        Output(f"qr-{mode}", "value"),
        Output(f"qr-result-{mode}", "children"),
        Input(f"qr-{mode}", "value"),
        State("ui-mode", "value"),
        prevent_initial_call=True,
    )
    def qr_scanned(text, ui_mode, _mode=mode):
        if not (text or "").strip():
            return (dash.no_update,) * 4
        qr = parse_qr(text)
        if qr is None:
            return (dash.no_update, dash.no_update, dash.no_update,
                    html.Span("Unrecognized QR — expected "
                              "Year|Location|Accession|Trial|Plot",
                              className="text-danger"))
        name = qr_to_sample_name(text)
        plot = qr["Plot"]
        if ui_mode == "test":
            return name, plot, "", html.Span(
                f"✓ {name} (test mode — no trial checks)",
                className="text-success")
        # Grading mode: resolve the trial file and check the plot NOW so
        # mistakes surface before the sample is poured.
        trial_path, err = verify_before_start(_mode, name, plot)
        if err:
            return name, plot, "", html.Span(f"✗ {err}",
                                             className="text-danger")
        return name, plot, "", html.Span(
            f"✓ Plot {plot} → will append to {Path(trial_path).name}",
            className="text-success")

    # ── Post-run weight popup (weighed-after-shelling modes) ─────────────────
    # Two callbacks on purpose. When opening and answering shared one
    # callback, a poll tick landing before the client had committed
    # is_open=True re-ran the open branch and blanked the field the
    # operator had just typed into — which showed up as "Save needs two
    # clicks". The open guard is now the run number, not is_open, so it
    # can fire only once per finished run no matter how the ticks land.
    if ctl.weight_after_stop:
        @app.callback(
            Output(f"wm-modal-{mode}", "is_open"),
            Output(f"wm-info-{mode}", "children"),
            Output(f"wm-weight-{mode}", "value"),
            Output(f"wm-error-{mode}", "children"),
            Output(f"wm-seen-{mode}", "data"),
            Input("poll", "n_intervals"),
            State(f"wm-seen-{mode}", "data"),
            prevent_initial_call=True,
        )
        def weight_modal_open(_n, seen, _mode=mode):
            ctlr = CONTROLLERS[_mode]
            pending = ctlr.snapshot().get("pending_weight")
            if not pending or pending.get("n") == seen:
                raise dash.exceptions.PreventUpdate
            info = html.Div([
                html.B(f"{pending['plot']}"),
                html.Span(f" — {pending['count']} "
                          f"{ctlr.label.lower()}s counted. Enter the "
                          f"shelled sample weight:"),
            ])
            return True, info, None, "", pending.get("n")

        @app.callback(
            Output(f"wm-modal-{mode}", "is_open", allow_duplicate=True),
            Output(f"wm-weight-{mode}", "value", allow_duplicate=True),
            Output(f"wm-error-{mode}", "children", allow_duplicate=True),
            Input(f"wm-save-{mode}", "n_clicks"),
            Input(f"wm-weight-{mode}", "n_submit"),
            Input(f"wm-skip-{mode}", "n_clicks"),
            State(f"wm-weight-{mode}", "value"),
            prevent_initial_call=True,
        )
        def weight_modal_answer(_s, _e, _k, weight, _mode=mode):
            ctlr = CONTROLLERS[_mode]
            if dash.ctx.triggered_id == f"wm-skip-{_mode}":
                ctlr.set_pending_weight(None)
                return False, None, ""
            err = ctlr.set_pending_weight(
                "" if weight is None else str(weight))
            if err:
                return (dash.no_update, dash.no_update,
                        html.Span(err, className="text-danger"))
            return False, None, ""

    # ── App lifecycle (restart in panel; start/stop in Admin) ────────────────
    @app.callback(
        Output(f"alert-{mode}", "children", allow_duplicate=True),
        Input(f"app-start-{mode}", "n_clicks"),
        Input(f"app-restart-{mode}", "n_clicks"),
        Input(f"app-stop-{mode}", "n_clicks"),
        prevent_initial_call=True,
    )
    def on_app_action(_s, _r, _x, _mode=mode):
        trig = str(dash.ctx.triggered_id or "")
        action = {"app-start": "start", "app-restart": "restart",
                  "app-stop": "stop"}.get(trig.rsplit("-", 1)[0], "")
        if not action:
            raise dash.exceptions.PreventUpdate
        err = CONTROLLERS[_mode].app_action(action)
        if err:
            return dbc.Alert(err, color="warning", dismissable=True,
                             className="small mb-0")
        return dbc.Alert(f"Camera app {action} requested — give it ~20 s.",
                         color="info", dismissable=True, duration=6000,
                         className="small mb-0")

    # ── Log show/hide (Admin) ────────────────────────────────────────────────
    @app.callback(
        Output(f"log-collapse-{mode}", "is_open"),
        Output(f"log-toggle-{mode}", "children"),
        Input(f"log-toggle-{mode}", "n_clicks"),
        State(f"log-collapse-{mode}", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_log(_n, is_open):
        opened = not is_open
        return opened, ("hide log" if opened else "show log")

    # ── Sessions: list / load this camera's previous sessions ────────────────
    @app.callback(
        Output(f"session-select-{mode}", "options"),
        Input(f"btn-refresh-sessions-{mode}", "n_clicks"),
    )   # also fires at page load
    def refresh_sessions(_n, _mode=mode):
        return [{"label": s["label"], "value": s["path"]}
                for s in SESSIONS[_mode].list_sessions()]

    @app.callback(
        Output(f"session-load-result-{mode}", "children"),
        Output(f"session-table-{mode}", "selected_rows"),
        Input(f"btn-load-session-{mode}", "n_clicks"),
        State(f"session-select-{mode}", "value"),
        prevent_initial_call=True,
    )
    def load_session_cb(_n, path, _mode=mode):
        if not path:
            return (html.Span("Choose a session first.",
                              className="text-warning"), dash.no_update)
        if CONTROLLERS[_mode].snapshot()["running"]:
            return (html.Span("Stop the camera before loading a session.",
                              className="text-danger"), dash.no_update)
        n, err = SESSIONS[_mode].load_session(path)
        if err:
            return html.Span(err, className="text-danger"), dash.no_update
        return (html.Span(f"✓ Loaded {n} run(s) — new runs continue in this "
                          "session.", className="text-success"), [])

    # ── Stored-runs download progress (poll-driven, live) ────────────────────
    @app.callback(
        Output(f"runs-progress-{mode}", "children"),
        Input("poll", "n_intervals"),
    )
    def runs_progress(_n, _mode=mode):
        dls = CONTROLLERS[_mode].snapshot().get("downloads") or []
        if not dls:
            return []
        items = []
        for d in dls:
            if d["state"] == "queued":
                items.append(html.Div(f"{d['name']} — queued…",
                                      className="small text-muted mb-1"))
            elif d["state"] == "downloading":
                mb = d["done"] / 1e6
                if d["total"]:
                    pct = min(99, int(d["done"] * 100 / d["total"]))
                    label = (f"{d['name']} — {mb:,.0f} / "
                             f"{d['total'] / 1e6:,.0f} MB ({pct}%)")
                else:
                    pct, label = 100, f"{d['name']} — {mb:,.0f} MB…"
                items.append(html.Div([
                    html.Div(label, className="small"),
                    dbc.Progress(value=pct, striped=True, animated=True,
                                 style={"height": "10px"}),
                ], className="mb-2"))
            elif d["state"] == "done":
                items.append(html.Div(f"✓ {d['name']} → {d['dest']}",
                                      className="small text-success mb-1"))
            else:
                items.append(html.Div(f"✗ {d['name']} — {d['error']}",
                                      className="small text-danger mb-1"))
        return items

    # ── Stored-runs modal (opened from Admin) ────────────────────────────────
    # Multi-select: tick any number of runs, then Download/Delete Selected.
    # Downloads queue on the per-camera worker (serialized, progress above).
    @app.callback(
        Output(f"runs-modal-{mode}", "is_open"),
        Output(f"runs-list-{mode}", "children"),
        Output(f"runs-select-{mode}", "options"),
        Output(f"runs-select-{mode}", "value", allow_duplicate=True),
        Output(f"runs-result-{mode}", "children"),
        Input(f"runs-open-{mode}", "n_clicks"),
        Input(f"runs-close-{mode}", "n_clicks"),
        Input(f"runs-refresh-{mode}", "n_clicks"),
        Input(f"runs-dl-selected-{mode}", "n_clicks"),
        Input(f"runs-del-selected-{mode}", "n_clicks"),
        State(f"runs-select-{mode}", "value"),
        prevent_initial_call=True,
    )
    def runs_modal(_o, _c, _r, _dl, _de, selected, _mode=mode):
        trig = dash.ctx.triggered_id
        ctlr = CONTROLLERS[_mode]
        keep = (dash.no_update,) * 3            # list, options, value
        if trig == f"runs-close-{_mode}":
            return False, *keep, ""
        if trig in (f"runs-open-{_mode}", f"runs-refresh-{_mode}"):
            if trig == f"runs-refresh-{_mode}":
                ctlr.clear_finished_downloads()
            summary, options, _dis = render_runs_list(_mode)
            return True, summary, options, [], ""
        selected = selected or []
        if not selected:
            return (True, *keep,
                    html.Span("Tick at least one run first.",
                              className="text-warning"))
        if trig == f"runs-dl-selected-{_mode}":
            try:
                sizes = {r["name"]: r.get("bytes", 0)
                         for r in ctlr.list_runs()}
            except CameraClientError:
                sizes = {}
            queued, busy = [], []
            for name in selected:
                err = ctlr.download_run_async(
                    name, total_bytes=sizes.get(name, 0))
                (busy if err else queued).append(name)
            parts = []
            if queued:
                parts.append(f"✓ {len(queued)} download(s) queued — "
                             f"progress above; they run one at a time.")
            if busy:
                parts.append(f"{len(busy)} already in flight.")
            return (True, dash.no_update, dash.no_update, [],
                    html.Span("  ".join(parts), className="text-success"))
        if trig == f"runs-del-selected-{_mode}":
            done, failed = [], []
            for name in selected:
                try:
                    ctlr.delete_run(name)
                    done.append(name)
                except CameraClientError as e:
                    failed.append(f"{name} ({e})")
            msg = []
            if done:
                msg.append(f"✓ Deleted {len(done)} run(s) from the camera.")
            if failed:
                msg.append("Failed: " + "; ".join(failed))
            summary, options, _dis = render_runs_list(_mode)
            return (True, summary, options, [],
                    html.Span(" ".join(msg),
                              className=("text-danger" if failed
                                         else "text-success")))
        raise dash.exceptions.PreventUpdate

    # Select all / clear (toggles between all enabled runs and none).
    @app.callback(
        Output(f"runs-select-{mode}", "value"),
        Input(f"runs-selall-{mode}", "n_clicks"),
        State(f"runs-select-{mode}", "options"),
        State(f"runs-select-{mode}", "value"),
        prevent_initial_call=True,
    )
    def runs_select_all(_n, options, value, _mode=mode):
        all_vals = [o["value"] for o in (options or [])]
        return [] if set(value or []) == set(all_vals) else all_vals


for _m in MODES:
    register_panel_callbacks(_m)


# ═════════════════════════════════════════════════════════════════════════════
# SHARED CALLBACKS  — status bar, downloads, admin export
# ═════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("status-bar", "children"),
    Input("poll", "n_intervals"),
)
def poll_shared(_n):
    parts = []
    for m in MODES:
        s = CONTROLLERS[m].snapshot()
        if not s["online"]:
            parts.append(f"{s['label']}: offline")
        elif s["running"]:
            parts.append(f"{s['label']}: running {s['count']} "
                         f"@ {s['fps']:.1f} fps")
        elif s["stopping"]:
            parts.append(f"{s['label']}: stopping…")
        else:
            parts.append(f"{s['label']}: {s['state']}")
    return "  |  ".join(parts)


# ── Selection helpers — both tables support MULTI selection; the download
# buttons act on the union of selected rows, or the latest run if none. ──────
def _selected_runs(sel_map: dict) -> list:
    """Session-row dicts for every selected row across both tables."""
    out = []
    for m in MODES:
        runs = SESSIONS[m].snapshot()["session"]
        for i in (sel_map.get(m) or []):
            if 0 <= i < len(runs):
                out.append(runs[i])
    return out


def _latest_run_row() -> Optional[dict]:
    """The most recently completed run (with a CSV) across both cameras."""
    latest = max((SESSIONS[m].snapshot() for m in MODES),
                 key=lambda s: s["last_csv_time"], default=None)
    if not latest or not latest["last_csv"]:
        return None
    csv_name = Path(latest["last_csv"]).name
    for m in MODES:
        for r in reversed(SESSIONS[m].snapshot()["session"]):
            if r.get("csv_file") == csv_name:
                return r
    return None


def _rows_with_csv(rows: list) -> list:
    """(row, csv_path) pairs for rows whose CSV exists on disk."""
    out = []
    for r in rows:
        csvf = r.get("csv_file")
        if csvf and csvf != "(none)" and (OUTPUT_DIR / csvf).exists():
            out.append((r, OUTPUT_DIR / csvf))
    return out


def _session_file_prefix() -> str:
    """Session name (sanitised) for download filenames, else 'grading'."""
    name = SESSIONS[MODES[0]].snapshot()["name"]
    return safe_filename(name) if name else "grading"


# ── Local downloads (Test Mode card) ─────────────────────────────────────────
@app.callback(
    Output("local-download", "data"),
    Output("local-result", "children"),
    Input("btn-local-download", "n_clicks"),
    State("local-filename", "value"),
    State(f"session-table-{MODES[0]}", "selected_rows"),
    State(f"session-table-{MODES[-1]}", "selected_rows"),
    prevent_initial_call=True,
)
def local_download(_n, filename, sel_a, sel_b):
    rows = _selected_runs({MODES[0]: sel_a, MODES[-1]: sel_b})
    if not rows:
        rows = [r for r in [_latest_run_row()] if r]
    pairs = _rows_with_csv(rows)
    if not pairs:
        return dash.no_update, dbc.Alert(
            "No recordings available for the selected run(s) yet.",
            color="warning", className="small mb-0")
    if len(pairs) == 1:
        _row, csv_path = pairs[0]
        name = (filename or "").strip() or csv_path.name
        if not name.lower().endswith(".csv"):
            name += ".csv"
        return (dcc.send_file(str(csv_path), filename=name),
                dbc.Alert(f"Downloading {name}", color="success",
                          dismissable=True, className="small mb-0"))
    # Several selected → one zip.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for _row, p in pairs:
            z.write(p, arcname=p.name)
    name = f"{_session_file_prefix()}_selected_raw.zip"
    return (dcc.send_bytes(buf.getvalue(), filename=name),
            dbc.Alert(f"Downloading {name} — {len(pairs)} run CSV(s)",
                      color="success", dismissable=True,
                      className="small mb-0"))


@app.callback(
    Output("local-download", "data", allow_duplicate=True),
    Output("local-result", "children", allow_duplicate=True),
    Input("btn-summary-download", "n_clicks"),
    prevent_initial_call=True,
)
def summary_download(_n):
    df = session_summary_table(combined_runs(), OUTPUT_DIR)
    if df.empty:
        return dash.no_update, dbc.Alert(
            "No completed runs to summarize yet.", color="warning",
            className="small mb-0")
    name = f"{_session_file_prefix()}_summary_{time.strftime('%Y%m%d')}.csv"
    return (dcc.send_string(df.to_csv(index=False), filename=name),
            dbc.Alert(f"Downloading {name} — {len(df)} plot(s) summarized",
                      color="success", dismissable=True,
                      className="small mb-0"))


@app.callback(
    Output("local-download", "data", allow_duplicate=True),
    Output("local-result", "children", allow_duplicate=True),
    Input("btn-run-summary-download", "n_clicks"),
    State(f"session-table-{MODES[0]}", "selected_rows"),
    State(f"session-table-{MODES[-1]}", "selected_rows"),
    prevent_initial_call=True,
)
def run_summary_download(_n, sel_a, sel_b):
    rows = _selected_runs({MODES[0]: sel_a, MODES[-1]: sel_b})
    if not rows:
        rows = [r for r in [_latest_run_row()] if r]
    df = session_summary_table(rows, OUTPUT_DIR)
    if df.empty:
        return dash.no_update, dbc.Alert(
            "No recordings available for the selected run(s) yet.",
            color="warning", className="small mb-0")
    if len(rows) == 1 and rows[0].get("csv_file") not in (None, "(none)"):
        name = f"summary_{Path(rows[0]['csv_file']).stem}.csv"
    else:
        name = f"{_session_file_prefix()}_selected_summary.csv"
    return (dcc.send_string(df.to_csv(index=False), filename=name),
            dbc.Alert(f"Downloading {name} — {len(df)} row(s)",
                      color="success", dismissable=True,
                      className="small mb-0"))


@app.callback(
    Output("local-download", "data", allow_duplicate=True),
    Output("local-result", "children", allow_duplicate=True),
    Input("btn-session-raw-download", "n_clicks"),
    prevent_initial_call=True,
)
def session_raw_download(_n):
    buf = io.BytesIO()
    n = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for r in combined_runs():
            csvf = r.get("csv_file")
            if csvf and csvf != "(none)" and (OUTPUT_DIR / csvf).exists():
                z.write(OUTPUT_DIR / csvf, arcname=csvf)
                n += 1
    if n == 0:
        return dash.no_update, dbc.Alert(
            "No run CSVs in the current sessions yet.", color="warning",
            className="small mb-0")
    name = f"{_session_file_prefix()}_raw_{time.strftime('%Y%m%d')}.zip"
    return (dcc.send_bytes(buf.getvalue(), filename=name),
            dbc.Alert(f"Downloading {name} — {n} run CSV(s)",
                      color="success", dismissable=True,
                      className="small mb-0"))


# ── Dropbox folder browser (Admin) ───────────────────────────────────────────
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
    Input({"type": "fb-folder", "path": ALL}, "n_clicks"),
    State("fb-path", "data"),
    State({"type": "fb-folder", "path": ALL}, "id"),
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


# ── Trial-file browser (Admin append target) ─────────────────────────────────
@app.callback(
    Output("afb-modal", "is_open"),
    Output("append-file", "value"),
    Output("afb-selected", "data"),
    Input("afb-open", "n_clicks"),
    Input("afb-cancel", "n_clicks"),
    Input({"type": "afb-file", "path": ALL}, "n_clicks"),
    State("afb-selected", "data"),
    prevent_initial_call=True,
)
def afb_modal(open_c, cancel_c, file_clicks, selected):
    trig = dash.ctx.triggered_id
    if trig == "afb-open":
        return True, dash.no_update, dash.no_update
    if isinstance(trig, dict) and trig.get("type") == "afb-file":
        if not dash.ctx.triggered[0]["value"]:
            raise dash.exceptions.PreventUpdate
        return False, trig["path"], trig["path"]
    return False, dash.no_update, selected  # cancel


@app.callback(
    Output("afb-path", "data"),
    Input("afb-open", "n_clicks"),
    Input("afb-root", "n_clicks"),
    Input("afb-up", "n_clicks"),
    Input({"type": "afb-folder", "path": ALL}, "n_clicks"),
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


# ── Admin: append every session run to the selected trial file, by Plot ─────
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

    runs = [r for r in combined_runs()
            if r.get("csv_file") and r["csv_file"] != "(none)"]
    if not runs:
        return _alert("No completed runs this session yet.", "warning")

    todo, skipped = [], []
    for r in runs:
        plot = r.get("plot") or parse_plot(r.get("plot_name") or "")
        if not plot or not str(plot).isdigit():
            skipped.append(f"{r.get('mode')} '{r.get('plot_name') or 'unnamed'}' "
                           f"(no numeric plot)")
            continue
        csv_path = OUTPUT_DIR / r["csv_file"]
        if not csv_path.exists():
            skipped.append(f"{r.get('mode')} '{r.get('plot')}' (csv missing)")
            continue
        todo.append((r, plot, csv_path, (r.get("weight") or "").rstrip("g")))
    if not todo:
        return _alert("No runs with a numeric Plot to append. "
                      "Skipped: " + "; ".join(skipped), "warning")

    try:
        with _APPEND_LOCK:
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
                return _alert("Nothing appended. " + " | ".join(failed),
                              "danger")
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


# ── Admin: upload session summary CSV to the chosen folder ──────────────────
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
    df = session_summary_table(combined_runs(), OUTPUT_DIR)
    if df.empty:
        return dbc.Alert("No completed runs to summarize yet.",
                         color="warning", className="small mb-0")
    folder = (folder or "").strip()
    if folder == DROPBOX_DEFAULT_DEST:
        folder = DROPBOX_ROOT
    name = f"grading_summary_{time.strftime('%Y%m%d')}.csv"
    dest = f"{folder.rstrip('/')}/{name}"
    try:
        meta = team_manager.upload_file(df.to_csv(index=False).encode(),
                                        dest, mode="overwrite")
        return dbc.Alert(
            f"Uploaded {len(df)} plot(s) to "
            f"{getattr(meta, 'path_display', dest)}",
            color="success", dismissable=True, className="small mb-0")
    except Exception as e:  # noqa: BLE001
        return dbc.Alert(f"Upload failed: {e}", color="danger",
                         dismissable=True, className="small mb-0")


# ── Admin: upload the selected/latest run CSV to the chosen folder ──────────
@app.callback(
    Output("dropbox-result", "children"),
    Input("btn-dropbox-download", "n_clicks"),
    State("dropbox-folder", "value"),
    State("dropbox-filename", "value"),
    State(f"session-table-{MODES[0]}", "selected_rows"),
    State(f"session-table-{MODES[-1]}", "selected_rows"),
    prevent_initial_call=True,
)
def dropbox_download(_n, folder, filename, sel_a, sel_b):
    if team_manager is None:
        return dbc.Alert("Dropbox client not configured (check .env).",
                         color="danger", className="small mb-0")
    rows = _selected_runs({MODES[0]: sel_a, MODES[-1]: sel_b})
    if len(rows) > 1:
        return dbc.Alert("Select a single run for the CSV upload (or use "
                         "the summary/append tools for many).",
                         color="warning", className="small mb-0")
    if not rows:
        rows = [r for r in [_latest_run_row()] if r]
    pairs = _rows_with_csv(rows)
    if not pairs:
        return dbc.Alert("No recording available for the selected run yet.",
                         color="warning", className="small mb-0")
    csv_path = pairs[0][1]
    name = (filename or "").strip() or csv_path.name
    if not name.lower().endswith(".csv"):
        name += ".csv"
    folder = (folder or "").strip()
    if folder == DROPBOX_DEFAULT_DEST:
        folder = DROPBOX_ROOT
    dest = f"{folder.rstrip('/')}/{name}"
    try:
        data = csv_path.read_bytes()
        meta = team_manager.upload_file(data, dest, mode="add")
        return dbc.Alert(f"Uploaded to {getattr(meta, 'path_display', dest)}",
                         color="success", dismissable=True,
                         className="small mb-0")
    except Exception as e:  # noqa: BLE001
        return dbc.Alert(f"Upload failed: {e}", color="danger",
                         dismissable=True, className="small mb-0")


if __name__ == "__main__":
    # Per-station overrides (set in the launcher): PBG_HOST=0.0.0.0 serves the
    # app to other machines on the LAN; default 127.0.0.1 is this machine only.
    host = os.environ.get("PBG_HOST", "127.0.0.1")
    port = int(os.environ.get("PBG_PORT", "8050"))
    url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}"
    print(f"PBG Grading — open {url}")
    print("NOTE: run this on a machine that can reach the cameras (port 8808).")
    if os.environ.get("PBG_NO_BROWSER", "") not in ("1", "true", "yes"):
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    app.run(debug=False, host=host, port=port)
