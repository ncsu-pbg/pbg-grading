"""
Run summary + Dropbox trial-file append for PBG Grading.

`compute_summary()` reduces a raw per-item grading CSV to a single summary
row: plot weight, count, and the mean / variance / skewness / kurtosis of the
corrected length, width and area. The statistics deliberately match the
QSorter Dash conventions so grading values line up with the seed data already
in the Peanut Database:

    mean       pandas Series.mean()
    variance   pandas Series.var()      -> sample (ddof=1)
    skewness   scipy.stats.skew()       -> biased  (bias=True)
    kurtosis   scipy.stats.kurtosis()   -> excess (fisher=True), biased

`merge_summary_into_workbook()` fills that row into an existing trial workbook,
matched on Plot -- mirroring QR Dash/PBG_QRscanner.py: it updates the matching
plot's row in the data sheet (adding any new columns) and preserves the other
sheets.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from scipy.stats import kurtosis, skew

# (summary label, raw-CSV column) for the three corrected measures.
_MEASURES = (("Length", "length_mm"), ("Width", "width_mm"), ("Area", "area_mm2"))

# Decimal places for values written to the summary row (matches QSorter).
_ROUND = 1

# Default data sheet in the trial workbook (matches QR Dash / QSorter files).
_DATA_SHEET = "Trial"


def _prefix(mode: str) -> str:
    """'Pod' for pod mode, 'Seed' otherwise — drives the column names."""
    return "Pod" if str(mode).lower().startswith("pod") else "Seed"


def all_summary_columns() -> list:
    """The canonical summary block: all Pod columns, then all Seed columns.

    Created wholesale on the first append to a file (whichever mode runs
    first), so Pod always precedes Seed in the trial file's column order
    even when seed data is appended before pod data.
    """
    cols = []
    for pfx in ("Pod", "Seed"):
        cols.append(f"{pfx} Weight [g]")
        cols.append(f"{pfx} Count")
        for label, _ in _MEASURES:
            for stat in ("M", "V", "S", "K"):
                cols.append(f"{pfx} {label} [{stat}]")
        # Whole-sample colour: mean coat colour as a hex code for both
        # modes; Hunter L (lightness) additionally for pods.
        if pfx == "Pod":
            cols.append("Pod Hunter L")
        cols.append(f"{pfx} Color [hex]")
    return cols


# ── Colour averaging ─────────────────────────────────────────────────────────
# Hex codes are gamma-encoded sRGB; a perceptually meaningful average must
# be computed in LINEAR light (decode → mean → re-encode). Averaging the
# hex bytes directly would bias dark.

def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> float:
    c = max(0.0, min(1.0, c))
    return c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def average_hex(values) -> str:
    """Mean of '#RRGGBB' colours in linear-light space; '' if none valid."""
    sums = [0.0, 0.0, 0.0]
    n = 0
    for v in values:
        s = str(v or "").strip().lstrip("#")
        if len(s) != 6:
            continue
        try:
            rgb = [int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
        except ValueError:
            continue
        for i in range(3):
            sums[i] += _srgb_to_linear(rgb[i])
        n += 1
    if n == 0:
        return ""
    return "#" + "".join(
        f"{round(_linear_to_srgb(s / n) * 255):02x}" for s in sums)


def parse_qr(text: str) -> Optional[Dict[str, str]]:
    """Parse a handheld-reader QR payload; None if it isn't one.

    Tags encode Year|Location|NC_Accession|Name|Plot (pipe-delimited, same
    format the QR Dash app scans). Mirrors its parsing: split on '|', strip
    each field, accept >= 5 parts (extras ignored). Readers often append a
    CR/newline suffix — stripped here.
    """
    parts = [p.strip() for p in str(text or "").strip().split("|")]
    if len(parts) < 5:
        return None
    return {"Year": parts[0], "Location": parts[1], "NC_Accession": parts[2],
            "Name": parts[3], "Plot": parts[4]}


def qr_to_sample_name(text: str) -> Optional[str]:
    """Underscore-joined sample name from a QR payload, or None.

    [Year]_[Location]_[NC_Accession]_[Name]_[Plot] — used for the raw CSV
    name and the image folder; parse_plot() recovers Plot from its last
    field at append time.
    """
    qr = parse_qr(text)
    if qr is None:
        return None
    return "_".join([qr["Year"], qr["Location"], qr["NC_Accession"],
                     qr["Name"], qr["Plot"]])


def parse_plot(sample_name: str) -> Optional[str]:
    """Plot number from a scanned sample name, or None.

    QR sample names follow [Year]_[Location]_[NC_Accession]_[Name]_[Plot];
    the plot is the last underscore-separated field. Returns it only when it
    is a whole number (manual names like 'Sample 1' yield None).
    """
    if not sample_name:
        return None
    tok = str(sample_name).strip().split("_")[-1].strip()
    return tok if tok.isdigit() else None


def _num(value) -> object:
    """Round a finite number to _ROUND places; blank ('') for nan/inf/None."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return ""
    return round(f, _ROUND) if math.isfinite(f) else ""


def compute_summary(csv_path, mode: str, plot_weight="", plot=None) -> Dict[str, object]:
    """Reduce a raw grading CSV to one summary row (column name -> value).

    Parameters
    ----------
    csv_path : path to the raw per-item CSV written by export.write_results_csv.
    mode     : 'pod' or 'seed' (selects the Pod/Seed column prefix).
    plot_weight : the user-entered plot weight (carried as '<Pod|Seed> Weight [g]').
    plot     : the plot identifier used to match a row in the trial workbook.
    """
    df = pd.read_csv(csv_path)
    pfx = _prefix(mode)
    out: Dict[str, object] = {}

    if plot is not None and str(plot).strip() != "":
        out["Plot"] = plot

    w = str(plot_weight).strip()
    out[f"{pfx} Weight [g]"] = _num(w) if w else ""
    out[f"{pfx} Count"] = int(len(df))

    for label, col in _MEASURES:
        s = (pd.to_numeric(df[col], errors="coerce").dropna()
             if col in df.columns else pd.Series(dtype=float))
        n = len(s)
        out[f"{pfx} {label} [M]"] = _num(s.mean()) if n else ""
        out[f"{pfx} {label} [V]"] = _num(s.var()) if n else ""            # ddof=1
        out[f"{pfx} {label} [S]"] = _num(skew(s)) if n else ""            # biased
        out[f"{pfx} {label} [K]"] = _num(kurtosis(s)) if n else ""        # excess, biased

    # Whole-sample colour (blank when the run had no colour calibration).
    if pfx == "Pod":
        hl = (pd.to_numeric(df["hunter_l"], errors="coerce").dropna()
              if "hunter_l" in df.columns else pd.Series(dtype=float))
        out["Pod Hunter L"] = _num(hl.mean()) if len(hl) else ""
    out[f"{pfx} Color [hex]"] = (average_hex(df["color_hex"])
                                 if "color_hex" in df.columns else "")

    return out


def session_summary_table(runs: List[dict], output_dir) -> pd.DataFrame:
    """Whole-session summary: one row per plot, same stats as the append.

    Compiles every completed run into the trial-file shape — Sample Name and
    Plot identifiers followed by the Pod block then the Seed block — so a
    session can be exported as summarized data without Dropbox. Pod and seed
    runs of the same plot merge into one row; a re-graded plot takes the later
    run's values; blanks never erase earlier values (same rules as the
    Dropbox append). Runs without a plot are keyed by sample name.
    """
    rows: Dict[str, Dict[str, object]] = {}
    order: List[str] = []
    for r in runs:
        csvf = r.get("csv_file")
        if not csvf or csvf == "(none)":
            continue
        p = Path(output_dir) / csvf
        if not p.exists():
            continue
        plot = r.get("plot") or parse_plot(r.get("plot_name") or "") or ""
        s = compute_summary(p, r.get("mode") or "",
                            plot_weight=(r.get("weight") or "").rstrip("g"),
                            plot=plot or None)
        s.pop("Plot", None)
        key = str(plot) if plot else (r.get("plot_name") or f"run {r.get('n')}")
        if key not in rows:
            rows[key] = {"Sample Name": "", "Plot": plot}
            order.append(key)
        rows[key]["Sample Name"] = r.get("plot_name") or rows[key]["Sample Name"]
        for col, val in s.items():
            if val == "":
                continue
            rows[key][col] = val
    cols = ["Sample Name", "Plot"] + all_summary_columns()
    if not order:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame([rows[k] for k in order])
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]


def merge_summary_into_workbook(existing_bytes: bytes, summary: Dict[str, object],
                                sheet_name: str = _DATA_SHEET) -> bytes:
    """Fill `summary` into the trial workbook, matched on Plot; return new bytes.

    Updates the row whose Plot equals summary['Plot'] in `sheet_name`, adding
    any columns that don't exist yet, and writes every other sheet back
    unchanged. Raises ValueError with a clear message on any mismatch.
    """
    sheets = pd.read_excel(io.BytesIO(existing_bytes), sheet_name=None)  # {name: df}
    if not sheets:
        raise ValueError("The selected workbook has no sheets.")
    if sheet_name not in sheets:
        sheet_name = next(iter(sheets))  # fall back to the first sheet

    data = sheets[sheet_name].copy()
    if "Plot" not in data.columns:
        raise ValueError(f"Sheet '{sheet_name}' has no 'Plot' column to match on.")

    plot = summary.get("Plot")
    if plot is None or str(plot).strip() == "":
        raise ValueError("No Plot value to append — enter/scan a plot number.")
    try:
        plot_key = int(str(plot).strip())
    except (TypeError, ValueError):
        raise ValueError(f"Plot '{plot}' is not a whole number.")

    data["Plot"] = pd.to_numeric(data["Plot"], errors="coerce").astype("Int64")
    mask = data["Plot"] == plot_key
    if not mask.any():
        raise ValueError(f"Plot {plot_key} was not found in sheet '{sheet_name}'.")

    # Create the whole canonical block up front (Pod columns before Seed) so
    # the file's column order is fixed no matter which mode appends first.
    for col in all_summary_columns():
        if col not in data.columns:
            data[col] = pd.NA

    updated = []
    for col, val in summary.items():
        if col == "Plot":
            continue
        # Like QR Dash: never let a blank erase a value already in the file
        # (e.g. re-running a plot without re-entering the weight).
        if val is None or (isinstance(val, str) and val.strip() == ""):
            continue
        if col not in data.columns:
            data[col] = pd.NA
        elif (pd.api.types.is_integer_dtype(data[col].dtype)
                and isinstance(val, float) and not val.is_integer()):
            # A column can round-trip through Excel as integer (all whole
            # values so far); pandas 3 raises rather than upcast on a float
            # write, so upcast explicitly.
            data[col] = data[col].astype("float64")
        elif (isinstance(val, str)
                and not (data[col].dtype == object
                         or pd.api.types.is_string_dtype(data[col].dtype))):
            # Same round-trip problem for text: an all-empty column (e.g.
            # 'Seed Color [hex]' created by a pod-only append) reads back
            # from Excel as float64 NaNs, and pandas 3 raises on writing a
            # string into it. Upcast to object first.
            data[col] = data[col].astype("object")
        data.loc[mask, col] = val
        updated.append(col)

    sheets[sheet_name] = data
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, sdf in sheets.items():
            sdf.to_excel(writer, sheet_name=name, index=False)
    return buf.getvalue()
