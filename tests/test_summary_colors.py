"""Colour columns in the summary (Pod Hunter L + linear-light hex mean)."""

from pathlib import Path

from summary import all_summary_columns, average_hex, compute_summary

CSV = (
    "track_id,length_mm,width_mm,area_mm2,hunter_l,hunter_a,hunter_b,"
    "color_hex,plot_name,plot_weight\n"
    "1,34.0,14.0,340.0,50.0,8.0,20.0,#000000,1001,500\n"
    "2,36.0,15.0,380.0,60.0,9.0,21.0,#ffffff,1001,500\n"
)


def test_average_hex_linear_light():
    # Mean of black and white in LINEAR light is 0.5, which encodes to
    # sRGB ~0.735 -> 0xbc. A naive byte average would give 0x80 (too dark).
    assert average_hex(["#000000", "#ffffff"]) == "#bcbcbc"
    assert average_hex(["#a08050", "#a08050"]) == "#a08050"
    assert average_hex(["", "garbage", None]) == ""


def test_compute_summary_color_columns(tmp_path):
    p = tmp_path / "pod_1001_x.csv"
    p.write_text(CSV, encoding="utf-8")
    row = compute_summary(p, mode="Pod", plot_weight="500", plot="1001")
    assert row["Pod Hunter L"] == 55.0
    assert row["Pod Color [hex]"] == "#bcbcbc"

    seed_row = compute_summary(p, mode="Seed", plot_weight="200", plot="1001")
    assert "Seed Hunter L" not in seed_row          # pod-only per request
    assert seed_row["Seed Color [hex]"] == "#bcbcbc"

    cols = all_summary_columns()
    assert "Pod Hunter L" in cols and "Pod Color [hex]" in cols
    assert "Seed Color [hex]" in cols and "Seed Hunter L" not in cols


def test_merge_hex_into_excel_roundtrip_column(tmp_path):
    """Regression: a pod-only append creates the canonical Seed columns
    empty; Excel round-trips them as float64 NaN, and pandas 3 raised on
    writing a hex STRING into them ('float64 error' in the field,
    2026-07-23). merge must upcast to object instead."""
    import io
    import pandas as pd
    from summary import merge_summary_into_workbook

    trial = pd.DataFrame({"Plot": [1001, 1002],
                          "Seed Color [hex]": [float("nan"), float("nan")],
                          "Seed Count": [float("nan"), float("nan")]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        trial.to_excel(w, sheet_name="Trial", index=False)
    assert pd.read_excel(io.BytesIO(buf.getvalue()))["Seed Color [hex]"].dtype == "float64"

    out = merge_summary_into_workbook(buf.getvalue(), {
        "Plot": "1001", "Seed Count": 287, "Seed Color [hex]": "#a5623f",
    })
    df = pd.read_excel(io.BytesIO(out), sheet_name="Trial")
    row = df[df["Plot"] == 1001].iloc[0]
    assert row["Seed Color [hex]"] == "#a5623f"
    assert row["Seed Count"] == 287
