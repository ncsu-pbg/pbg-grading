"""
CSV export for tracked pod measurements.

Writes one row per unique tracked pod with pixel measurements, real-world
(mm) measurements, Hunter L lightness (when colour calibration is available),
and plot-level metadata.

Hunter L pipeline
-----------------
mean_BGR (gamma-encoded, recorded per pod at max-area frame)
  → linearise (sRGB TRC, IEC 61966-2-1)
  → reorder to [R, G, B]
  → Calibration.color_matrix + color_bias  →  CIE XYZ D65
  → Hunter L = 10 × √Y

Because the colour correction matrix is an affine transform over a linear
space, applying it to the per-pod mean pixel value is mathematically
equivalent to correcting every pixel and then averaging.  No approximation
is introduced by this post-processing approach.

The hunter_l column is written as an empty string when either the
Calibration object has no colour matrix or the pod has no mean_bgr value
(mean_bgr is populated by future pipeline wiring in host_node.py).
"""

import csv
from pathlib import Path
from typing import Dict

from calibration import Calibration


def write_results_csv(
    track_dims: Dict[int, "TrackMeasurement"],
    output_path: str | Path,
    calibration: Calibration,
    plot_name: str = "",
    plot_weight: str = "",
) -> int:
    """
    Write accumulated per-track stats to a CSV file.

    Parameters
    ----------
    track_dims : dict
        ``{track_id: TrackMeasurement}`` from ``RealtimeTracker.max_dims``.
    output_path : str or Path
        Destination CSV file path.
    calibration : Calibration
        Pixel-to-mm and optional colour-correction calibration constants.
    plot_name : str
        User-entered plot identifier (metadata column).
    plot_weight : str
        User-entered plot weight (metadata column).

    Returns
    -------
    int
        Number of rows written.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    has_color = calibration.has_color()

    fieldnames = [
        "track_id",
        "length_px",
        "width_px",
        "area_px2",
        "length_mm",
        "width_mm",
        "area_mm2",
        "hunter_l",
        "width_quality",
        "plot_name",
        "plot_weight",
    ]

    rows = []
    for tid, meas in sorted(track_dims.items()):
        # Hunter L — empty string when not available so the column is always
        # present in the CSV but readable as a blank rather than "None".
        hunter_l: float | str = ""
        if has_color and meas.mean_bgr is not None:
            hl = calibration.hunter_l(meas.mean_bgr)
            if hl is not None:
                hunter_l = round(hl, 2)

        rows.append({
            "track_id":      tid,
            "length_px":     round(meas.length, 1),
            "width_px":      round(meas.width, 1),
            "area_px2":      round(meas.area, 1),
            "length_mm":     round(calibration.px_to_mm(meas.length), 2),
            "width_mm":      round(calibration.px_to_mm(meas.width), 2),
            "area_mm2":      round(calibration.area_px_to_mm2(meas.area), 2),
            "hunter_l":      hunter_l,
            "width_quality": getattr(meas, "width_quality", "clean"),
            "plot_name":     plot_name,
            "plot_weight":   plot_weight,
        })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)
