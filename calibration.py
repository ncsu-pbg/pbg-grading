"""
Pixel ↔ millimetre conversion and colour-correction utilities.

Calibration source: dual 10 mm ArUco markers (4×4_50 IDs 0 & 1,
14 mm centre-to-centre) plus 24 X-Rite ColorChecker Classic patches.

Defaults (unchanged from previous single-marker calibration):
    MM_PER_PX  = 0.347975
    PX_PER_MM  = 2.8738
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Calibration:
    """
    Immutable pixel ↔ mm and colour-correction constants.

    Attributes
    ----------
    mm_per_px : float
    px_per_mm : float
    color_matrix : nested tuple (3, 3) or None
        Affine 3×3 matrix that maps *linear-light* camera RGB → CIE XYZ (D65).
        Computed during setup from the 24-patch ColorChecker target.
    color_bias : tuple (3,) or None
        Additive XYZ offset that accompanies color_matrix.
    """

    mm_per_px:    float = 0.347975
    px_per_mm:    float = 2.8738
    color_matrix: Optional[Tuple[Tuple[float, ...], ...]] = None   # (3, 3)
    color_bias:   Optional[Tuple[float, ...]]             = None   # (3,)

    # ── Length / area ─────────────────────────────────────────────────────

    def px_to_mm(self, px: float) -> float:
        """Convert a linear measurement from pixels to mm."""
        return px * self.mm_per_px

    def mm_to_px(self, mm: float) -> float:
        """Convert a linear measurement from mm to pixels."""
        return mm * self.px_per_mm

    def area_px_to_mm2(self, area_px: float) -> float:
        """Convert an area from pixels² to mm²."""
        return area_px * (self.mm_per_px ** 2)

    def area_mm2_to_px(self, area_mm2: float) -> float:
        """Convert an area from mm² to pixels²."""
        return area_mm2 * (self.px_per_mm ** 2)

    # ── Colour correction ─────────────────────────────────────────────────

    def has_color(self) -> bool:
        """Return True if a colour correction matrix is available."""
        return self.color_matrix is not None and self.color_bias is not None

    def hunter_l(self, mean_bgr: Sequence[float]) -> Optional[float]:
        """
        Compute Hunter L lightness from a mean BGR camera measurement.

        Parameters
        ----------
        mean_bgr : array-like (3,)
            Mean [B, G, R] pixel values in [0, 255] from the pod mask.

        Returns
        -------
        float or None
            Hunter L (0 = black, ~100 = white reference).
            Returns None if no colour matrix has been set.

        Pipeline
        --------
        mean_BGR [0–255]
          → linearise (remove sRGB gamma, IEC 61966-2-1)
          → reorder to [R, G, B]
          → apply color_matrix + color_bias  →  CIE XYZ D65
          → Hunter L  =  10 × √Y

        Because the colour matrix is an affine (linear) transform, applying
        it to the per-pod mean is mathematically identical to applying it to
        every pixel then averaging — no approximation is introduced by the
        post-processing approach.
        """
        if self.color_matrix is None or self.color_bias is None:
            return None

        bgr = np.asarray(mean_bgr, dtype=np.float64) / 255.0
        rgb = bgr[::-1]                             # BGR → RGB
        lin = np.where(                             # sRGB gamma removal
            rgb <= 0.04045,
            rgb / 12.92,
            ((rgb + 0.055) / 1.055) ** 2.4,
        )
        M = np.asarray(self.color_matrix, dtype=np.float64)   # (3, 3)
        b = np.asarray(self.color_bias,   dtype=np.float64)   # (3,)
        xyz = lin @ M.T + b                        # CIE XYZ D65
        Y   = float(np.clip(xyz[1], 0.0, None))   # luminance
        return round(10.0 * np.sqrt(Y), 2) if Y > 0 else 0.0


def load_calibration(cfg: dict) -> Calibration:
    """Build a Calibration from a parsed config.yaml dict."""
    cal = cfg.get("calibration", {})

    cm_raw = cal.get("color_matrix")
    cb_raw = cal.get("color_bias")

    color_matrix = (
        tuple(tuple(float(v) for v in row) for row in cm_raw) if cm_raw else None
    )
    color_bias = (
        tuple(float(v) for v in cb_raw) if cb_raw else None
    )

    return Calibration(
        mm_per_px=float(cal.get("mm_per_px", 0.347975)),
        px_per_mm=float(cal.get("px_per_mm", 2.8738)),
        color_matrix=color_matrix,
        color_bias=color_bias,
    )
