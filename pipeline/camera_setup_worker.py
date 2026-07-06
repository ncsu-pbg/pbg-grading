"""
Camera Setup Worker — dual-ArUco calibration target helper.

Target layout (calibration_target_30573.pdf)
--------------------------------------------
Two 10 mm × 10 mm ArUco 4×4_50 markers (IDs 0 and 1) stacked vertically,
14 mm centre-to-centre, at the right end of 24 X-Rite ColorChecker Classic
patches arranged in a 4-row × 6-column grid (6 mm × 6 mm each, 7 mm pitch).

Per-frame pipeline
------------------
1.  Detect both ArUco markers (sub-pixel corner refinement).
2.  Compute px/mm from 8 side lengths and the 14 mm C-C distance (weighted).
3.  Estimate camera tilt with IPPE ambiguity resolution via patches:
      a. IPPE on the 8 marker corners → up to 2 ambiguous mirror-twin
         candidate poses. For a small near-perpendicular target both
         candidates reproject the markers with virtually identical
         sub-pixel error, so picking by marker-only reprojection
         error is noise-driven and on certain frames the wrong
         (mirror) pose wins, giving 100°+ tilt readings.
      b. Run the full patch-refinement pipeline against EACH candidate:
         project the 24 patch world centres, refine to sub-pixel via
         gradient-magnitude centroid in a 7 mm window (with a ring-
         vs-interior density gate to reject random background
         texture), then re-solve with marker corners + valid patches
         using cv2.solvePnPRefineLM. Score each candidate by
         (n_valid_patches, −post_refine_reprojection_error). The
         wrong-twin pose lands patches in the background where almost
         nothing passes the density gate; the correct pose finds
         most of them.
      c. The patches widen the pose baseline from 10 × 24 mm (markers
         alone) to ~52 × 24 mm — roughly a 5× improvement in
         tilt-around-vertical-axis accuracy.
    10-frame rolling average; clears on positional jumps > 5°.
4.  Compute axis-aligned 100 mm × 100 mm ROI centred on the marker midpoint.
5.  Once tilt is stable (< 1°, 10-frame window full), project the 24 patch
    world positions to image coords, sample mean BGR in ± 2 mm boxes, and
    accumulate 5 stable frames.
6.  Fit a 3×3 + bias colour-correction matrix (linear camera RGB → CIE XYZ D65)
    using least squares over the 24 averaged patch colours.

World coordinate system
-----------------------
Origin  = midpoint between the two ArUco marker centres.
+X      = rightward across the target (toward the markers from the patches)
+Y      = upward (from bottom marker toward top marker)
+Z      = toward the camera (out of the surface)

Patch layout (mm, patches are at negative X)
    Y = +10.5   Row 1: dark skin … bluish-green
    Y =  +3.5   Row 2: orange … orange-yellow
    Y =  -3.5   Row 3: blue … cyan
    Y = -10.5   Row 4: white … black  (neutrals)
    X = -47, -40, -33, -26, -19, -12 mm (columns left-to-right)

Intrinsics source priority
--------------------------
1.  Device factory calibration via device.readCalibration()
2.  Approximate fallback (focal ≈ frame width); save is blocked.
"""

import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import depthai as dai
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage


# ─────────────────────────────────────────────────────────────────────────────
# Physical target constants
# ─────────────────────────────────────────────────────────────────────────────

ARUCO_DICT_ID   = cv2.aruco.DICT_4X4_50

MARKER_SIZE_MM  = 10.0            # side length of each ArUco marker
MARKER_C_C_MM   = 14.0            # centre-to-centre distance between markers
HALF_MARKER_MM  = MARKER_SIZE_MM / 2.0   # 5.0 mm
HALF_C_C_MM     = MARKER_C_C_MM  / 2.0   # 7.0 mm

ROI_SIZE_MM     = 100.0
HALF_ROI_MM     = ROI_SIZE_MM / 2.0

AXIS_LENGTH_MM  = 8.0             # drawFrameAxes display length (mm)
PATCH_SAMPLE_MM = 2.0             # ± mm box around each patch centre to sample

# Tilt thresholds
TILT_OK_DEG   = 1.0    # < 1°  → green
TILT_WARN_DEG = 3.0    # < 3°  → orange; ≥ 3° → red

# Rolling-window sizes
TILT_WINDOW  = 10   # frames for stable tilt average
COLOR_WINDOW = 5    # frames for stable colour average (after tilt is OK)

# BGR colour palette for overlays
_GREEN  = (0,   220,   0)
_ORANGE = (0,   165, 255)
_RED    = (0,     0, 255)
_WHITE  = (255, 255, 255)
_GREY   = (160, 160, 160)


# ─────────────────────────────────────────────────────────────────────────────
# Pose object points (world coordinates, mm)
#
# Corner order matches cv2.aruco output: TL, TR, BR, BL.
# World Y is up, so the marker that appears higher in the image has +Y centre.
# ─────────────────────────────────────────────────────────────────────────────

# Top marker: centre at (0, +7, 0) mm
_OBJ_TOP = np.array([
    [-HALF_MARKER_MM,  HALF_C_C_MM + HALF_MARKER_MM, 0.0],   # TL  (-5, +12)
    [ HALF_MARKER_MM,  HALF_C_C_MM + HALF_MARKER_MM, 0.0],   # TR  (+5, +12)
    [ HALF_MARKER_MM,  HALF_C_C_MM - HALF_MARKER_MM, 0.0],   # BR  (+5,  +2)
    [-HALF_MARKER_MM,  HALF_C_C_MM - HALF_MARKER_MM, 0.0],   # BL  (-5,  +2)
], dtype=np.float64)

# Bottom marker: centre at (0, -7, 0) mm
_OBJ_BOT = np.array([
    [-HALF_MARKER_MM, -HALF_C_C_MM + HALF_MARKER_MM, 0.0],   # TL  (-5,  -2)
    [ HALF_MARKER_MM, -HALF_C_C_MM + HALF_MARKER_MM, 0.0],   # TR  (+5,  -2)
    [ HALF_MARKER_MM, -HALF_C_C_MM - HALF_MARKER_MM, 0.0],   # BR  (+5, -12)
    [-HALF_MARKER_MM, -HALF_C_C_MM - HALF_MARKER_MM, 0.0],   # BL  (-5, -12)
], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Colour patch world positions and reference values
# ─────────────────────────────────────────────────────────────────────────────

_PATCH_X_MM = np.array([-47., -40., -33., -26., -19., -12.], dtype=np.float64)
_PATCH_Y_MM = np.array([+10.5, +3.5, -3.5, -10.5],           dtype=np.float64)

# (24, 3) world positions, row-major: Row1-Col1 … Row4-Col6
_PATCH_WORLD = np.array(
    [[x, y, 0.0] for y in _PATCH_Y_MM for x in _PATCH_X_MM],
    dtype=np.float64,
)

# X-Rite ColorChecker Classic reference sRGB (D65), same row-major order.
# Values taken directly from the calibration target PDF (verified match).
_PATCH_REF_SRGB = np.array([
    # Row 1 — top (Y = +10.5 mm)
    [115,  82,  68], [194, 150, 130], [ 98, 122, 157],
    [ 87, 108,  67], [133, 128, 177], [103, 189, 170],
    # Row 2 (Y = +3.5 mm)
    [214, 126,  44], [ 80,  91, 166], [193,  90,  99],
    [ 94,  60, 108], [157, 188,  64], [224, 163,  46],
    # Row 3 (Y = -3.5 mm)
    [ 56,  61, 150], [ 70, 148,  73], [175,  54,  60],
    [231, 199,  31], [187,  86, 149], [  8, 133, 161],
    # Row 4 — neutrals (Y = -10.5 mm)
    [243, 243, 242], [200, 200, 200], [160, 160, 160],
    [122, 122, 121], [ 85,  85,  85], [ 52,  52,  52],
], dtype=np.float32)

# IEC 61966-2-1  linear sRGB → CIE XYZ D65
_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# ArUco detector factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_detector() -> Any:
    """Return an ArUco detector with sub-pixel corner refinement enabled."""
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    p = cv2.aruco.DetectorParameters()
    try:
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    except AttributeError:
        pass
    try:
        return cv2.aruco.ArucoDetector(d, p)
    except AttributeError:
        return (d, p)


def _detect_markers(detector: Any, gray: np.ndarray):
    if isinstance(detector, tuple):
        d, p = detector
        return cv2.aruco.detectMarkers(gray, d, parameters=p)
    return detector.detectMarkers(gray)


# ─────────────────────────────────────────────────────────────────────────────
# Colour utilities
# ─────────────────────────────────────────────────────────────────────────────

def _srgb_linearize(v: np.ndarray) -> np.ndarray:
    """Remove sRGB gamma (IEC 61966-2-1); input and output in [0, 1]."""
    return np.where(
        v <= 0.04045,
        v / 12.92,
        ((v + 0.055) / 1.055) ** 2.4,
    )


def _compute_color_matrix(
    obs_bgr: np.ndarray,   # (24, 3) camera mean BGR, float [0, 255]
    ref_srgb: np.ndarray,  # (24, 3) reference sRGB,  float [0, 255]
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit a 3×3 + bias affine colour-correction matrix in linear light:

        linear_camera_RGB  →  CIE XYZ (D65)

    Both the camera observations and the reference values are linearised
    (sRGB gamma removed) before fitting so that the resulting matrix
    operates in radiometrically linear space.  This lets Hunter L be
    computed directly from the XYZ output without a further linearisation
    step at export time.

    Returns
    -------
    M    : (3, 3) ndarray  —  maps [R_lin, G_lin, B_lin] → [X, Y, Z]
    bias : (3,)   ndarray  —  additive XYZ offset
    rms  : float           —  RMS residual as % of unit XYZ range
    """
    ref_lin = _srgb_linearize(ref_srgb.astype(np.float64) / 255.0)   # (24, 3)
    ref_xyz = ref_lin @ _SRGB_TO_XYZ.T                                # (24, 3)

    cam_lin = _srgb_linearize(obs_bgr.astype(np.float64) / 255.0)    # (24, 3) BGR
    cam_rgb = cam_lin[:, ::-1]                                         # → RGB order

    # Augmented system:  [cam_rgb | 1] @ W  =  ref_xyz
    A = np.hstack([cam_rgb, np.ones((len(ref_xyz), 1), dtype=np.float64)])
    W, _, _, _ = np.linalg.lstsq(A, ref_xyz, rcond=None)             # (4, 3)

    M    = W[:3, :].T    # (3, 3)
    bias = W[3, :]       # (3,)

    rms = float(np.sqrt(np.mean((A @ W - ref_xyz) ** 2)) * 100.0)
    return M, bias, rms


# ─────────────────────────────────────────────────────────────────────────────
# Patch sampler
# ─────────────────────────────────────────────────────────────────────────────

def _sample_patch_colors(
    bgr: np.ndarray,
    world_pts: np.ndarray,   # (24, 3) world positions (mm)
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    half_px: float,          # sample-box half-width in pixels (≈ PATCH_SAMPLE_MM × px/mm)
    frame_w: int,
    frame_h: int,
) -> Optional[np.ndarray]:
    """
    Project patch world positions to image space and sample mean BGR
    in a square of ± half_px pixels around each projected centre.

    Returns (24, 3) float32 [0, 255], or None if any patch clips the frame.
    """
    proj, _ = cv2.projectPoints(
        world_pts.astype(np.float32), rvec, tvec, K, dist
    )
    proj = proj.reshape(-1, 2)

    half   = max(1, int(round(half_px)))
    colors = np.zeros((len(world_pts), 3), dtype=np.float32)

    for i, (px, py) in enumerate(proj):
        x1 = int(round(px)) - half
        y1 = int(round(py)) - half
        x2 = x1 + 2 * half
        y2 = y1 + 2 * half
        if x1 < 0 or y1 < 0 or x2 > frame_w or y2 > frame_h:
            return None
        colors[i] = bgr[y1:y2, x1:x2].mean(axis=(0, 1))

    return colors


# ─────────────────────────────────────────────────────────────────────────────
# Sub-pixel patch-centre refinement for wide-baseline pose
# ─────────────────────────────────────────────────────────────────────────────

def _refine_patch_centers(
    gray: np.ndarray,
    initial_proj: np.ndarray,    # (N, 2) projected patch centres in image px
    px_per_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sub-pixel refine projected patch centres via gradient-magnitude centroid.

    Each colour patch is 6 mm × 6 mm with a 1 mm white margin before the
    next patch (centre-to-centre = 7 mm). A 7 mm × 7 mm window centred on
    the projected centre captures the patch's full perimeter — the square
    ring of strong edges where the patch meets the printed white border —
    without reaching into adjacent patches. The centroid of the gradient
    magnitude in that window is at the geometric centre of the ring by
    symmetry, regardless of patch colour. This works for the row-4
    neutrals too (white / light-grey) because the edge contrast against
    the white border survives even for the lightest patch.

    Validity gates (in order):
      1. Window fully inside the frame.
      2. Peak gradient ≥ 25 (rejects flat regions and the lightest neutral).
      3. Ring-vs-interior gradient density ≥ 3× (rejects random texture
         like concrete cracks — a real patch has strong gradients only
         on the perimeter, while uniform texture distributes gradients
         throughout the window).
      4. Centroid offset ≤ ½ × half_win (rejects projections too far
         off — the centroid window can't recover larger shifts).

    Gate (3) is what stops a wrong IPPE pose from latching onto bogus
    patches in the background: random texture passes (1) and (2) but
    fails (3), so the wrong-pose candidate ends up with very few valid
    patches and loses the IPPE-tiebreaker in `_solve_target_pose`.

    Returns
    -------
    refined : (N, 2) float64 image-space centres
    valid   : (N,)   bool mask — True where refinement succeeded
    """
    h, w = gray.shape[:2]
    half_win = max(6, int(round(3.5 * px_per_mm)))
    refined = initial_proj.astype(np.float64).copy()
    valid   = np.zeros(len(initial_proj), dtype=bool)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = cv2.magnitude(gx, gy)

    n = 2 * half_win + 1
    coords = np.arange(n, dtype=np.float64) - half_win
    cx_grid, cy_grid = np.meshgrid(coords, coords)

    # Pre-compute interior / ring masks centred on the window. The interior
    # box is the patch's solid colour region (low gradient); the ring is
    # the perimeter where the patch meets the white border (high gradient).
    abs_x = np.abs(cx_grid)
    abs_y = np.abs(cy_grid)
    interior_half_px = 2.0 * px_per_mm   # mm — well inside the 3 mm patch half
    ring_inner_px    = 2.5 * px_per_mm   # ring band: 2.5 mm to 3.5 mm
    ring_outer_px    = 3.5 * px_per_mm
    interior_mask = (abs_x < interior_half_px) & (abs_y < interior_half_px)
    in_outer      = (abs_x < ring_outer_px)    & (abs_y < ring_outer_px)
    in_inner      = (abs_x < ring_inner_px)    & (abs_y < ring_inner_px)
    ring_mask     = in_outer & ~in_inner
    interior_area = max(1, int(interior_mask.sum()))
    ring_area     = max(1, int(ring_mask.sum()))

    for i in range(len(initial_proj)):
        px, py = float(initial_proj[i, 0]), float(initial_proj[i, 1])
        ix, iy = int(round(px)), int(round(py))
        x1 = ix - half_win
        y1 = iy - half_win
        x2 = ix + half_win + 1
        y2 = iy + half_win + 1
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            continue

        win = gmag[y1:y2, x1:x2]
        peak = float(win.max())
        if peak < 25.0:
            continue

        interior_density = float(win[interior_mask].sum()) / interior_area
        ring_density     = float(win[ring_mask].sum())     / ring_area
        if ring_density < 3.0 * (interior_density + 1.0):
            continue   # window doesn't contain a square-ring boundary

        # Threshold suppresses weak in-patch texture and bright specular
        # highlights; only the strong perimeter ring is kept.
        thresh  = 0.35 * peak
        weights = np.where(win > thresh, win - thresh, 0.0)
        total   = float(weights.sum())
        if total <= 0.0:
            continue

        ox = float((weights * cx_grid).sum() / total)
        oy = float((weights * cy_grid).sum() / total)

        # Reject patches whose initial projection was too far off — the
        # centroid window can't recover a shift larger than half its size.
        if abs(ox) > 0.5 * half_win or abs(oy) > 0.5 * half_win:
            continue

        refined[i, 0] = ix + ox
        refined[i, 1] = iy + oy
        valid[i] = True

    return refined, valid


def _solve_target_pose(
    gray: np.ndarray,
    obj_marker: np.ndarray,    # (8, 3) marker corner world coords (mm)
    img_marker: np.ndarray,    # (8, 2) marker corner image coords (px)
    K: np.ndarray,
    dist: np.ndarray,
    px_per_mm: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Planar pose for the dual-marker / 24-patch target with IPPE
    ambiguity resolution via patch refinement.

    IPPE returns up to two physically-distinct ambiguous solutions for a
    planar target. For a small near-perpendicular thin target like ours
    both solutions reproject the 8 marker corners with virtually
    identical sub-pixel error — picking by reprojection error on markers
    alone is essentially noise-driven, and on certain frames the
    mirror-twin pose wins. That pose has the surface normal flipped,
    giving wildly wrong tilt readings (e.g. 100°+ when the truth is < 1°).

    Resolution: run the full refinement pipeline against EACH IPPE
    candidate — project the 24 patch centres, refine to sub-pixel via
    gradient-magnitude centroid (which itself rejects random background
    texture, see `_refine_patch_centers`), and score the candidate by
    (n_valid_patches, −reprojection_error_after_LM_refine). The wrong
    pose projects patches into background where almost nothing refines
    successfully; the correct pose lands them on the printed grid and
    most refine. Score tuple compares lexicographically: more valid
    patches always wins, ties broken by lower post-refine error.

    Stage-2 widens the pose baseline from 10 × 24 mm (markers alone) to
    ~52 × 24 mm (markers + 24 patches), giving ~5× better tilt accuracy
    around the vertical axis.

    If neither candidate finds ≥ 6 valid patches, fall back to the
    marker-only pose with the lowest reprojection error among
    physically-valid candidates. If nothing is physically valid, return
    the first IPPE solution (downstream logic flags it as bad tilt).

    Returns (rvec, tvec, n_pose_points) or None on PnP failure.
    """
    obj_marker_f = obj_marker.astype(np.float64)
    img_marker_f = img_marker.astype(np.float64)

    try:
        retval, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            obj_marker_f, img_marker_f, K, dist, flags=cv2.SOLVEPNP_IPPE,
        )
    except Exception:
        return None
    if not retval:
        return None

    # Filter to physically-valid candidates (printed face toward camera).
    candidates: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(retval):
        Ri, _ = cv2.Rodrigues(rvecs[i])
        if float(Ri[2, 2]) < 0.0:
            candidates.append((rvecs[i].copy(), tvecs[i].copy()))
    if not candidates:
        # Nothing physically valid — return first IPPE solution and let
        # the caller's tilt threshold flag it.
        return rvecs[0].copy(), tvecs[0].copy(), len(obj_marker_f)

    n_marker_pts = len(obj_marker_f)
    patch_world  = _PATCH_WORLD.astype(np.float64)

    # Score each candidate by (n_valid_patches, −post_refine_error).
    # Tuple ordering means more valid patches always wins; equal counts
    # are broken by lower reprojection error.
    best_score: Optional[Tuple[int, float]] = None
    best_pose:  Optional[Tuple[np.ndarray, np.ndarray, int]] = None

    for rvec_c, tvec_c in candidates:
        # Marker-only reprojection error (used as a low-priority fallback
        # score when no patches refine for any candidate).
        proj_m, _ = cv2.projectPoints(obj_marker_f, rvec_c, tvec_c, K, dist)
        marker_err = float(np.linalg.norm(
            proj_m.reshape(-1, 2) - img_marker_f, axis=1,
        ).mean())

        n_valid = 0
        rvec_out, tvec_out = rvec_c, tvec_c
        n_pts_out = n_marker_pts
        score_err = marker_err

        if px_per_mm > 0.0:
            proj_p, _ = cv2.projectPoints(patch_world, rvec_c, tvec_c, K, dist)
            proj_p = proj_p.reshape(-1, 2)
            refined, valid = _refine_patch_centers(gray, proj_p, px_per_mm)
            n_valid = int(valid.sum())

            if n_valid >= 6:
                obj_all = np.vstack([obj_marker_f, patch_world[valid]])
                img_all = np.vstack([img_marker_f, refined[valid]])
                rvec_r, tvec_r = rvec_c.copy(), tvec_c.copy()
                try:
                    rvec_r, tvec_r = cv2.solvePnPRefineLM(
                        obj_all, img_all, K, dist, rvec_r, tvec_r,
                    )
                except Exception:
                    pass
                proj_r, _ = cv2.projectPoints(obj_all, rvec_r, tvec_r, K, dist)
                score_err = float(np.linalg.norm(
                    proj_r.reshape(-1, 2) - img_all, axis=1,
                ).mean())
                rvec_out, tvec_out = rvec_r, tvec_r
                n_pts_out = len(obj_all)

        score = (n_valid, -score_err)
        if best_score is None or score > best_score:
            best_score = score
            best_pose  = (rvec_out, tvec_out, n_pts_out)

    return best_pose


# ─────────────────────────────────────────────────────────────────────────────
# Metrics dict schema
# ─────────────────────────────────────────────────────────────────────────────
# Keys emitted via metrics_ready every frame:
#
#   n_markers          int            0, 1, or 2
#   detected           bool           True iff n_markers == 2 (backward compat)
#   approx_intrinsics  bool
#   px_per_mm          float | None
#   mm_per_px          float | None
#   tilt_deg           float | None   single-frame total tilt
#   tilt_x_deg         float | None   single-frame X component
#   tilt_y_deg         float | None   single-frame Y component
#   tilt_avg_deg       float | None   rolling-average total tilt
#   tilt_x_avg         float | None   rolling-average X
#   tilt_y_avg         float | None   rolling-average Y
#   tilt_status        str            "ok" | "warn" | "bad" | "unknown"
#   n_samples          int            frames in tilt window
#   capture_crop_tl    (int,int) | None   capture_crop in full-frame pixels
#   capture_crop_br    (int,int) | None
#   entry_zone_y       int | None     pixel y of entry-zone bottom
#   color_frames       int            frames in colour window (0–COLOR_WINDOW)
#   color_sampled      bool           True once matrix has been computed
#   color_rms          float | None   fit RMS % (lower is better)

def _empty_metrics() -> Dict:
    return dict(
        n_markers=0,
        detected=False,
        approx_intrinsics=True,
        px_per_mm=None, mm_per_px=None,
        tilt_deg=None, tilt_x_deg=None, tilt_y_deg=None,
        tilt_avg_deg=None, tilt_x_avg=None, tilt_y_avg=None,
        tilt_status="unknown", n_samples=0,
        capture_crop_tl=None, capture_crop_br=None, entry_zone_y=None,
        color_frames=0, color_sampled=False, color_rms=None,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Worker thread
# ══════════════════════════════════════════════════════════════════════════════

class CameraSetupWorker(QThread):
    """
    QThread that drives the OAK camera and emits per-frame metrics.

    Public attributes (read by SetupDialog._on_save once worker is stable)
    -----------------------------------------------------------------------
    color_matrix : np.ndarray (3, 3) or None
        Maps [R_lin, G_lin, B_lin] → CIE XYZ D65.
    color_bias : np.ndarray (3,) or None
        Additive XYZ offset.

    Signals
    -------
    frame_ready(QImage)
    metrics_ready(dict)
    log_message(str)
    error_occurred(str)
    worker_stopped()
    """

    frame_ready    = pyqtSignal(QImage)
    metrics_ready  = pyqtSignal(dict)
    log_message    = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    worker_stopped = pyqtSignal()

    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._stop_requested = False
        self._tilt_window:  Deque[Tuple[float, float]] = deque(maxlen=TILT_WINDOW)
        self._color_window: Deque[np.ndarray]          = deque(maxlen=COLOR_WINDOW)
        # Public colour calibration result (None until first stable computation)
        self.color_matrix: Optional[np.ndarray] = None
        self.color_bias:   Optional[np.ndarray] = None
        self._color_rms:   Optional[float] = None
        # Device factory lens intrinsics (set in run(); read by
        # SetupDialog._on_save to persist for measurement-path undistortion).
        self.camera_K:    Optional[np.ndarray] = None
        self.camera_dist: Optional[np.ndarray] = None
        self.intrinsics_approx: bool = True

    def request_stop(self):
        self._stop_requested = True

    # ── Thread entry point ────────────────────────────────────────────────────

    def run(self):
        cfg       = self.cfg
        cam_cfg   = cfg.get("camera", {})
        cap_w     = cam_cfg.get("width",  1280)
        cap_h     = cam_cfg.get("height",  800)
        fps       = min(cam_cfg.get("fps", 30), 15)
        device_ip = cfg.get("device_ip", "")

        try:
            device_info = dai.DeviceInfo(device_ip)
        except Exception as exc:
            self.error_occurred.emit(f"Invalid device IP '{device_ip}': {exc}")
            self.worker_stopped.emit()
            return

        K, dist, approx = _load_intrinsics_from_device(device_info, cam_cfg, cap_w, cap_h)
        self.camera_K = K
        self.camera_dist = dist
        self.intrinsics_approx = approx
        if approx:
            self.log_message.emit(
                "WARNING: Could not read device calibration — using approximate "
                "intrinsics.  Tilt readings may have 2–3° systematic error.  "
                "Calibration cannot be saved while approximate intrinsics are active."
            )
        else:
            self.log_message.emit(
                "Device calibration loaded.  Sub-degree tilt accuracy available."
            )

        detector = _make_detector()
        self._tilt_window.clear()
        self._color_window.clear()

        try:
            with dai.Pipeline(dai.Device(device_info)) as pipeline:
                camera = pipeline.create(dai.node.Camera)
                camera.initialControl.setAutoFocusMode(
                    dai.CameraControl.AutoFocusMode.OFF
                )
                camera.initialControl.setManualFocus(cam_cfg.get("manual_focus", 154))
                camera.initialControl.setManualExposure(
                    cam_cfg.get("exposure_us", 700),
                    cam_cfg.get("iso", 1000),
                )
                camera = camera.build(dai.CameraBoardSocket.CAM_A)
                cam_out = camera.requestOutput(
                    size=(cap_w, cap_h),
                    type=dai.ImgFrame.Type.BGR888i,
                    fps=fps,
                )
                frame_q = cam_out.createOutputQueue(maxSize=2, blocking=False)
                pipeline.start()
                self.log_message.emit(
                    "Camera connected.  Place the calibration target flat in view."
                )

                rotate_180 = bool(cam_cfg.get("rotate_180", False))

                while pipeline.isRunning() and not self._stop_requested:
                    if frame_q.has():
                        bgr = frame_q.get().getCvFrame()
                        if rotate_180:
                            bgr = cv2.flip(bgr, -1)
                        metrics, annotated = self._process_frame(
                            bgr, detector, K, dist, approx
                        )
                        self.frame_ready.emit(_to_qimage(annotated))
                        self.metrics_ready.emit(metrics)
                    else:
                        time.sleep(0.030)

        except Exception as exc:
            self.error_occurred.emit(str(exc))
        finally:
            self.worker_stopped.emit()

    # ── Per-frame processing ──────────────────────────────────────────────────

    def _process_frame(
        self,
        bgr: np.ndarray,
        detector: Any,
        K: np.ndarray,
        dist: np.ndarray,
        approx_intrinsics: bool,
    ) -> Tuple[Dict, np.ndarray]:
        frame_h, frame_w = bgr.shape[:2]
        out = bgr.copy()
        m = _empty_metrics()
        m["approx_intrinsics"] = approx_intrinsics

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners_all, ids_all, _ = _detect_markers(detector, gray)

        # ── Collect target marker IDs (0 and 1 only) ──────────────────────
        id_to_pts: Dict[int, np.ndarray] = {}
        if ids_all is not None:
            for idx, mid in enumerate(ids_all.flatten()):
                if int(mid) in (0, 1):
                    id_to_pts[int(mid)] = (
                        corners_all[idx].reshape(4, 2).astype(np.float64)
                    )

        n_found = len(id_to_pts)
        m["n_markers"] = n_found
        m["detected"]  = (n_found == 2)

        if n_found == 0:
            self._tilt_window.clear()
            self._color_window.clear()
            _put_text(out, "Place calibration target in view", (20, 50), 0.9, _GREY)
            if approx_intrinsics:
                _put_text(
                    out, "! APPROX INTRINSICS — tilt unreliable",
                    (20, out.shape[0] - 15), 0.55, _ORANGE,
                )
            return m, out

        # Lose one marker → clear colour window (can't project patches)
        if n_found < 2:
            self._color_window.clear()

        # ── px/mm ─────────────────────────────────────────────────────────
        all_sides: List[float] = []
        for pts in id_to_pts.values():
            all_sides += [
                float(np.linalg.norm(pts[(i + 1) % 4] - pts[i]))
                for i in range(4)
            ]
        px_mm_sides = float(np.mean(all_sides)) / MARKER_SIZE_MM

        if n_found == 2:
            c0 = id_to_pts[0].mean(axis=0)
            c1 = id_to_pts[1].mean(axis=0)
            px_mm_cc = float(np.linalg.norm(c0 - c1)) / MARKER_C_C_MM
            # 8 side measurements + C-C baseline (weighted 4× for longer span)
            px_per_mm = (px_mm_sides * 8.0 + px_mm_cc * 4.0) / 12.0
        else:
            px_per_mm = px_mm_sides

        m["px_per_mm"] = px_per_mm
        m["mm_per_px"] = 1.0 / px_per_mm if px_per_mm > 0 else None

        # ── Pose estimation ───────────────────────────────────────────────
        # Tilt is computed only when both markers are visible. A single
        # 10 mm marker gives a pose far too noisy to drive the rolling
        # average — better to let tilt go silent than to pollute the
        # window with single-marker readings.
        rvec = tvec = None

        if n_found == 2:
            # Pair the visually-higher marker with the +Y world-coord block.
            # This makes pose orientation invariant to which physical marker
            # (ID 0 or 1) ends up at the image top.
            top_id = 0 if id_to_pts[0].mean(0)[1] < id_to_pts[1].mean(0)[1] else 1
            bot_id = 1 - top_id
            obj_pts = np.vstack([_OBJ_TOP, _OBJ_BOT])
            img_pts = np.vstack([id_to_pts[top_id], id_to_pts[bot_id]])

            pose_result = _solve_target_pose(
                gray, obj_pts, img_pts, K, dist, px_per_mm,
            )
            if pose_result is not None:
                rvec, tvec, _n_pose_pts = pose_result

        if rvec is not None:
            R, _   = cv2.Rodrigues(rvec)
            normal = R[:, 2]
            ref    = float(-normal[2])   # ≈ 1 when perpendicular
            tilt_x = float(np.degrees(np.arctan2(float(normal[0]), max(ref, 1e-9))))
            tilt_y = float(np.degrees(np.arctan2(float(normal[1]), max(ref, 1e-9))))
            cos_t  = abs(float(normal[2]))
            tilt_t = float(np.degrees(np.arccos(np.clip(cos_t, 0., 1.))))

            m["tilt_deg"]   = tilt_t
            m["tilt_x_deg"] = tilt_x
            m["tilt_y_deg"] = tilt_y

            # Rolling average — clear on repositioning jump > 5°
            if self._tilt_window:
                lx, ly = self._tilt_window[-1]
                if abs(tilt_x - lx) + abs(tilt_y - ly) > 5.0:
                    self._tilt_window.clear()
                    self._color_window.clear()

            self._tilt_window.append((tilt_x, tilt_y))
            n_samp = len(self._tilt_window)
            m["n_samples"] = n_samp

            if n_samp >= 3:
                arr   = np.array(self._tilt_window)
                avg_x = float(np.mean(arr[:, 0]))
                avg_y = float(np.mean(arr[:, 1]))
                m["tilt_x_avg"]   = avg_x
                m["tilt_y_avg"]   = avg_y
                m["tilt_avg_deg"] = float(np.sqrt(avg_x ** 2 + avg_y ** 2))
                tilt_chk = m["tilt_avg_deg"]
            else:
                tilt_chk = tilt_t

            m["tilt_status"] = (
                "ok"   if tilt_chk < TILT_OK_DEG  else
                "warn" if tilt_chk < TILT_WARN_DEG else
                "bad"
            )
        else:
            self._tilt_window.clear()
            self._color_window.clear()

        # ── capture_crop overlay (read live from cfg) ─────────────────────
        # The user can update capture_crop interactively via "Set Capture
        # Crop" in the dialog — those edits land in self.cfg["capture_crop"]
        # in place, and we redraw the rectangle on every frame.
        cap = self.cfg.get("capture_crop") or {}
        try:
            cap_x_min = float(cap.get("x_min", 0.0))
            cap_x_max = float(cap.get("x_max", 1.0))
            cap_y_min = float(cap.get("y_min", 0.0))
            cap_y_max = float(cap.get("y_max", 1.0))
            ctl = (int(cap_x_min * frame_w), int(cap_y_min * frame_h))
            cbr = (int(cap_x_max * frame_w), int(cap_y_max * frame_h))
            m["capture_crop_tl"] = ctl
            m["capture_crop_br"] = cbr
            # entry-zone band runs along the top of the crop
            entry_rel = float(self.cfg.get("entry_zone_threshold_rel", 0.15))
            crop_h_px = max(1, cbr[1] - ctl[1])
            m["entry_zone_y"]  = int(ctl[1] + entry_rel * crop_h_px)
        except Exception:
            pass

        # ── Colour patch sampling ─────────────────────────────────────────
        # Accumulate colour frames whenever both markers are detected and
        # a pose is available — the actual pose handles target tilt
        # transparently in the projection step, so the tilt magnitude is
        # irrelevant. Camera-to-surface tilt verification is handled in
        # the IMU mount setup dialog, not here.
        pose_ready = (n_found == 2 and rvec is not None and px_per_mm > 0)
        if pose_ready:
            half_px = PATCH_SAMPLE_MM * px_per_mm
            patch_obs = _sample_patch_colors(
                bgr, _PATCH_WORLD, rvec, tvec, K, dist,
                half_px, frame_w, frame_h,
            )
            if patch_obs is not None:
                self._color_window.append(patch_obs)

        nw = len(self._color_window)
        m["color_frames"] = nw

        if nw >= COLOR_WINDOW:
            avg_bgr = np.mean(np.stack(list(self._color_window)), axis=0)
            M, bias, rms = _compute_color_matrix(avg_bgr, _PATCH_REF_SRGB)
            self.color_matrix = M
            self.color_bias   = bias
            self._color_rms   = rms
            m["color_sampled"] = True
            m["color_rms"]     = rms
        elif self.color_matrix is not None:
            # Report last good result while window is being refreshed
            m["color_sampled"] = True
            m["color_rms"]     = self._color_rms

        # Project patch centres for overlay
        proj_pts: Optional[np.ndarray] = None
        if rvec is not None and n_found == 2:
            try:
                pp, _ = cv2.projectPoints(
                    _PATCH_WORLD.astype(np.float32), rvec, tvec, K, dist
                )
                proj_pts = pp.reshape(-1, 2)
            except Exception:
                pass

        _draw_overlays(out, id_to_pts, m, rvec, tvec, K, dist, proj_pts, px_per_mm)
        return m, out


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_overlays(
    out: np.ndarray,
    id_to_pts: Dict[int, np.ndarray],
    m: Dict,
    rvec: Optional[np.ndarray],
    tvec: Optional[np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
    proj_patch_pts: Optional[np.ndarray],
    px_per_mm: float,
) -> None:
    # ── Marker outlines ────────────────────────────────────────────────────
    for pts in id_to_pts.values():
        cv2.polylines(
            out, [pts.astype(np.int32)], isClosed=True, color=_GREEN, thickness=2
        )

    # ── Pose axes at world origin (marker midpoint) ────────────────────────
    if rvec is not None and tvec is not None:
        try:
            cv2.drawFrameAxes(out, K, dist, rvec, tvec, AXIS_LENGTH_MM)
        except Exception:
            pass

    # ── capture_crop rectangle + entry zone ───────────────────────────────
    ctl = m.get("capture_crop_tl")
    cbr = m.get("capture_crop_br")
    if ctl is not None and cbr is not None:
        cv2.rectangle(out, ctl, cbr, _GREEN, 2)
        _put_text(
            out, "capture_crop",
            (ctl[0] + 4, ctl[1] - 6), 0.50, _GREEN,
        )
        ez_y = m.get("entry_zone_y")
        if ez_y is not None:
            # Solid orange line across the crop at the entry-zone boundary;
            # everything above this line is the "skip" band.
            cv2.line(out, (ctl[0], ez_y), (cbr[0], ez_y), _ORANGE, 1)
            _put_text(
                out, "entry zone",
                (ctl[0] + 4, ez_y - 4), 0.45, _ORANGE,
            )

    # ── Colour patch dots ──────────────────────────────────────────────────
    # Draw small filled circles at each projected patch centre, coloured
    # with the reference sRGB value so the user can see alignment.
    if proj_patch_pts is not None:
        dot_r = max(2, int(round(px_per_mm * 1.5)))   # ~1.5 mm radius
        for i, (px, py) in enumerate(proj_patch_pts):
            ref = _PATCH_REF_SRGB[i]
            dot_bgr = (int(ref[2]), int(ref[1]), int(ref[0]))   # RGB → BGR
            cv2.circle(out, (int(round(px)), int(round(py))), dot_r,     dot_bgr, -1)
            cv2.circle(out, (int(round(px)), int(round(py))), dot_r + 1, (0, 0, 0), 1)

    # ── px/mm ──────────────────────────────────────────────────────────────
    if m.get("px_per_mm") is not None:
        _put_text(
            out,
            f"px/mm: {m['px_per_mm']:.4f}   mm/px: {m['mm_per_px']:.5f}",
            (20, 40), 0.60, _WHITE,
        )

    # ── Marker count ───────────────────────────────────────────────────────
    nc = m.get("n_markers", 0)
    mc = _GREEN if nc == 2 else (_ORANGE if nc == 1 else _RED)
    _put_text(out, f"Markers: {nc}/2", (20, 66), 0.60, mc)

    # ── Colour status ──────────────────────────────────────────────────────
    nw = m.get("color_frames", 0)
    if m.get("color_sampled"):
        rms = m.get("color_rms") or 0.0
        _put_text(out, f"Colour: OK  RMS {rms:.2f}%", (20, 88), 0.60, _GREEN)
    elif nw > 0:
        _put_text(
            out, f"Colour: sampling {nw}/{COLOR_WINDOW}",
            (20, 88), 0.60, _ORANGE,
        )


def _put_text(img, text, org, scale, color):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,   1, cv2.LINE_AA)


def _to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()


# ─────────────────────────────────────────────────────────────────────────────
# Intrinsics loader (unchanged from previous version)
# ─────────────────────────────────────────────────────────────────────────────

def _load_intrinsics_from_device(
    device_info: dai.DeviceInfo,
    cam_cfg: dict,
    cap_w: int,
    cap_h: int,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Open the device, read factory camera intrinsics, close it, sleep 0.5 s.
    Returns (K, dist, approx_used).
    """
    try:
        device = dai.Device(device_info)
        calib  = device.readCalibration()
        K_raw  = calib.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A, cap_w, cap_h
        )
        D_raw  = calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A)
        device.close()
        time.sleep(0.5)
        return np.array(K_raw, dtype=np.float64), np.array(D_raw, dtype=np.float64), False
    except Exception:
        focal = float(cam_cfg.get("focal_length_px", cap_w))
        K = np.array(
            [[focal,  0.,   cap_w / 2.],
             [0.,   focal,  cap_h / 2.],
             [0.,    0.,           1.]],
            dtype=np.float64,
        )
        return K, np.zeros(5, dtype=np.float64), True
