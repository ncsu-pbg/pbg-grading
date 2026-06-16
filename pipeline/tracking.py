"""
Frame-by-frame distance + IoU object tracker.

Ported from latest_tracker_class.py for real-time use.  Operates on
per-frame detection dicts rather than a CSV, but uses identical logic
so offline and real-time results are directly comparable.

Two matching back-ends are supported:

  * ``matching="greedy"`` (default, legacy) -- iterates over existing
    tracks in dict-insertion order and greedily picks the closest
    below-and-within-threshold detection.  This is the behaviour the
    on-camera production pipeline currently uses.

  * ``matching="hungarian"`` -- builds a velocity-aware cost matrix
    over (tracks x detections) and lets ``scipy.optimize.linear_sum_assignment``
    pick a globally optimum one-to-one match.  Each track carries an
    EMA ``dy`` estimate so a stuck track (dy~=0) won't steal a fast-moving
    detection that better belongs to a faster track.  Requires scipy at
    runtime; falls back to greedy with a warning if scipy isn't available.

Track ageing (``max_age``)
--------------------------
With ``max_age=0`` (default) an unmatched track is dropped immediately --
this is the legacy on-camera behaviour and is byte-for-byte unchanged.
With ``max_age=N`` an unmatched track is *coasted* for up to N frames:
its predicted centre is advanced by its EMA ``dy`` each frame and it
remains eligible to re-match (e.g. across a one-frame NN drop or a brief
occlusion behind the chute wall).  Newly-born tracks inherit the median
``dy`` of currently-active tracks so they "expect" downward motion from
their first frame instead of expecting to stay still.

Width validity (``width_valid``)
---------------------------------
Detections may carry an optional ``width_valid`` flag.  When False, the
per-frame width measurement is excluded from the track's min-width
aggregate -- used to suppress measurements taken when the mask was
edge-clipped or occluded by another pod.  Defaults to True so existing
callers are unaffected.  Note: ``width_valid`` affects *only* the width
aggregate; it never influences matching or track creation.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _calculate_iou(box1: Tuple, box2: Tuple) -> float:
    """IoU between two (x1, y1, x2, y2) boxes."""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    xi = max(x1, x3)
    yi = max(y1, y3)
    iw = max(0.0, min(x2, x4) - xi)
    ih = max(0.0, min(y2, y4) - yi)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a2 = max(0.0, x4 - x3) * max(0.0, y4 - y3)
    denom = a1 + a2 - inter
    return float(inter / denom) if denom > 0 else 0.0


def _median(values: List[float]) -> float:
    """Median of a list; 0.0 for an empty list."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


# ═══════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TrackMeasurement:
    """Aggregate morphological stats for one tracked pod."""
    track_id: int
    length: float = 0.0   # crop-pixel length at frame of max area
    width: float = 0.0    # crop-pixel min width across all *valid* frames
    area: float = 0.0     # crop-pixel max mask area
    mean_bgr: Optional[Tuple[float, float, float]] = None
    # mean_bgr: mean [B, G, R] over the segmentation mask at the frame of
    # max area.  Populated only when the pipeline passes raw BGR frames to
    # the tracking node alongside detections (future wiring in host_node.py /
    # oak_pipeline.py).  Until then this field remains None and the hunter_l
    # column in exported CSVs will be empty.

    # Width-quality bookkeeping.  ``width`` above is the min over frames
    # where width_valid=True.  If a track has zero such frames we fall
    # back to the min width seen on *invalid* frames so the CSV row still
    # carries a number, and ``width_quality`` lets downstream filtering
    # spot those rows.
    width_quality: str = "clean"          # 'clean' | 'fallback'
    _width_invalid_min: float = field(default=float("inf"), repr=False)
    _width_valid_count: int = field(default=0, repr=False)


# ═══════════════════════════════════════════════════════════════════════
# Tracker
# ═══════════════════════════════════════════════════════════════════════

class RealtimeTracker:
    """
    Frame-by-frame distance + IoU tracker.

    Call ``update(detections)`` once per frame with a list of dicts
    containing ``cx, cy, length, width, area`` (and optionally
    ``width_valid``).  Accumulated per-track max-area / min-width stats
    live in ``self.max_dims``.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        distance_threshold: float = 300.0,
        matching: str = "greedy",
        max_age: int = 0,
    ):
        self.iou_threshold = float(iou_threshold)
        self.distance_threshold = float(distance_threshold)
        if matching not in ("greedy", "hungarian"):
            raise ValueError(
                f"matching must be 'greedy' or 'hungarian', got {matching!r}"
            )
        self.matching = matching
        self.max_age = max(0, int(max_age))

        self._frame_count: int = 0
        self._next_track_id: int = 0
        self._tracking_objects: Dict[int, Tuple[int, int]] = {}
        self._track_dy: Dict[int, float] = {}     # EMA of per-frame dy
        self._track_age: Dict[int, int] = {}      # frames since last match
        self._prev_centers: List[Tuple[int, int]] = []
        self.max_dims: Dict[int, TrackMeasurement] = {}

        # Lazy scipy probe so 'greedy' callers don't pay any cost.
        self._scipy_assign = None
        self._scipy_checked = False

    # EMA factor for per-track dy.  Closer to 1.0 -> more responsive to
    # recent motion; closer to 0.0 -> smoother but slower to track
    # acceleration.  0.5 is a reasonable middle.
    _DY_EMA_ALPHA = 0.5

    # Hungarian backward-motion tolerance (px).  Pods travel down the
    # chute, but a stuck or slowing pod jitters a few px upward between
    # frames; forbidding ALL backward motion makes such a pod re-ID on
    # every jitter.  We forbid only backward motion larger than this,
    # mirroring the greedy matcher's IoU "same-position" rescue.
    _BACK_TOL_PX = 12.0

    @property
    def total_unique(self) -> int:
        """Number of unique pods tracked so far."""
        return len(self.max_dims)

    @property
    def active_count(self) -> int:
        """Number of currently active tracks (includes coasting tracks)."""
        return len(self._tracking_objects)

    def reset(self):
        """Clear all tracking state for a fresh run."""
        self._frame_count = 0
        self._next_track_id = 0
        self._tracking_objects.clear()
        self._track_dy.clear()
        self._track_age.clear()
        self._prev_centers.clear()
        self.max_dims.clear()

    # ── Track lifecycle helpers ──────────────────────────────────────
    def _update_dy(self, tid: int, prev_pt: Tuple[int, int],
                   new_pt: Tuple[int, int]) -> None:
        dy_new = float(new_pt[1] - prev_pt[1])
        prior = self._track_dy.get(tid, dy_new)
        self._track_dy[tid] = (
            self._DY_EMA_ALPHA * dy_new + (1.0 - self._DY_EMA_ALPHA) * prior
        )

    def _bootstrap_dy(self) -> float:
        """
        Median dy of currently-active tracks, used to initialise a
        newborn track so it expects downward motion from frame 1.
        Considers only tracks with a non-trivial dy so a crowd of
        just-born (dy=0) tracks doesn't drag the estimate to zero.
        """
        moving = [v for v in self._track_dy.values() if abs(v) > 1e-6]
        return _median(moving)

    def _age_or_drop(self, tid: int) -> None:
        """
        Handle a track that found no match this frame.  Drops it once it
        has gone unmatched for more than ``max_age`` frames; otherwise
        coasts its predicted centre forward by its EMA dy.

        With max_age=0 this always drops on the first miss -- identical
        to the legacy on-camera behaviour.
        """
        age = self._track_age.get(tid, 0) + 1
        if age > self.max_age:
            self._tracking_objects.pop(tid, None)
            self._track_dy.pop(tid, None)
            self._track_age.pop(tid, None)
        else:
            self._track_age[tid] = age
            cx, cy = self._tracking_objects[tid]
            dy = self._track_dy.get(tid, 0.0)
            self._tracking_objects[tid] = (int(cx), int(round(cy + dy)))

    def _spawn_track(self, pt: Tuple[int, int], dy_seed: float) -> None:
        tid = self._next_track_id
        self._tracking_objects[tid] = pt
        self._track_dy[tid] = dy_seed
        self._track_age[tid] = 0
        self._next_track_id += 1

    # ── Hungarian matcher ────────────────────────────────────────────
    def _get_scipy_assign(self):
        if self._scipy_checked:
            return self._scipy_assign
        self._scipy_checked = True
        try:
            from scipy.optimize import linear_sum_assignment
            self._scipy_assign = linear_sum_assignment
        except ImportError:
            print(
                "[tracker] WARN: scipy not available, falling back to "
                "greedy matching.  pip install scipy to enable hungarian."
            )
            self._scipy_assign = None
        return self._scipy_assign

    def _match_hungarian(
        self,
        cur_centers: List[Tuple[int, int]],
    ):
        """
        Returns (assignments, lost_track_ids, unmatched_centers), or
        (None, None, None) to signal the caller to fall back to greedy.

        Cost = |predicted_y - cand_y| + |pt2_x - cand_x|, with
        predicted_y = pt2_y + ema_dy(track).  Pairs whose Euclidean
        distance from the *last known* center exceeds distance_threshold,
        or which would require upward motion, are forbidden (inf cost).
        """
        assign_fn = self._get_scipy_assign()
        if assign_fn is None:
            return None, None, None     # signal caller to use greedy

        track_ids = list(self._tracking_objects.keys())
        if not track_ids or not cur_centers:
            return {}, list(track_ids), list(cur_centers)

        INF = 1e9
        n_t = len(track_ids)
        n_d = len(cur_centers)
        cost = [[INF] * n_d for _ in range(n_t)]

        for ti, tid in enumerate(track_ids):
            tx, ty = self._tracking_objects[tid]
            pred_y = ty + self._track_dy.get(tid, 0.0)
            for di, (cx, cy) in enumerate(cur_centers):
                # Forbid only *significant* backward motion; small upward
                # jitter from a stuck/slowing pod must still match.
                if cy < ty - self._BACK_TOL_PX:
                    continue
                # Hard gate on raw distance.
                if math.hypot(cx - tx, cy - ty) >= self.distance_threshold:
                    continue
                cost[ti][di] = abs(pred_y - cy) + abs(tx - cx)

        try:
            row_ind, col_ind = assign_fn(cost)
        except Exception as exc:
            print(f"[tracker] WARN: hungarian assignment failed ({exc}); "
                  f"falling back to greedy this frame.")
            return None, None, None

        assignments: Dict[int, Tuple[int, int]] = {}
        matched_track_idxs = set()
        matched_det_idxs = set()
        for ri, ci in zip(row_ind, col_ind):
            if cost[ri][ci] >= INF:
                continue
            assignments[track_ids[ri]] = cur_centers[ci]
            matched_track_idxs.add(ri)
            matched_det_idxs.add(ci)

        lost = [track_ids[i] for i in range(n_t) if i not in matched_track_idxs]
        unmatched = [cur_centers[i] for i in range(n_d)
                     if i not in matched_det_idxs]
        return assignments, lost, unmatched

    # ── Main entry point ─────────────────────────────────────────────
    def update(self, detections: List[dict]) -> Dict[int, Tuple[int, int]]:
        """
        Process one frame of detections.

        Parameters
        ----------
        detections : list of dict
            Each dict must have keys: cx, cy (int), length, width, area (float).
            Optional: ``width_valid`` (bool, default True), ``mean_bgr``.

        Returns
        -------
        dict
            Current ``{track_id: (cx, cy)}`` mapping.
        """
        cur_centers: List[Tuple[int, int]] = []
        center_to_dims: Dict[Tuple[int, int], dict] = {}

        for d in detections:
            c = (d["cx"], d["cy"])
            cur_centers.append(c)
            center_to_dims[c] = {
                "length":      d["length"],
                "width":       d["width"],
                "area":        d["area"],
                "width_valid": d.get("width_valid", True),
                "mean_bgr":    d.get("mean_bgr"),
            }

        self._frame_count += 1

        # ── Association ──────────────────────────────────────────────
        if self._frame_count <= 2:
            # Bootstrap: seed tracks from any pair of close-by centers
            # across the first two frames (unchanged legacy behaviour).
            for pt in cur_centers:
                for pt2 in self._prev_centers:
                    if math.hypot(pt2[0] - pt[0], pt2[1] - pt[1]) < 200:
                        self._spawn_track(pt, 0.0)
        elif self.matching == "hungarian":
            assignments, lost, unmatched = self._match_hungarian(cur_centers)
            if assignments is None:    # scipy unavailable; fall back
                self._match_greedy(cur_centers)
            else:
                for tid, pt in assignments.items():
                    self._update_dy(tid, self._tracking_objects[tid], pt)
                    self._tracking_objects[tid] = pt
                    self._track_age[tid] = 0
                for tid in lost:
                    self._age_or_drop(tid)
                # Compute the newborn dy seed AFTER matched tracks have
                # refreshed their dy but BEFORE new tracks are added, so
                # newborns don't bootstrap off each other's zeros.
                dy_seed = self._bootstrap_dy()
                for pt in unmatched:
                    self._spawn_track(pt, dy_seed)
        else:
            self._match_greedy(cur_centers)

        # ── Update aggregate measurements ────────────────────────────
        for tid, center in self._tracking_objects.items():
            dims = center_to_dims.get(center)
            if dims is None:
                continue            # coasting track (no detection this frame)
            length      = dims["length"]
            width       = dims["width"]
            area        = dims["area"]
            mean_bgr    = dims.get("mean_bgr")
            width_valid = bool(dims.get("width_valid", True))

            if tid in self.max_dims:
                m = self.max_dims[tid]
                if area > m.area:
                    m.area   = area
                    m.length = length
                    if mean_bgr is not None:
                        m.mean_bgr = mean_bgr
                if width_valid:
                    if m._width_valid_count == 0 or width < m.width:
                        m.width = width
                    m._width_valid_count += 1
                    m.width_quality = "clean"
                else:
                    if width < m._width_invalid_min:
                        m._width_invalid_min = width
            else:
                # First time we see this track.  Seed width from the
                # current frame regardless of validity (so the field is
                # never zero), but flag fallback if invalid.
                if width_valid:
                    m = TrackMeasurement(
                        track_id=tid, length=length, width=width, area=area,
                        mean_bgr=mean_bgr, width_quality="clean",
                    )
                    m._width_valid_count = 1
                else:
                    m = TrackMeasurement(
                        track_id=tid, length=length, width=width, area=area,
                        mean_bgr=mean_bgr, width_quality="fallback",
                    )
                    m._width_invalid_min = width
                self.max_dims[tid] = m

        # Resolve fallback widths for tracks that *never* saw a valid
        # frame -- use the min over invalid frames so the CSV still
        # carries a number.  Tracks with at least one valid frame keep
        # their clean min.
        for m in self.max_dims.values():
            if m._width_valid_count == 0 and m._width_invalid_min != float("inf"):
                m.width = m._width_invalid_min
                m.width_quality = "fallback"

        self._prev_centers = list(center_to_dims.keys())
        return self._tracking_objects

    # ── Greedy matcher (legacy behaviour + ageing hook) ──────────────
    def _match_greedy(self, cur_centers: List[Tuple[int, int]]) -> None:
        objects_copy = self._tracking_objects.copy()
        for object_id, pt2 in objects_copy.items():
            object_exists = False
            min_dist = float("inf")
            closest_pt = None

            for pt in cur_centers:
                dist = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                max_iou = 0.0
                for _, prev_pt in objects_copy.items():
                    prev_box = (prev_pt[0] - 5, prev_pt[1] - 5,
                                prev_pt[0] + 5, prev_pt[1] + 5)
                    cur_box = (pt[0] - 5, pt[1] - 5,
                               pt[0] + 5, pt[1] + 5)
                    iou = _calculate_iou(prev_box, cur_box)
                    if iou > max_iou:
                        max_iou = iou

                if max_iou > self.iou_threshold:
                    self._tracking_objects[object_id] = pt
                    closest_pt = pt
                    object_exists = True
                    continue

                if dist < self.distance_threshold and pt[1] > pt2[1]:
                    if dist < min_dist:
                        min_dist = dist
                        closest_pt = pt
                        object_exists = True

            if object_exists and closest_pt is not None:
                # dy measured against the track's position at frame start
                # (pt2), so the IoU mid-loop mutation above can't corrupt it.
                self._update_dy(object_id, pt2, closest_pt)
                self._tracking_objects[object_id] = closest_pt
                self._track_age[object_id] = 0
                if closest_pt in cur_centers:
                    cur_centers.remove(closest_pt)
            else:
                self._age_or_drop(object_id)

        dy_seed = self._bootstrap_dy()
        for pt in cur_centers:
            self._spawn_track(pt, dy_seed)
