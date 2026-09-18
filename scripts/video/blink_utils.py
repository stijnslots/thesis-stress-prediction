"""
Shared blink-episode utilities for the video pipeline.

Replaces a fixed, one-participant-derived blink-duration threshold with a
per-participant adaptive threshold: the fixed values didn't generalize to
other participants with different natural blink rates. This module derives
a threshold from each participant's own blink-duration distribution
instead.
"""

import numpy as np
from scipy.ndimage import binary_dilation

# Features that stayed flagged as contaminated even after adaptive
# blink-filtering and camera calibration. Excluded from
# the aggregated feature set and from the validation check, not from
# the raw cached output. Defined once here so the two scripts can't
# diverge on which features are excluded (CHECK this).
EXCLUDED_CONTAMINATED_FEATURES = {"pose_Rx", "gaze_1_z"}


def au45_blink_run_indices(au45_c):
    on = np.asarray(au45_c) == 1
    diff = np.diff(on.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if on[0]:
        starts = np.insert(starts, 0, 0)
    if on[-1]:
        ends = np.append(ends, len(on))
    return starts, ends


def au45_blink_run_durations_ms(au45_c, frame_time_s):
    starts, ends = au45_blink_run_indices(au45_c)
    return (ends - starts) * frame_time_s * 1000.0


def derive_adaptive_blink_thresholds(durations_ms, frame_time_ms, min_events=10, iqr_multiplier=1.5):
    """Per-participant adaptive blink-duration thresholds.

    NOTE on method history: the original approach (see git history / prior
    report) tried to automate the "eyeball two big gaps in the histogram"
    method used by hand on a 35-event p20 test clip. Pooling the FULL
    recording (~340-840 events per participant) exposed that those gaps
    don't actually exist at the population level -- the duration
    distribution near 350-600ms is a smooth continuum, not two separated
    clusters (checked directly: p20's pooled data has essentially one event
    every ~17-20ms across that whole range). The apparent gap on the small
    sample was a small-N sampling artifact, not a real feature of the
    distribution -- so gap-detection was replaced with a standard,
    textbook-robust alternative:

    - min_duration_ms: a fixed *physical* floor of 2 frame-durations. A
      genuine open-close-open blink cycle cannot complete in a single
      frame, so a 1-frame AU45_c "on" run is almost certainly classifier
      jitter, not a real blink -- this needs no statistical justification
      beyond "measurement granularity."
    - max_duration_ms: Tukey's mild-outlier fence, Q3 + iqr_multiplier*IQR,
      computed on THIS participant's own pooled AU45_c run-duration
      distribution. Standard, adaptive, and -- checked on p20/p21/p22 --
      lands consistently around the 86th-89th percentile for all three
      despite their quite different raw blink rates and durations.

    Pool across all available tasks before calling this (more events =
    a more stable IQR) rather than deriving separately per task.
    """
    d = np.sort(np.asarray(durations_ms, dtype=float))
    n = len(d)
    min_duration_ms = 2 * frame_time_ms

    if n < min_events:
        return {
            "n_events": n, "min_duration_ms": min_duration_ms, "max_duration_ms": float("inf"),
            "q1_ms": None, "q3_ms": None, "iqr_ms": None,
            "fallback_reason": f"only {n} blink episodes found (<{min_events}) -- max threshold left unbounded",
        }

    q1, q3 = np.percentile(d, [25, 75])
    iqr = q3 - q1
    max_duration_ms = float(q3 + iqr_multiplier * iqr)

    return {
        "n_events": n,
        "min_duration_ms": float(min_duration_ms),
        "max_duration_ms": max_duration_ms,
        "q1_ms": float(q1), "q3_ms": float(q3), "iqr_ms": float(iqr),
        "fallback_reason": None,
    }


def compute_blink_exclusion_mask(au45_c, frame_time_s, margin_frames, min_duration_ms, max_duration_ms):
    """True = exclude this frame: it falls within a REAL blink episode
    (duration in [min_duration_ms, max_duration_ms]) plus a +/-margin_frames
    margin."""
    n = len(au45_c)
    starts, ends = au45_blink_run_indices(au45_c)
    durations_ms = (ends - starts) * frame_time_s * 1000.0
    is_real_blink = (durations_ms >= min_duration_ms) & (durations_ms <= max_duration_ms)

    mask = np.zeros(n, dtype=bool)
    for s, e in zip(starts[is_real_blink], ends[is_real_blink]):
        mask[s:e] = True
    return binary_dilation(mask, structure=np.ones(2 * margin_frames + 1, dtype=bool))
