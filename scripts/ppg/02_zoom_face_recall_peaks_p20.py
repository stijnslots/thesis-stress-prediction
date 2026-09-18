"""
Zoomed-in peak-detection comparison for p20's Face Recall PPG.

Follow-up to 01_explore_ppg_p20.py: that script found HeartPy's RMSSD on
this window (281.8 ms) too high compared to NeuroKit2 (91.8 ms). This
script zooms into a representative 13-second slice and overlays both
methods' detected peaks on the waveform, to check whether heartbeats are
being double-counted (mismatches), noise is being picked up as a peak, or
beats are being missed.

Usage:
    python 02_zoom_face_recall_peaks_p20.py
    python 02_zoom_face_recall_peaks_p20.py --start-s 60 --duration-s 15
"""

import argparse
import importlib.util
import os
import time

if not hasattr(time, "clock"):
    time.clock = time.perf_counter

import heartpy as hp
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np


def _load_explore_module():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "01_explore_ppg_p20.py")
    spec = importlib.util.spec_from_file_location("ppg_explore_p20", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify_intervals(peak_samples, sample_rate, plausible_min_bpm=50, plausible_max_bpm=150):
    """RR intervals shorter than what plausible_max_bpm implies can't be a
    genuine single beat-to-beat gap at this HR — they're either a second
    detection within one true beat (e.g. the dicrotic notch) or a noise
    blip. RR intervals longer than plausible_min_bpm implies would instead
    mean a beat was missed."""
    min_interval_s = 60.0 / plausible_max_bpm
    max_interval_s = 60.0 / plausible_min_bpm
    intervals_s = np.diff(peak_samples) / sample_rate
    too_short = intervals_s < min_interval_s
    too_long = intervals_s > max_interval_s
    return intervals_s, too_short, too_long


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--participant", default="p20")
    parser.add_argument("--task", default="face_recall")
    parser.add_argument("--ppg-path", default=os.path.join("data", "raw", "ppg", "P20_Shimmer.csv"))
    parser.add_argument("--windows-path", default=os.path.join("data", "processed", "synced", "p20_task_windows.csv"))
    parser.add_argument("--out", default=os.path.join("results", "ppg_exploration"))
    parser.add_argument("--start-s", type=float, default=0.0, help="Offset (s) into the task window where the zoom starts")
    parser.add_argument("--duration-s", type=float, default=13.0)
    parser.add_argument("--out-name", default=None,
                         help="Override the auto-generated output filename (saved under --out).")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    m = _load_explore_module()

    ppg_df = m.load_ppg(args.ppg_path)
    sample_rate = m.estimate_sample_rate(ppg_df[m.TS_COL].values)
    window = m.load_task_window(args.windows_path, args.task)
    start_unix_ms = float(window["start_abs_unix"]) * 1000.0
    end_unix_ms = float(window["end_abs_unix"]) * 1000.0
    segment = m.cut_window(ppg_df, start_unix_ms, end_unix_ms)
    full_signal = segment[m.PPG_COL].to_numpy()

    # --- run both pipelines on the FULL task window (as in 01_explore_...),
    # then slice the zoom window out of their outputs, so peak positions match
    # exactly what was reported before (no reprocessing-boundary artifacts).
    hp_preprocessed = m.preprocess_for_heartpy(full_signal, sample_rate)
    hp_working, hp_measures, hp_error, _ = m.run_heartpy(full_signal, sample_rate)
    if hp_error is not None:
        raise RuntimeError(f"HeartPy failed on the full window: {hp_error}")

    nk_signals, nk_info, _ = m.run_neurokit(full_signal, sample_rate)

    start_idx = int(args.start_s * sample_rate)
    end_idx = int((args.start_s + args.duration_s) * sample_rate)
    end_idx = min(end_idx, len(full_signal))
    t_rel = np.arange(start_idx, end_idx) / sample_rate

    hp_peaks_all = np.array(hp_working["peaklist"])
    hp_accepted_mask = np.array(hp_working["binary_peaklist"]).astype(bool)
    hp_peaks_accepted = hp_peaks_all[hp_accepted_mask]
    hp_peaks_rejected = hp_peaks_all[~hp_accepted_mask]

    nk_peaks_all = np.where(nk_signals["PPG_Peaks"].values == 1)[0]

    def in_window(peaks):
        return peaks[(peaks >= start_idx) & (peaks < end_idx)]

    hp_zoom_accepted = in_window(hp_peaks_accepted)
    hp_zoom_rejected = in_window(hp_peaks_rejected)
    nk_zoom_peaks = in_window(nk_peaks_all)

    # --- interval diagnostics over the FULL window (more robust than the
    # ~13s zoom alone) to characterize the systematic error mode.
    hp_intervals_s, hp_too_short, hp_too_long = classify_intervals(hp_peaks_accepted, sample_rate)
    nk_intervals_s, nk_too_short, nk_too_long = classify_intervals(nk_peaks_all, sample_rate)

    expected_beats = window["duration_s"] / (60.0 / 74.0)  # ~74 bpm from the spectral check in 01_explore_...
    print(f"Task window duration: {window['duration_s']:.1f}s, ~{expected_beats:.0f} beats expected at ~74 bpm")
    print(f"HeartPy:   {len(hp_peaks_accepted)} accepted peaks ({len(hp_peaks_rejected)} rejected), "
          f"{hp_too_short.sum()}/{len(hp_intervals_s)} intervals too short (<{60/150:.2f}s, i.e. >150 bpm), "
          f"{hp_too_long.sum()} too long (<50 bpm)")
    print(f"NeuroKit2: {len(nk_peaks_all)} peaks, "
          f"{nk_too_short.sum()}/{len(nk_intervals_s)} intervals too short, {nk_too_long.sum()} too long")

    if hp_too_short.sum() > nk_too_short.sum():
        print("\nDiagnosis: HeartPy has far more implausibly-short RR intervals than NeuroKit2 — "
              "consistent with double-counting (detecting a second peak, e.g. the dicrotic notch, "
              "within a single true heartbeat) rather than missing beats.")

    # --- zoomed overlay plot ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax = axes[0]
    ax.plot(t_rel, hp_preprocessed[start_idx:end_idx], color="tab:blue", linewidth=1.0, label="HeartPy input (bandpass + scaled)")
    if len(hp_zoom_accepted):
        ax.scatter(hp_zoom_accepted / sample_rate, hp_preprocessed[hp_zoom_accepted], color="green", s=60, zorder=5, label="HeartPy accepted peak")
    if len(hp_zoom_rejected):
        ax.scatter(hp_zoom_rejected / sample_rate, hp_preprocessed[hp_zoom_rejected], color="red", s=60, marker="x", zorder=5, label="HeartPy rejected peak")
    # annotate accepted-peak intervals within the zoom window
    zoom_accepted_sorted = np.sort(hp_zoom_accepted)
    for a, b in zip(zoom_accepted_sorted[:-1], zoom_accepted_sorted[1:]):
        dt_ms = (b - a) / sample_rate * 1000
        mid_t = (a + b) / 2 / sample_rate
        color = "red" if dt_ms < (60000 / 150) else "black"
        ax.annotate(f"{dt_ms:.0f} ms", (mid_t, ax.get_ylim()[1] * 0.9), fontsize=7, ha="center", color=color)
    ax.set_title("HeartPy peak detection")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Filtered & scaled amplitude")
    ax.legend(loc="upper right", fontsize=8)

    ax2 = axes[1]
    nk_clean = nk_signals["PPG_Clean"].to_numpy()
    ax2.plot(t_rel, nk_clean[start_idx:end_idx], color="tab:orange", linewidth=1.0, label="NeuroKit2 cleaned signal")
    if len(nk_zoom_peaks):
        ax2.scatter(nk_zoom_peaks / sample_rate, nk_clean[nk_zoom_peaks], color="darkgreen", s=60, zorder=5, label="NeuroKit2 peak")
    nk_zoom_sorted = np.sort(nk_zoom_peaks)
    for a, b in zip(nk_zoom_sorted[:-1], nk_zoom_sorted[1:]):
        dt_ms = (b - a) / sample_rate * 1000
        mid_t = (a + b) / 2 / sample_rate
        color = "red" if dt_ms < (60000 / 150) else "black"
        ax2.annotate(f"{dt_ms:.0f} ms", (mid_t, ax2.get_ylim()[1] * 0.9), fontsize=7, ha="center", color=color)
    ax2.set_title("NeuroKit2 peak detection")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Cleaned amplitude")
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out_name = args.out_name or f"{args.participant}_{args.task}_zoom_{int(args.start_s)}s-{int(args.start_s + args.duration_s)}s_peak_comparison.png"
    out_path = os.path.join(args.out, out_name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved zoom comparison plot: {out_path}")
    print(f"  HeartPy peaks in zoom window: {len(hp_zoom_accepted)} accepted, {len(hp_zoom_rejected)} rejected")
    print(f"  NeuroKit2 peaks in zoom window: {len(nk_zoom_peaks)}")


if __name__ == "__main__":
    main()
