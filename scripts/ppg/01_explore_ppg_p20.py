"""
Exploratory PPG analysis for participant p20, Stroop task window.

Compares HeartPy (proposal) against NeuroKit2 as an
alternative pipeline, per the open decision point in
notes/methodology_notes.md ("PPG Processing").

Steps:
    1. Load the raw Shimmer PPG csv (128 Hz, Unix ms timestamps).
    2. Load the task-window boundaries (data/processed/synced/p20_task_windows.csv).
    3. Cut out the Stroop window.
    4. Plot the raw waveform.
    5. Run HeartPy peak detection + HRV metrics (SDNN, RMSSD, pNN50, LF/HF).
    6. Run NeuroKit2 on the same window.
    7. Compare the two outputs and flag physiologically implausible values
       (mean HR outside ~50-150 bpm) or large disagreements between methods.

Note: the raw signal amplitude has a peak-to-peak swing of only 10-20 mV around a 1400 mV DC baseline (see the raw waveform
plot). A naive std/unique-value check on that raw signal looks "flat", but a
Welch PSD shows a clear narrowband peak at ~1.2-1.4 Hz (~70-85 bpm) carrying
~95%+ of the sub-15Hz power — i.e. there is a real, low-amplitude cardiac
signal, HeartPy's default peak-fit just can't find it without the bandpass
filter + rescale applied in `preprocess_for_heartpy` below (standard HeartPy
practice for raw ADC-scale signals). NeuroKit2's `ppg_process` already does
equivalent cleaning internally, which is why it succeeds on the raw signal.

NeuroKit2 wins
"""

import argparse
import os
import time

# HeartPy calls a function that's not in Python (newer version), only used
# for an internal timing print, so removing is safe.
if not hasattr(time, "clock"):
    time.clock = time.perf_counter

import heartpy as hp
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy import signal as sps

PPG_COL = "S5E1C_PPG_A13_CAL"
TS_COL = "S5E1C_Timestamp_Unix_CAL"

# Physiologically plausible resting/task HR range used for the sanity check
# in step 7.
HR_PLAUSIBLE_MIN = 50
HR_PLAUSIBLE_MAX = 150

# SDNN/RMSSD above this over a window is atypical and usually indicates
# missed or double counted beats.
HRV_PLAUSIBLE_MAX_MS = 200


def load_ppg(ppg_path):
    df = pd.read_csv(ppg_path, sep="\t", skiprows=[0, 2])
    df[TS_COL] = df[TS_COL].astype(float)
    df[PPG_COL] = df[PPG_COL].astype(float)
    return df


def estimate_sample_rate(timestamps_ms):
    diffs = np.diff(timestamps_ms)
    median_dt_ms = np.median(diffs)
    return 1000.0 / median_dt_ms


def load_task_window(windows_path, task):
    windows = pd.read_csv(windows_path)
    row = windows[windows["task"] == task]
    if row.empty:
        available = ", ".join(windows["task"].unique())
        raise ValueError(f"Task '{task}' not found in {windows_path}. Available: {available}")
    return row.iloc[0]


def cut_window(df, start_unix_ms, end_unix_ms):
    mask = (df[TS_COL] >= start_unix_ms) & (df[TS_COL] <= end_unix_ms)
    return df.loc[mask].reset_index(drop=True)


CARDIAC_BAND = (0.6, 3.5)  # Hz, ~36-210 bpm — generous band for a spectral sanity check


def signal_quality_diagnostics(signal, sample_rate):
    """Spectral sanity check: does the signal actually carry a cardiac-band
    oscillation, regardless of its raw amplitude? Raw Shimmer PPG has a small
    peak-to-peak swing (tens of mV on a 1400 mV baseline), so a time-domain
    std/unique-value check alone is misleading (see module docstring). Instead
    we check how much of the sub-15Hz power concentrates in a narrow window
    around the dominant peak inside CARDIAC_BAND: a real pulse shows a sharp
    spectral peak there; broadband sensor noise does not."""
    n_unique = len(np.unique(signal))
    std = float(np.std(signal))
    mean = float(np.mean(signal))
    rel_std_pct = (std / abs(mean) * 100) if mean != 0 else np.nan

    freqs, psd = sps.welch(signal - mean, fs=sample_rate, nperseg=min(4096, len(signal)))
    band_mask = (freqs >= CARDIAC_BAND[0]) & (freqs <= CARDIAC_BAND[1])
    if not band_mask.any() or psd[band_mask].sum() == 0:
        peak_freq_hz, peak_bpm, band_dominance = np.nan, np.nan, np.nan
    else:
        band_freqs, band_psd = freqs[band_mask], psd[band_mask]
        peak_i = np.argmax(band_psd)
        peak_freq_hz = float(band_freqs[peak_i])
        peak_bpm = peak_freq_hz * 60
        # power within +-0.15 Hz of the peak, as a fraction of all in-band power
        window_mask = np.abs(band_freqs - peak_freq_hz) <= 0.15
        band_dominance = float(band_psd[window_mask].sum() / band_psd.sum())

    has_plausible_cardiac_peak = (
        not np.isnan(peak_bpm)
        and HR_PLAUSIBLE_MIN <= peak_bpm <= HR_PLAUSIBLE_MAX
        and band_dominance >= 0.15
    )
    return {
        "n_samples": len(signal),
        "n_unique": n_unique,
        "mean": mean,
        "std": std,
        "rel_std_pct": rel_std_pct,
        "peak_freq_hz": peak_freq_hz,
        "peak_bpm": peak_bpm,
        "band_dominance": band_dominance,
        "has_plausible_cardiac_peak": has_plausible_cardiac_peak,
    }


def preprocess_for_heartpy(signal, sample_rate):
    """Bandpass-filter + rescale before HeartPy's peak-fit. Standard HeartPy
    practice for raw ADC-scale signals (see hp.filter_signal/hp.scale_data
    docs) — needed here because the raw Shimmer PPG swing is only ~10-20 mV
    on a ~1400 mV baseline, too small for hp.process's default peak-fit to
    lock onto directly (see module docstring)."""
    filtered = hp.filter_signal(signal, cutoff=list(CARDIAC_BAND), sample_rate=sample_rate,
                                 order=3, filtertype="bandpass")
    return hp.scale_data(filtered)


def run_heartpy(signal, sample_rate):
    preprocessed = preprocess_for_heartpy(signal, sample_rate)
    lf_hf_error = None
    try:
        working_data, measures = hp.process(preprocessed, sample_rate=sample_rate, calc_freq=True)
    except hp.exceptions.BadSignalWarning as e:
        return None, None, str(e), None
    except TypeError:
        # HeartPy's frequency-domain calculation isn't compatible with the
        # current NumPy version. Time-domain metrics still run; LF/HF is
        # reported as unavailable rather than faked.
        lf_hf_error = "heartpy==1.2.4 calc_fd_measures() is incompatible with the installed NumPy version (np.linspace float 'num' arg)"
        working_data, measures = hp.process(preprocessed, sample_rate=sample_rate, calc_freq=False)
    return working_data, measures, None, lf_hf_error


def run_neurokit(signal, sample_rate):
    signals, info = nk.ppg_process(signal, sampling_rate=sample_rate)
    peak_idx = np.where(signals["PPG_Peaks"].values == 1)[0]
    hrv = nk.hrv(peak_idx, sampling_rate=sample_rate, show=False)
    return signals, info, hrv


def summarize_neurokit_hrv(hrv_df):
    def get(col):
        return float(hrv_df[col].iloc[0]) if col in hrv_df.columns else np.nan

    lf = get("HRV_LF")
    hf = get("HRV_HF")
    lf_hf = lf / hf if lf is not None and hf not in (None, 0) and not np.isnan(lf) and not np.isnan(hf) else np.nan
    return {
        "sdnn": get("HRV_SDNN"),
        "rmssd": get("HRV_RMSSD"),
        "pnn50": get("HRV_pNN50"),
        "lf_hf": lf_hf,
    }


def summarize_heartpy(measures):
    # HeartPy reports pnn50 as a proportion, NeuroKit2 as a percentage.
    # Rescaled here so the comparison is apples-to-apples.
    pnn50 = measures.get("pnn50", np.nan)
    pnn50_pct = pnn50 * 100 if pnn50 is not None and not np.isnan(pnn50) else np.nan
    return {
        "bpm": measures.get("bpm", np.nan),
        "sdnn": measures.get("sdnn", np.nan),
        "rmssd": measures.get("rmssd", np.nan),
        "pnn50": pnn50_pct,
        "lf": measures.get("lf", np.nan),
        "hf": measures.get("hf", np.nan),
        "lf_hf": measures.get("lf/hf", np.nan),
    }


def plausibility_flags(quality, hp_summary, hp_error, nk_bpm, nk_rate_min, nk_rate_max):
    flags = []
    if not quality["has_plausible_cardiac_peak"]:
        flags.append(
            f"No clear cardiac-band spectral peak found (dominant freq={quality['peak_freq_hz']:.3f} Hz "
            f"/ {quality['peak_bpm']:.1f} bpm, dominance={quality['band_dominance']:.2f} of in-band power). "
            "Treat any HR/HRV numbers below as unreliable regardless of which method produced them."
        )

    if hp_error is not None:
        flags.append(f"HeartPy could not fit peaks at all (BadSignalWarning): {next((l for l in hp_error.splitlines() if l.strip("- \t")), hp_error)}")
    elif not (HR_PLAUSIBLE_MIN <= hp_summary["bpm"] <= HR_PLAUSIBLE_MAX):
        flags.append(f"HeartPy mean HR {hp_summary['bpm']:.1f} bpm is outside the plausible "
                      f"{HR_PLAUSIBLE_MIN}-{HR_PLAUSIBLE_MAX} bpm range.")

    if not (HR_PLAUSIBLE_MIN <= nk_bpm <= HR_PLAUSIBLE_MAX):
        flags.append(f"NeuroKit2 mean HR {nk_bpm:.1f} bpm is outside the plausible "
                      f"{HR_PLAUSIBLE_MIN}-{HR_PLAUSIBLE_MAX} bpm range.")
    if nk_rate_max > HR_PLAUSIBLE_MAX or nk_rate_min < HR_PLAUSIBLE_MIN:
        flags.append(
            f"NeuroKit2 instantaneous rate ranges {nk_rate_min:.1f}-{nk_rate_max:.1f} bpm — "
            f"swings outside {HR_PLAUSIBLE_MIN}-{HR_PLAUSIBLE_MAX} bpm even though the mean looks "
            "plausible, which is a sign it is tracking noise peaks rather than true beats."
        )

    if hp_error is None:
        hr_diff = abs(hp_summary["bpm"] - nk_bpm)
        if hr_diff > 5:
            flags.append(f"Mean HR disagreement between methods is {hr_diff:.1f} bpm (> 5 bpm threshold).")
        for metric in ("sdnn", "rmssd"):
            if hp_summary[metric] > HRV_PLAUSIBLE_MAX_MS:
                flags.append(f"HeartPy {metric.upper()}={hp_summary[metric]:.1f} ms is atypically high "
                              f"(>{HRV_PLAUSIBLE_MAX_MS} ms) — likely missed/double-counted beats, not genuine HRV.")

    return flags


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--participant", default="p20")
    parser.add_argument("--task", default="stroop", choices=["stroop", "arithmetic", "face_recall"])
    parser.add_argument("--ppg-path", default=os.path.join("data", "raw", "ppg", "P20_Shimmer.csv"))
    parser.add_argument("--windows-path", default=os.path.join("data", "processed", "synced", "p20_task_windows.csv"))
    parser.add_argument("--out", default=os.path.join("results", "ppg_exploration"))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading PPG data from {args.ppg_path} ...")
    ppg_df = load_ppg(args.ppg_path)
    sample_rate = estimate_sample_rate(ppg_df[TS_COL].values)
    print(f"  {len(ppg_df)} samples, estimated sample rate: {sample_rate:.2f} Hz")

    print(f"Loading task window '{args.task}' from {args.windows_path} ...")
    window = load_task_window(args.windows_path, args.task)
    start_unix_ms = float(window["start_abs_unix"]) * 1000.0
    end_unix_ms = float(window["end_abs_unix"]) * 1000.0
    print(f"  window: {window['start_s']:.2f}s - {window['end_s']:.2f}s "
          f"(rel. to expStart), duration {window['duration_s']:.1f}s")
    print(f"  absolute unix ms: {start_unix_ms:.0f} - {end_unix_ms:.0f}")

    segment = cut_window(ppg_df, start_unix_ms, end_unix_ms)
    print(f"  cut segment: {len(segment)} samples (~{len(segment) / sample_rate:.1f}s)")
    if len(segment) == 0:
        raise ValueError("Cut segment is empty — check that ppg-path and windows-path refer to the same participant/session.")

    signal = segment[PPG_COL].to_numpy()
    t_rel = np.arange(len(signal)) / sample_rate

    quality = signal_quality_diagnostics(signal, sample_rate)
    print(f"\nSegment quality check: {quality['n_unique']} unique raw values / {quality['n_samples']} samples, "
          f"mean={quality['mean']:.1f} mV, std={quality['std']:.2f} mV ({quality['rel_std_pct']:.2f}% of mean)")
    print(f"  Spectral check: dominant cardiac-band peak at {quality['peak_freq_hz']:.3f} Hz "
          f"(~{quality['peak_bpm']:.1f} bpm), carrying {quality['band_dominance']*100:.1f}% of in-band power")
    if not quality["has_plausible_cardiac_peak"]:
        print("  WARNING: no plausible/dominant cardiac-band spectral peak — signal may not be usable.")

    # --- Step 4: raw waveform plot ---
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(t_rel, signal, linewidth=0.6)
    ax.set_title(f"{args.participant} — raw PPG waveform — {args.task} window")
    ax.set_xlabel("Time (s, relative to task start)")
    ax.set_ylabel("PPG (mV)")
    fig.tight_layout()
    raw_plot_path = os.path.join(args.out, f"{args.participant}_{args.task}_raw_ppg.png")
    fig.savefig(raw_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved raw waveform plot: {raw_plot_path}")

    # --- Step 5: HeartPy ---
    print("\nRunning HeartPy...")
    hp_working, hp_measures, hp_error, hp_lf_hf_error = run_heartpy(signal, sample_rate)
    if hp_error is not None:
        print(f"  HeartPy FAILED to fit peaks: {next((l for l in hp_error.splitlines() if l.strip("- \t")), hp_error)}")
        hp_summary = {k: np.nan for k in ("bpm", "sdnn", "rmssd", "pnn50", "lf", "hf", "lf_hf")}
    else:
        hp_summary = summarize_heartpy(hp_measures)
        hp.plotter(hp_working, hp_measures, show=False)
        plt.gcf().set_size_inches(14, 4)
        hp_plot_path = os.path.join(args.out, f"{args.participant}_{args.task}_heartpy_peaks.png")
        plt.savefig(hp_plot_path, dpi=150)
        plt.close("all")
        print(f"  Saved HeartPy peak plot: {hp_plot_path}")
        lf_hf_str = f"{hp_summary['lf_hf']:.2f}" if hp_lf_hf_error is None else "unavailable"
        print(f"  HeartPy: bpm={hp_summary['bpm']:.1f}  sdnn={hp_summary['sdnn']:.1f}  "
              f"rmssd={hp_summary['rmssd']:.1f}  pnn50={hp_summary['pnn50']:.3f}  "
              f"lf/hf={lf_hf_str}")
        if hp_lf_hf_error is not None:
            print(f"  NOTE: {hp_lf_hf_error}")

    # --- Step 6: NeuroKit2 ---
    print("\nRunning NeuroKit2...")
    nk_signals, nk_info, nk_hrv = run_neurokit(signal, sample_rate)
    nk_summary = summarize_neurokit_hrv(nk_hrv)
    nk_bpm = float(nk_signals["PPG_Rate"].mean())
    nk_rate_min = float(nk_signals["PPG_Rate"].min())
    nk_rate_max = float(nk_signals["PPG_Rate"].max())

    nk.ppg_plot(nk_signals, info=nk_info)
    nk_plot_path = os.path.join(args.out, f"{args.participant}_{args.task}_neurokit_peaks.png")
    plt.gcf().set_size_inches(14, 8)
    plt.savefig(nk_plot_path, dpi=150)
    plt.close("all")
    print(f"  Saved NeuroKit2 peak plot: {nk_plot_path}")
    print(f"  NeuroKit2: bpm={nk_bpm:.1f} (range {nk_rate_min:.1f}-{nk_rate_max:.1f})  "
          f"sdnn={nk_summary['sdnn']:.1f}  rmssd={nk_summary['rmssd']:.1f}  "
          f"pnn50={nk_summary['pnn50']:.3f}  lf/hf={nk_summary['lf_hf']:.2f}")

    # --- Step 7: comparison + plausibility flags ---
    comparison = pd.DataFrame({
        "metric": ["bpm", "sdnn", "rmssd", "pnn50", "lf_hf"],
        "heartpy": [hp_summary["bpm"], hp_summary["sdnn"], hp_summary["rmssd"],
                    hp_summary["pnn50"], hp_summary["lf_hf"]],
        "neurokit2": [nk_bpm, nk_summary["sdnn"], nk_summary["rmssd"],
                      nk_summary["pnn50"], nk_summary["lf_hf"]],
    })
    comparison["abs_diff"] = (comparison["heartpy"] - comparison["neurokit2"]).abs()
    comparison["pct_diff"] = (comparison["abs_diff"] / comparison["heartpy"].abs()) * 100

    print("\n=== Comparison: HeartPy vs NeuroKit2 ===")
    if hp_error is not None:
        print("(HeartPy columns are NaN because it refused to fit peaks on this segment — see plausibility check below)")
    print(comparison.to_string(index=False))

    flags = plausibility_flags(quality, hp_summary, hp_error, nk_bpm, nk_rate_min, nk_rate_max)
    print("\n=== Plausibility check ===")
    if flags:
        for f in flags:
            print(f"  WARNING: {f}")
    else:
        print("  No plausibility issues flagged (both HR estimates in "
              f"{HR_PLAUSIBLE_MIN}-{HR_PLAUSIBLE_MAX} bpm, methods agree within 5 bpm).")

    comparison_path = os.path.join(args.out, f"{args.participant}_{args.task}_hrv_comparison.csv")
    comparison.to_csv(comparison_path, index=False)
    print(f"\nSaved comparison table: {comparison_path}")


if __name__ == "__main__":
    main()
