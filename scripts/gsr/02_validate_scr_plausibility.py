"""
Exploratory. Validates detected skin-conductance-response counts and
tonic/phasic decomposition quality across the GSR participant set. Not in
main pipeline yet.

Known SCR-detection methods can over-detect responses on low-variance,
low-amplitude EDA signals, fitting small peaks to what is really noise or
slow drift. This script applies two checks per participant and task:

  1. Rate plausibility: compares the detected SCR rate against a
     physiologically plausible range, flagging both implausibly low
     (near-flat signal) and implausibly high (likely noise-driven) results.
  2. Tonic/phasic variance attribution: what fraction of the signal's
     variance ends up in the tonic component versus the phasic component.
     A window where almost all variance is tonic leaves little genuine
     signal for any detected peaks to represent, regardless of how
     plausible their count looks.

Usage:
    python 02_validate_scr_plausibility.py
    python 02_validate_scr_plausibility.py --participants p3 p4 p5
"""

import argparse
import os

import pandas as pd

GSR_DIR = os.path.join("data", "processed", "gsr")
OUT_DIR = os.path.join("results", "gsr_exploration")

# Plausible SCR rate range, widened on both ends, not resting. Values
# outside this are flagged for a closer look.
RATE_LOW_PLAUSIBLE = 0.5
RATE_HIGH_PLAUSIBLE = 6.0

# Below this raw-signal-range (uS) over the full task window, the segment
# is close to a flat line -- same diagnostic used for the p10 finding.
FLAT_RAW_RANGE_UOM = 0.05

# Below this mean amplitude, detected peaks look close to the noise
# floor seen elsewhere in this sample.
LOW_AMPLITUDE_UOM = 0.03

# Above this % of raw variance attributed to the tonic component, only a
# small residual is left for genuine phasic responses.
TONIC_DOMINANT_PCT = 97.0


def find_participants(gsr_dir):
    ids = set()
    for f in os.listdir(gsr_dir):
        if f.endswith("_gsr_features.csv"):
            ids.add(f[: -len("_gsr_features.csv")])
    return sorted(ids, key=lambda x: int(x[1:]))


def load_features(participants):
    rows = []
    for p in participants:
        path = os.path.join(GSR_DIR, f"{p}_gsr_features.csv")
        if not os.path.exists(path):
            print(f"  WARNING: no features file for {p} -- skipping")
            continue
        df = pd.read_csv(path)
        df["participant"] = p
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def compute_raw_signal_diagnostics(participants, tasks):
    """Raw range/std and tonic/phasic variance split, from the cached
    per-task signal csv (written by 01_extract_gsr_features.py)."""
    rows = []
    for p in participants:
        for task in tasks:
            path = os.path.join(GSR_DIR, f"{p}_{task}_eda_signal.csv")
            if not os.path.exists(path):
                continue
            sig = pd.read_csv(path)
            raw = sig["EDA_Raw"]
            raw_range = float(raw.max() - raw.min())
            raw_std = float(raw.std())
            tonic_std = float(sig["EDA_Tonic"].std())
            pct_tonic = tonic_std / raw_std * 100.0 if raw_std > 0 else float("nan")
            rows.append({
                "participant": p, "task": task,
                "raw_range_uS": raw_range, "raw_std": raw_std,
                "tonic_std": tonic_std, "pct_variance_tonic": pct_tonic,
            })
    return pd.DataFrame(rows)


def classify(row):
    flags = []
    if row["nscr_per_min"] < RATE_LOW_PLAUSIBLE:
        flags.append("LOW_RATE")
    if row["nscr_per_min"] > RATE_HIGH_PLAUSIBLE:
        flags.append("HIGH_RATE")
    if pd.notna(row.get("raw_range_uS")) and row["raw_range_uS"] < FLAT_RAW_RANGE_UOM:
        flags.append("FLAT_RAW_SIGNAL")
    if pd.notna(row.get("scr_amplitude_mean")) and row["scr_amplitude_mean"] < LOW_AMPLITUDE_UOM:
        flags.append("LOW_AMPLITUDE_SUSPECT_NOISE")
    if pd.notna(row.get("pct_variance_tonic")) and row["pct_variance_tonic"] > TONIC_DOMINANT_PCT:
        flags.append("TONIC_DOMINATED_DECOMPOSITION")
    return ";".join(flags) if flags else "PLAUSIBLE"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--participants", nargs="+", default=None,
                         help="Participant ids to validate (default: every participant with a "
                              "*_gsr_features.csv already extracted).")
    parser.add_argument("--tasks", nargs="+", default=["stroop", "arithmetic", "face_recall"])
    args = parser.parse_args()

    participants = args.participants or find_participants(GSR_DIR)
    print(f"Validating {len(participants)} participants: {participants}")

    features = load_features(participants)
    diagnostics = compute_raw_signal_diagnostics(participants, args.tasks)
    merged = features.merge(diagnostics, on=["participant", "task"], how="left")

    merged["flags"] = merged.apply(classify, axis=1)

    out_cols = ["participant", "task", "scl_mean", "n_scr", "nscr_per_min",
                "scr_amplitude_mean", "raw_range_uS", "pct_variance_tonic", "flags"]
    out = merged[out_cols].sort_values(["participant", "task"])
    out_path = os.path.join(OUT_DIR, "gsr_scr_plausibility_validation.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(out.to_string(index=False))

    print("\n=== Flag frequency ===")
    all_flags = out["flags"].str.split(";").explode()
    print(all_flags.value_counts().to_string())

    print("\n=== Participants with a flag on ALL 3 tasks (systematic, not task-specific) ===")
    per_participant_all_flagged = out.groupby("participant")["flags"].apply(
        lambda s: (s != "PLAUSIBLE").all())
    systematic = per_participant_all_flagged[per_participant_all_flagged].index.tolist()
    print(systematic if systematic else "(none)")


if __name__ == "__main__":
    main()
