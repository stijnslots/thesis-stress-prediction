"""
Compares untrimmed versus trimmed PPG and HRV features.

Usage:
    python scripts/evaluation/04_ppg_trim_comparison.py
"""

import glob
import os

import pandas as pd

PPG_DIR = os.path.join("data", "processed", "ppg")
OUT_DIR = os.path.join("results", "eda_preliminary")
TASK_ORDER = ["stroop", "arithmetic", "face_recall"]
CORE_HRV = ["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50", "HRV_LFHF"]
SHORT_WINDOW_FLOOR_S = 60.0


def load_pair(participant):
    untrimmed_path = os.path.join(PPG_DIR, f"{participant}_ppg_features.csv")
    trimmed_path = os.path.join(PPG_DIR, f"{participant}_ppg_features_trimmed.csv")
    if not (os.path.exists(untrimmed_path) and os.path.exists(trimmed_path)):
        return None, None
    return pd.read_csv(untrimmed_path), pd.read_csv(trimmed_path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_untrimmed = sorted(glob.glob(os.path.join(PPG_DIR, "p*_ppg_features.csv")))
    all_untrimmed = [p for p in all_untrimmed if "trimmed" not in p and "rr_intervals" not in p]
    candidates = sorted({os.path.basename(p).split("_ppg_features")[0] for p in all_untrimmed},
                         key=lambda x: int(x[1:]))

    rows_untrimmed, rows_trimmed = [], []
    missing_trimmed = []
    for p in candidates:
        u, t = load_pair(p)
        if u is None:
            missing_trimmed.append(p)
            continue
        rows_untrimmed.append(u)
        rows_trimmed.append(t)

    if missing_trimmed:
        print(f"NOTE: {len(missing_trimmed)} participant(s) have untrimmed PPG but no trimmed version yet "
              f"(still processing?): {missing_trimmed}")

    untrimmed = pd.concat(rows_untrimmed, ignore_index=True)
    trimmed = pd.concat(rows_trimmed, ignore_index=True)
    n_participants = untrimmed["participant"].nunique()
    print(f"=== Scope ===")
    print(f"Vergelijking op basis van N={n_participants} participanten met zowel ongetrimde als "
          f"getrimde PPG-features ({len(untrimmed)} observaties elk).")

    # --- 4. Comparison table: mean per task, untrimmed vs trimmed, abs + % diff ---
    print(f"\n=== 4. Vergelijking kern-HRV: ongetrimd vs. getrimd (gemiddeld per taak) ===")
    comparison_rows = []
    for feat in CORE_HRV:
        u_mean = untrimmed.groupby("task", observed=True)[feat].mean()
        t_mean = trimmed.groupby("task", observed=True)[feat].mean()
        for task in TASK_ORDER:
            uv, tv = u_mean.get(task), t_mean.get(task)
            abs_diff = tv - uv
            pct_diff = 100 * abs_diff / uv if uv else float("nan")
            comparison_rows.append({"feature": feat, "task": task, "untrimmed_mean": uv,
                                     "trimmed_mean": tv, "abs_diff": abs_diff, "pct_diff": pct_diff})
    comp_df = pd.DataFrame(comparison_rows)
    with pd.option_context("display.max_rows", None, "display.width", 150):
        print(comp_df.to_string(index=False))

    comp_out = os.path.join(OUT_DIR, "ppg_trim_comparison.csv")
    comp_df.to_csv(comp_out, index=False)
    print(f"\nSaved: {comp_out}")

    # --- 5. Peak/RR count and short-window check ---
    print(f"\n=== 5. Aantal peaks en check op te korte taakvensters na trim ===")
    peaks_compare = untrimmed[["participant", "task", "n_peaks"]].merge(
        trimmed[["participant", "task", "n_peaks", "effective_duration_s"]],
        on=["participant", "task"], suffixes=("_untrimmed", "_trimmed"))
    peaks_compare["n_peaks_pct_change"] = 100 * (
        peaks_compare["n_peaks_trimmed"] - peaks_compare["n_peaks_untrimmed"]) / peaks_compare["n_peaks_untrimmed"]

    print("Gemiddelde afname in n_peaks per taak (ongetrimd -> getrimd):")
    print(peaks_compare.groupby("task", observed=True)[["n_peaks_untrimmed", "n_peaks_trimmed",
                                                          "n_peaks_pct_change"]].mean())

    short_flagged = peaks_compare[peaks_compare["effective_duration_s"] < SHORT_WINDOW_FLOOR_S]
    print(f"\nParticipant/taak-combinaties met effective_duration_s < {SHORT_WINDOW_FLOOR_S:.0f}s na trim "
          f"(potentieel onbetrouwbare HRV): {len(short_flagged)}/{len(peaks_compare)}")
    if len(short_flagged):
        print(short_flagged[["participant", "task", "effective_duration_s", "n_peaks_trimmed"]]
              .sort_values("effective_duration_s").to_string(index=False))
    else:
        print("Geen enkele participant/taak-combinatie valt onder de 60s-drempel na trim.")

    print("\nPer taak, kortste effective_duration_s na trim (worst case):")
    print(peaks_compare.groupby("task", observed=True)["effective_duration_s"].min())

    peaks_out = os.path.join(OUT_DIR, "ppg_trim_peaks_comparison.csv")
    peaks_compare.to_csv(peaks_out, index=False)
    print(f"\nSaved: {peaks_out}")


if __name__ == "__main__":
    main()
