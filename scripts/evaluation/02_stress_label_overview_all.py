"""
Overview of the stress-label distribution across all participants.

Usage:
    python scripts/evaluation/02_stress_label_overview_all.py
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd

LABELS_DIR = os.path.join("data", "processed", "labels")
OUT_DIR = os.path.join("results", "eda_preliminary")
OUT_PNG = os.path.join(OUT_DIR, "stress_label_overview_all.png")
TASK_ORDER = ["stroop", "arithmetic", "face_recall"]
VALID_RANGE = (0.0, 9.0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(LABELS_DIR, "p*_labels.csv")),
                   key=lambda p: int(os.path.basename(p).split("_")[0][1:]))
    if not paths:
        raise FileNotFoundError(f"No *_labels.csv files found in {LABELS_DIR}")

    frames = [pd.read_csv(p) for p in paths]
    all_labels = pd.concat(frames, ignore_index=True)
    all_labels["task"] = pd.Categorical(all_labels["task"], categories=TASK_ORDER, ordered=True)

    n_participants = all_labels["participant"].nunique()
    n_obs_total = len(all_labels)
    n_obs_valid = all_labels["stress_label"].notna().sum()

    print(f"=== 5. Scope ===")
    print(f"Gebaseerd op N={n_participants} participanten, {n_obs_total} observaties "
          f"({n_obs_valid} met een niet-missende stress_label).")
    if n_participants < 76:
        print(f"LET OP: dit zijn nog niet alle 76 participanten -- {76 - n_participants} ontbreken nog.")
    missing = all_labels[all_labels["stress_label"].isna()]
    if len(missing):
        print(f"{len(missing)} observatie(s) met missende stress_label:")
        print(missing[["participant", "task"]].to_string(index=False))

    # --- 1. Descriptive statistics per task + overall ---
    print(f"\n=== 1. Beschrijvende statistieken ===")
    desc_per_task = all_labels.groupby("task", observed=True)["stress_label"].agg(
        ["mean", "median", "std", "min", "max", "count"])
    print("Per taak:")
    print(desc_per_task)

    overall = all_labels["stress_label"].agg(["mean", "median", "std", "min", "max", "count"])
    print("\nOver de hele set (alle taken samen):")
    print(overall.to_string())

    # --- 4. Ceiling/floor effect ---
    print(f"\n=== 4. Plafond-/vloereffect ===")
    lo, hi = VALID_RANGE
    valid = all_labels["stress_label"].dropna()
    n_floor = int((valid == lo).sum())
    n_ceiling = int((valid == hi).sum())
    n_extreme = n_floor + n_ceiling
    print(f"Op vloerwaarde ({lo}): {n_floor}/{len(valid)} ({100*n_floor/len(valid):.1f}%)")
    print(f"Op plafondwaarde ({hi}): {n_ceiling}/{len(valid)} ({100*n_ceiling/len(valid):.1f}%)")
    print(f"Totaal op uiterste waarden: {n_extreme}/{len(valid)} ({100*n_extreme/len(valid):.1f}%)")

    print("\nPer taak:")
    for t in TASK_ORDER:
        sub = all_labels.loc[all_labels["task"] == t, "stress_label"].dropna()
        nf = int((sub == lo).sum())
        nc = int((sub == hi).sum())
        print(f"  {t}: vloer={nf}/{len(sub)} ({100*nf/len(sub):.1f}%), "
              f"plafond={nc}/{len(sub)} ({100*nc/len(sub):.1f}%), "
              f"totaal uiterst={100*(nf+nc)/len(sub):.1f}%")

    # --- 2 & 3. Plots: 3 histograms + 1 boxplot, combined figure ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    bins = list(range(0, 11))  # 0..9 inclusive, integer-width bins

    for ax, t in zip(axes[:3], TASK_ORDER):
        sub = all_labels.loc[all_labels["task"] == t, "stress_label"].dropna()
        ax.hist(sub, bins=bins, edgecolor="black", align="left", rwidth=0.9)
        ax.set_title(f"{t} (n={len(sub)})")
        ax.set_xlabel("stress score")
        ax.set_ylabel("amount")
        ax.set_xticks(range(0, 10))
        ax.yaxis.set_major_locator(MultipleLocator(2))

    box_data = [all_labels.loc[all_labels["task"] == t, "stress_label"].dropna().values for t in TASK_ORDER]
    axes[3].boxplot(box_data, tick_labels=TASK_ORDER)
    axes[3].set_ylabel("stress score")
    axes[3].set_title("Boxplot per taak")

    fig.suptitle(f"stress_label verdeling -- N={n_participants} participanten, {n_obs_valid} observaties "
                 f"({'alle 76' if n_participants == 76 else f'nog niet alle 76, {76-n_participants} ontbreken'})")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {OUT_PNG}")


if __name__ == "__main__":
    main()
