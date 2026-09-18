"""
Clean the Qualtrics background, export PSS-10 per-participant in a table.
(see notes/methodology_notes.md, "Qualtrics/PSS processing"):

    1. Exclude non-participant rows: any row with a missing/blank participant
       number, an unparseable ID ('wrong').
    2. Normalize participant IDs: strip whitespace, lowercase, and reduce to
       "p" + the trailing digit sequence (fixed inconsistencies).
    3. PSS-10 total score: Qualtrics coded responses as 1-5 instead of the
       labelled 0-4 (confirmed: min==1/max==5 on all 10 items, no exceptions).
       Subtract 1 from every item first, THEN reverse-score question 4, 5, 7, 8
       (4 - corrected_value) per standard PSS-10 scoring, then sum all 10.
    4. Recode Gender: 1.0 -> 'female', 2.0 -> 'male'.

Note: 75 of the 76 expected participants (p2-p76) are recovered; p1 has no Qualtrics
response at all. (in mid Oct there will be some more participants)
"""

import os
import re
import pandas as pd

RAW_PATH = os.path.join("data", "raw", "qualtrics", "NERVE_background.csv")
OUT_PATH = os.path.join("data", "processed", "qualtrics", "pss_demographics_cleaned.csv")

PSS_COLS = [f"Perceived_stress _{i}" for i in range(1, 11)]
REVERSE_ITEMS = {4, 5, 7, 8}  # 1-indexed PSS-10 item numbers, standard reverse-scored set

DEMOGRAPHIC_COLS = {
    "Ed_level": "ed_level",
    "Handedness": "handedness",
    "chronic_disease": "chronic_disease",
    "taking_medication": "taking_medication",
    "caffeinated_drinks": "caffeinated_drinks",
    "smoking": "smoking",
    "drinking": "drinking",
    "vision": "vision",
    "sleep": "sleep",
    "physical_act": "physical_act",
}

GENDER_MAP = {1.0: "female", 2.0: "male"}

EXPECTED_PARTICIPANTS = {f"p{i}" for i in range(1, 77)}

def normalize_id(raw_id):
    """Strip whitespace, lowercase, then reduce to 'p' + number."""
    s = str(raw_id).strip().lower()
    m = re.search(r"(\d+)$", s)
    if not m:
        return None
    return "p" + m.group(1)


def main():
    df = pd.read_csv(RAW_PATH, skiprows=[1, 2])
    df.columns = [c.strip() for c in df.columns]
    n_total = len(df)

    pid_raw = df["participantNumber"].astype(str).str.strip()
    is_blank = df["participantNumber"].isna()
    is_wrong = pid_raw.str.lower() == "wrong"
    is_abandoned = df["Finished"] == 0

    exclude_mask = is_blank | is_wrong | is_abandoned
    excluded = df.loc[exclude_mask, ["participantNumber", "Age", "Gender", "Finished", "Progress",
                                      "StartDate", "Duration (in seconds)"]]
    print(f"Loaded {n_total} raw rows from {RAW_PATH}")
    print(f"Excluding {exclude_mask.sum()} non-participant rows:")
    print(excluded.to_string())

    df = df.loc[~exclude_mask].copy()
    df["participant"] = df["participantNumber"].apply(normalize_id)

    unmapped = df[df["participant"].isna()]
    if len(unmapped):
        raise ValueError(f"{len(unmapped)} row(s) survived exclusion but could not be ID-normalized:\n{unmapped}")

    dupes = df["participant"][df["participant"].duplicated(keep=False)]
    if len(dupes):
        raise ValueError(f"Duplicate normalized participant IDs found: {sorted(dupes.unique())}")

    # PSS-10: Qualtrics coded 1-5 instead of the labelled 0-4.
    pss_corrected = df[PSS_COLS] - 1
    pss_scored = pss_corrected.copy()
    for i in REVERSE_ITEMS:
        col = f"Perceived_stress _{i}"
        pss_scored[col] = 4 - pss_corrected[col]
    df["pss_total_score"] = pss_scored.sum(axis=1, skipna=False)

    df["gender"] = df["Gender"].map(GENDER_MAP)
    unmapped_gender = df[df["Gender"].notna() & df["gender"].isna()]
    if len(unmapped_gender):
        raise ValueError(f"Gender value(s) outside the expected {{1.0, 2.0}} found:\n{unmapped_gender[['participant', 'Gender']]}")

    df["age"] = df["Age"]

    out_cols = ["participant", "age", "gender", "pss_total_score"] + list(DEMOGRAPHIC_COLS.values())
    result = df.rename(columns=DEMOGRAPHIC_COLS)[out_cols].copy()
    result["_sort_key"] = result["participant"].str.extract(r"(\d+)").astype(int)
    result = result.sort_values("_sort_key").drop(columns="_sort_key").reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    result.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(result)} participant rows -> {OUT_PATH}")
    print(f"PSS-10 total score range: {result['pss_total_score'].min():.0f}-{result['pss_total_score'].max():.0f} "
          f"(valid range 0-40)")
    print("\nGender counts:")
    print(result["gender"].value_counts())

    missing_expected = sorted(EXPECTED_PARTICIPANTS - set(result["participant"]), key=lambda x: int(x[1:]))
    print(f"\nExpected participants (p1-p76) NOT present in this file: {missing_expected}")
    if missing_expected == ["p1"]:
        print("CONFIRMED: p1 has no Qualtrics response (see methodology_notes.md, 'Qualtrics/PSS processing').")
        print("This file intentionally has NO row for p1 -- do not fill it in.")
        print("Downstream merges must use a LEFT/OUTER join against the full p1-p76 participant list")
        print("so p1 surfaces as NaN in age/gender/pss_total_score/etc., not as a silently dropped row.")
    else:
        raise ValueError(f"Expected exactly ['p1'] missing, got {missing_expected} -- investigate before proceeding.")


if __name__ == "__main__":
    main()
