"""Clean the NEISS surfing-injury export into an analysis-ready table.

Source: data/raw/NEISS_2010-2025_surfing.xlsx (CPSC National Electronic
Injury Surveillance System, surfing-related ER visits, USA, 2010-2025).
Output: data/processed/neiss_clean.csv
"""

import numpy as np
import pandas as pd

RAW_PATH = "data/raw/NEISS_2010-2025_surfing.xlsx"
OUT_PATH = "data/processed/neiss_clean.csv"

SEASON_BY_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall",
}

AGE_BINS = [0, 12, 18, 30, 45, 60, 120]
AGE_LABELS = ["0-12 (Child)", "13-18 (Teen)", "19-30 (Young Adult)",
              "31-45 (Adult)", "46-60 (Middle Age)", "60+ (Senior)"]

# NEISS groups the ~25 raw diagnosis codes into a smaller set of clinically
# similar buckets so the severity model isn't fit against 25 sparse levels.
DIAGNOSIS_GROUP = {
    "57 - FRACTURE": "Fracture",
    "55 - DISLOCATION": "Fracture",
    "50 - AMPUTATION": "Fracture",
    "59 - LACERATION": "Laceration/Puncture",
    "63 - PUNCTURE": "Laceration/Puncture",
    "56 - FOREIGN BODY": "Laceration/Puncture",
    "72 - AVULSION": "Laceration/Puncture",
    "53 - CONTUSIONS, ABR.": "Contusion/Sprain",
    "64 - STRAIN, SPRAIN": "Contusion/Sprain",
    "58 - HEMATOMA": "Contusion/Sprain",
    "54 - CRUSHING": "Contusion/Sprain",
    "52 - CONCUSSION": "Concussion/Head",
    "62 - INTERNAL INJURY": "Internal Injury",
    "69 - SUBMERSION": "Internal Injury",
    "61 - NERVE DAMAGE": "Internal Injury",
    "66 - HEMORRHAGE": "Internal Injury",
    "65 - ANOXIA": "Internal Injury",
}

# Severity is an ordinal proxy built from the disposition (what happened
# after treatment), which is the only outcome-like field NEISS provides.
SEVERITY_MAP = {
    "1 - TREATED/EXAMINED AND RELEASED": "Mild",
    "6 - LEFT WITHOUT BEING SEEN": "Mild",
    "2 - TREATED AND TRANSFERRED": "Moderate",
    "5 - HELD FOR OBSERVATION": "Moderate",
    "4 - TREATED AND ADMITTED/HOSPITALIZED": "Severe",
    "8 - FATALITY INCL. DOA, DIED IN ER": "Severe",
}


def strip_code(desc: str) -> str:
    """'31 - UPPER TRUNK' -> 'Upper Trunk'."""
    if pd.isna(desc):
        return np.nan
    text = desc.split(" - ", 1)[-1] if " - " in desc else desc
    return text.title()


def decode_age_years(age_code: int) -> float:
    """NEISS codes ages under 2 as 200 + months (e.g. 212 = 12 months)."""
    if age_code >= 200:
        return round((age_code - 200) / 12, 2)
    return float(age_code)


def load_raw() -> pd.DataFrame:
    return pd.read_excel(RAW_PATH, sheet_name="Surfing_Cases")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["case_number"] = df["CPSC_Case_Number"]
    out["date"] = pd.to_datetime(df["Treatment_Date"])
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["season"] = out["month"].map(SEASON_BY_MONTH)

    out["age_years"] = df["Age"].apply(decode_age_years)
    out["age_group"] = pd.cut(out["age_years"], bins=AGE_BINS, labels=AGE_LABELS, right=True)

    out["sex"] = df["Sex_Desc"].str.title()
    out["race"] = df["Race_Desc"].apply(strip_code)

    out["body_part"] = df["Body_Part_Desc"].apply(strip_code)
    out["diagnosis"] = df["Diagnosis_Desc"].apply(strip_code)
    out["diagnosis_group"] = df["Diagnosis_Desc"].map(DIAGNOSIS_GROUP).fillna("Other")

    out["disposition"] = df["Disposition_Desc"].apply(strip_code)
    out["severity"] = df["Disposition_Desc"].map(SEVERITY_MAP)
    out["severity"] = pd.Categorical(out["severity"], categories=["Mild", "Moderate", "Severe"], ordered=True)

    out["narrative"] = df["Narrative_1"]
    # National estimate weight (NEISS is a probability sample of US EDs);
    # keep it so charts can report weighted estimates, not just raw counts.
    out["weight"] = df["Weight"]

    return out.dropna(subset=["severity"]).reset_index(drop=True)


def main():
    raw = load_raw()
    clean_df = clean(raw)
    clean_df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(clean_df)} rows -> {OUT_PATH}")
    print(clean_df["severity"].value_counts())


if __name__ == "__main__":
    main()
