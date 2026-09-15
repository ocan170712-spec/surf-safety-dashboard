"""Clean the Global Shark Attack File export down to a usable USA table.

Source: data/raw/global_shark_attacks.csv (public mirror of the GSAF /
Kaggle "Global Shark Attacks" dataset, notoriously messy: inconsistent
casing, free-text ages, duplicate/blank trailing columns).
Output: data/processed/shark_attacks_us_clean.csv (USA rows only, with a
`surf_related` flag for the surfing/board-sport subset).
"""

import re

import numpy as np
import pandas as pd

RAW_PATH = "data/raw/global_shark_attacks.csv"
OUT_PATH = "data/processed/shark_attacks_us_clean.csv"

# US state postal codes, used to build a state-level choropleth in the app.
STATE_ABBREV = {
    "Florida": "FL", "California": "CA", "Hawaii": "HI", "South Carolina": "SC",
    "North Carolina": "NC", "Texas": "TX", "Oregon": "OR", "New Jersey": "NJ",
    "New York": "NY", "Georgia": "GA", "Virginia": "VA", "Alabama": "AL",
    "Massachusetts": "MA", "Washington": "WA", "Puerto Rico": "PR",
    "Louisiana": "LA", "Mississippi": "MS", "Delaware": "DE", "Maine": "ME",
    "Connecticut": "CT", "Rhode Island": "RI", "Maryland": "MD",
}

SURF_KEYWORDS = ["surf", "boogie board", "body board", "skim board",
                 "kitesurf", "windsurf"]
NOT_SURF_KEYWORDS = ["surf fishing", "surf casting", "surf ski"]


def extract_age(raw_age) -> float:
    """'12 or 13' -> 12, '60s' -> 60, 'Teen' -> NaN."""
    if pd.isna(raw_age):
        return np.nan
    match = re.search(r"\d+", str(raw_age))
    return float(match.group()) if match else np.nan


def classify_surf_related(activity: str) -> bool:
    if pd.isna(activity):
        return False
    text = activity.lower()
    if any(bad in text for bad in NOT_SURF_KEYWORDS):
        return False
    return any(kw in text for kw in SURF_KEYWORDS)


def parse_date(day_month: str, year: int):
    """Rebuild the date from the day-month text plus the trusted Year column,
    since the raw file's 2-digit years are ambiguous ('76' vs '76-Mon-01')."""
    if pd.isna(day_month) or pd.isna(year) or year <= 0:
        return pd.NaT
    match = re.match(r"(\d{1,2})-([A-Za-z]{3})", str(day_month).strip())
    if not match:
        return pd.NaT
    day, mon = match.groups()
    try:
        return pd.to_datetime(f"{day}-{mon}-{int(year)}", format="%d-%b-%Y")
    except (ValueError, TypeError):
        return pd.NaT


def clean_fatal(value) -> str:
    text = str(value).strip().upper()
    return text if text in ("Y", "N") else np.nan


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, encoding="latin1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Country"] = df["Country"].astype(str).str.strip().str.upper()
    usa = df[df["Country"] == "USA"].copy()

    out = pd.DataFrame()
    out["year"] = usa["Year"]
    out["date"] = [parse_date(d, y) for d, y in zip(usa["Date"], usa["Year"])]
    out["month"] = out["date"].dt.month
    out["state"] = usa["Area"].astype(str).str.strip()
    out["state_abbrev"] = out["state"].map(STATE_ABBREV)
    out["location"] = usa["Location"]
    out["activity"] = usa["Activity"].astype(str).str.strip()
    out["surf_related"] = out["activity"].apply(classify_surf_related)
    out["age"] = usa["Age"].apply(extract_age)
    out["sex"] = usa["Sex"].astype(str).str.strip().str.upper().replace({"NAN": np.nan})
    out["injury"] = usa["Injury"]
    out["fatal"] = usa["Fatal (Y/N)"].apply(clean_fatal)
    out["attack_type"] = usa["Type"].astype(str).str.strip()

    out = out[out["year"] > 0]
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def main():
    raw = load_raw()
    clean_df = clean(raw)
    clean_df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(clean_df)} USA rows -> {OUT_PATH}")
    print(f"Surf-related: {clean_df['surf_related'].sum()}")
    print(clean_df.loc[clean_df["surf_related"], "state"].value_counts())


if __name__ == "__main__":
    main()
