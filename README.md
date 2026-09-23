# Surfing Accidents & Ocean Safety Dashboard

A data analysis project that aggregates and analyzes US surfing accident data to explore what drives injury severity. Built with Streamlit into an interactive dashboard covering trend visualization, statistical testing, and machine-learning severity prediction.

## Live site

[https://surf-safety-dashboard.onrender.com](https://surf-safety-dashboard.onrender.com) (hosted on Render's free tier — the instance sleeps after inactivity, so the first load can take 20-30 seconds)

## Data sources

| Dataset | Content | Coverage | Source |
|---|---|---|---|
| NEISS | US emergency-room (ER) records for surfing-related injuries | 2010-2025, 1,850 cases | [CPSC National Electronic Injury Surveillance System](https://www.cpsc.gov/Research--Statistics/NEISS-Injury-Data) |
| Global Shark Attack File | Shark attack records (filtered to the US, 695 of which are surf-related) | ~1900s-2016 | Mirror of [sharkattackfile.net](http://sharkattackfile.net/) ([GitHub](https://github.com/mariobru/global-shark-attacks)) |

**Limitations of the data**
- Both datasets cover the US only, not the whole world.
- The Global Shark Attack File is a snapshot through roughly 2016 and doesn't include recent years.
- Shark attack locations are recorded only as state names, with no coordinates, so map points are scattered randomly near each state's centroid for visualization. They do not represent exact incident locations.
- NEISS is not a full census — it's a weighted sample from a subset of US hospitals (the `Weight` column carries the weight used for national estimates).

## Key findings

Analyzing the NEISS data (1,850 cases) surfaced the following patterns (via the logistic regression / odds ratios on the Detailed Analysis tab).

- **Age is the strongest, most consistent driver:** each additional year of age raises the odds of a severe outcome by about 3.5% (odds ratio 1.04, p<0.001). Severity rates for those 60+ reach 11-27%, several times higher than younger age groups (mostly under 5%).
- **Age and season interact noticeably:** among surfers 60+, the summer severity rate is 27% and spring is 20% — more than double fall/winter (~11%). This seasonal gap doesn't show up in younger groups, hinting at differences in wave conditions older surfers ride, or in physical resilience.
- **Body part injured strongly affects severity:** trunk injuries carry roughly 6.4x the odds of a severe outcome compared to arm/hand injuries (p<0.001); head/face/neck injuries carry roughly 3.4x (p<0.01). Impacts to the trunk or head are more likely to lead to hospitalization.
- **Sex differences:** men have roughly 2.0x the odds of a severe outcome compared to women (p=0.02). The sample sizes differ (1,395 men vs. 455 women), but the association is still statistically significant.
- **Season alone isn't significant:** a chi-square test of season against severity gives p=0.579 — no clear standalone relationship (the age x season interaction above appears to matter more).
- **Limits of the severity prediction model:** the random forest's 3-class classification reaches 76% overall accuracy, but the moderate class (21 cases) is effectively unclassifiable, and precision for the severe class is a low 0.18. Age, body part, and season alone aren't enough to reliably predict severity — mechanism of injury and medical history, which aren't in NEISS, are likely important missing pieces.

## Methodology

1. **Cleaning:** standardizes inconsistent formatting (columns mixing codes and text, free-text ages, two-digit years) and builds features like season, age group, and severity (`src/clean_neiss.py`, `src/clean_shark_attacks.py`).
2. **Descriptive statistics & testing:** aggregates severity distributions by body part, age group, and season; runs a chi-square test for season vs. severity; builds a severity-rate heatmap from an age x season cross-tab.
3. **Explanatory model (logistic regression):** explains the odds of a severe outcome using only circumstantial factors known at the time of the accident (age, sex, season, body part) — deliberately excluding diagnosis (`src/stats.py`). Visualizes odds ratios and 95% confidence intervals as a forest plot.
4. **Prediction model (random forest):** predicts severity (mild/moderate/severe) using age, sex, season, and body part plus diagnosis category (`src/model.py`). The moderate class has few samples (21), so accuracy there is limited.

The explanatory and prediction models deliberately use different feature sets: including diagnosis category (fracture, laceration, etc.) in the explanatory model would produce a near-tautological result ("it's severe because it's a fracture"). The factor-analysis model is designed to explain severity without relying on the diagnosis itself.

## Dashboard structure

- **Overview:** KPIs like total case count and severe-case rate, the yearly trend, and a breakdown by severity and season. The sidebar's year slider filters the time range.
- **Detailed Analysis:** severity by body part and age group, a chi-square test of season vs. severity, an age x season severity-rate heatmap, an odds-ratio forest plot, and a state-level shark attack map (filterable via the sidebar's state selector, with hover for individual incidents).
- **Prediction Model:** enter age, sex, season, body part, and diagnosis category to see predicted severity probabilities, along with model evaluation metrics and feature importances.

## Directory structure

```
data/
  raw/            source data (xlsx, csv)
  processed/      cleaned CSVs and the trained model
src/
  clean_neiss.py          NEISS data cleaning
  clean_shark_attacks.py  shark attack data cleaning
  stats.py                logistic regression for severity factors (odds ratios)
  model.py                severity prediction model (random forest) training
app/
  app.py          the Streamlit dashboard
  geo.py          state centroid coordinates (for the map)
```

## Setup and usage

```bash
pip install -r requirements.txt

python src/clean_neiss.py
python src/clean_shark_attacks.py
python src/model.py

streamlit run app/app.py
```

## Future directions

- Adding global-scale data (wave data and break type by country) could extend this into the originally-envisioned global map analysis.
- The severe class has few samples; addressing the class imbalance with techniques like SMOTE could improve model accuracy.
