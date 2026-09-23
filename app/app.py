import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import chi2_contingency

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.model import BODY_PART_GROUP, FEATURES  # noqa: E402
from src.stats import fit_severity_logit  # noqa: E402
from geo import STATE_CENTROIDS  # noqa: E402

st.set_page_config(page_title="Surfing Accidents & Ocean Safety Dashboard", layout="wide", page_icon="🏄")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Ocean-themed palette so charts read as one system rather than default
# plotly rainbow colors.
OCEAN_COLORS = ["#0f6f8c", "#f2a154", "#2a9d8f", "#e76f51", "#264653", "#8ecae6"]
SEVERITY_COLORS = {"Mild": "#8ecae6", "Moderate": "#f2a154", "Severe": "#e76f51"}
px.defaults.color_discrete_sequence = OCEAN_COLORS
px.defaults.template = "plotly_dark"


@st.cache_data
def load_neiss() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "neiss_clean.csv", parse_dates=["date"])
    df["body_part_group"] = df["body_part"].map(BODY_PART_GROUP).fillna("Other")
    return df


@st.cache_data
def load_sharks() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "shark_attacks_us_clean.csv", parse_dates=["date"])
    return df


@st.cache_resource
def load_model():
    pipeline = joblib.load(DATA_DIR / "severity_model.joblib")
    with open(DATA_DIR / "model_metrics.json") as f:
        metrics = json.load(f)
    return pipeline, metrics


neiss = load_neiss()
sharks = load_sharks()
pipeline, metrics_raw = load_model()

st.title("🏄 Surfing Accidents & Ocean Safety Dashboard")
st.caption(
    "Scope: USA only. (1) NEISS (US emergency-room surveillance, surfing injuries 2010-2025, "
    f"{len(neiss):,} cases) (2) Global Shark Attack File (US only, surf-related "
    f"{sharks['surf_related'].sum():,} cases, 1900s-2016)."
)

with st.sidebar:
    st.header("Filters")
    st.caption("NEISS data (Overview & Detailed Analysis tabs)")
    year_min, year_max = int(neiss["year"].min()), int(neiss["year"].max())
    year_range = st.slider("Year range (NEISS)", year_min, year_max, (year_min, year_max))

    st.divider()
    st.caption("Shark attack data (map on the Detailed Analysis tab)")
    state_options = sorted(sharks.loc[sharks["surf_related"], "state"].dropna().unique())
    selected_states = st.multiselect("State (shark attack map)", options=state_options, default=state_options)

neiss_f = neiss[(neiss["year"] >= year_range[0]) & (neiss["year"] <= year_range[1])]

tab_overview, tab_detail, tab_model = st.tabs(["Overview", "Detailed Analysis", "Prediction Model"])

# ---------------------------------------------------------------- Overview
with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("NEISS cases", f"{len(neiss_f):,}")
    severe_pct = (neiss_f["severity"] == "Severe").mean() * 100
    col2.metric("Severe (hospitalized/fatal) rate", f"{severe_pct:.1f}%")
    col3.metric("Shark attacks while surfing", f"{int(sharks['surf_related'].sum()):,}")
    fatal_surf = sharks.loc[sharks["surf_related"], "fatal"].eq("Y").sum()
    col4.metric("Of which fatal", f"{int(fatal_surf)}")

    yearly = neiss_f.groupby("year").size().reset_index(name="count")
    fig_yearly = px.line(yearly, x="year", y="count", markers=True,
                          title="NEISS: yearly surfing injury count (USA)")
    fig_yearly.update_traces(line_color=OCEAN_COLORS[0])
    st.plotly_chart(fig_yearly, width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        sev_counts = neiss_f["severity"].value_counts().reindex(["Mild", "Moderate", "Severe"])
        sev_df = sev_counts.rename_axis("severity").reset_index(name="count")
        fig_sev = px.bar(sev_df, x="severity", y="count", color="severity",
                          color_discrete_map=SEVERITY_COLORS, title="Severity breakdown")
        st.plotly_chart(fig_sev, width="stretch")
    with col_b:
        season_counts = neiss_f["season"].value_counts()
        fig_season = px.pie(season_counts, values=season_counts.values, names=season_counts.index,
                             title="Injuries by season")
        st.plotly_chart(fig_season, width="stretch")

    st.markdown(
        "**Takeaway**: over 90% of cases are mild — treated and released — but "
        f"about {severe_pct:.0f}% are severe, resulting in hospitalization or death. Cases aren't "
        "concentrated in summer, suggesting surfers are active year-round."
    )

# ------------------------------------------------------------- Detail tab
with tab_detail:
    st.subheader("NEISS: factors affecting severity")

    filter_col1, filter_col2 = st.columns(2)
    seasons = filter_col1.multiselect("Season", options=neiss_f["season"].unique(),
                                       default=list(neiss_f["season"].unique()))
    age_groups = filter_col2.multiselect("Age group", options=neiss_f["age_group"].dropna().unique(),
                                          default=list(neiss_f["age_group"].dropna().unique()))
    filtered = neiss_f[neiss_f["season"].isin(seasons) & neiss_f["age_group"].isin(age_groups)]

    col_c, col_d = st.columns(2)
    with col_c:
        body_part_sev = filtered.groupby(["body_part_group", "severity"], observed=True).size().reset_index(name="count")
        fig_bp = px.bar(body_part_sev, x="body_part_group", y="count", color="severity",
                         color_discrete_map=SEVERITY_COLORS, title="Body part group x severity", barmode="stack")
        st.plotly_chart(fig_bp, width="stretch")
    with col_d:
        age_sev = filtered.groupby(["age_group", "severity"], observed=True).size().reset_index(name="count")
        fig_age = px.bar(age_sev, x="age_group", y="count", color="severity",
                          color_discrete_map=SEVERITY_COLORS, title="Age group x severity", barmode="stack")
        st.plotly_chart(fig_age, width="stretch")

    st.markdown("#### Statistical test: is severity related to season?")
    contingency = pd.crosstab(filtered["season"], filtered["severity"])
    if contingency.shape[0] > 1 and contingency.shape[1] > 1:
        chi2, p_value, dof, _ = chi2_contingency(contingency)
        verdict = "a statistically significant association was found" if p_value < 0.05 else "no statistically significant association was found"
        st.write(f"Chi-square test: p-value = {p_value:.3f} -> {verdict} (alpha = 0.05)")
    else:
        st.write("Not enough data under the current filters to run this test.")

    st.markdown("#### Age group x season: severity-rate heatmap")
    rate = (
        filtered.assign(is_severe=(filtered["severity"] == "Severe").astype(int))
        .pivot_table(index="age_group", columns="season", values="is_severe", aggfunc="mean", observed=True)
        * 100
    )
    if not rate.empty:
        fig_heat = px.imshow(rate.round(1), text_auto=True, color_continuous_scale="Sunset",
                              labels=dict(color="Severity rate (%)"), title="Severity rate (%) by age group x season")
        st.plotly_chart(fig_heat, width="stretch")

    st.divider()
    st.subheader("Factors associated with severity (logistic regression / odds ratios)")
    st.caption(
        "This model explains the probability of a severe (hospitalized/fatal) outcome using only "
        "factors known at the time of the accident — age, sex, season, and body part — without using "
        "the diagnosis. An odds ratio above 1 means higher odds of severity; if the confidence interval "
        "crosses 1, the effect isn't statistically significant. Reference categories: sex = Female, "
        "season = Fall, body part = Arm/Hand."
    )
    logit_df = neiss_f.dropna(subset=["age_years", "sex", "season", "body_part_group", "severity"])
    try:
        odds_df = fit_severity_logit(logit_df)
        fig_forest = px.scatter(
            odds_df, x="odds_ratio", y="term", log_x=True,
            error_x=odds_df["ci_high"] - odds_df["odds_ratio"],
            error_x_minus=odds_df["odds_ratio"] - odds_df["ci_low"],
            title="Odds ratio for a severe outcome (95% CI)",
            labels={"odds_ratio": "Odds ratio (log scale)", "term": ""},
        )
        fig_forest.add_vline(x=1, line_dash="dash", line_color="gray")
        fig_forest.update_traces(marker=dict(size=10, color=OCEAN_COLORS[0]))
        st.plotly_chart(fig_forest, width="stretch")
    except Exception as e:
        st.warning(f"Couldn't fit the model under the current filters ({e}).")

    st.divider()
    st.subheader("Shark attack map (USA, surf-related)")
    st.caption("Only state names are recorded, with no coordinates, so points are scattered randomly near "
               "each state's centroid. These are not exact incident locations.")

    surf_sharks = sharks[sharks["surf_related"]].dropna(subset=["state"]).copy()
    surf_sharks = surf_sharks[surf_sharks["state"].isin(STATE_CENTROIDS)]
    surf_sharks = surf_sharks[surf_sharks["state"].isin(selected_states)]

    if surf_sharks.empty:
        st.info("No surf-related shark attack data for the selected states. Try adjusting the state filter in the sidebar.")
    else:
        rng = np.random.default_rng(42)
        lat, lon = zip(*surf_sharks["state"].map(STATE_CENTROIDS))
        surf_sharks["lat"] = np.array(lat) + rng.normal(0, 0.4, len(surf_sharks))
        surf_sharks["lon"] = np.array(lon) + rng.normal(0, 0.4, len(surf_sharks))

        fig_map = px.scatter_geo(
            surf_sharks, lat="lat", lon="lon", scope="usa",
            hover_name="state",
            hover_data={"date": True, "location": True, "injury": True, "fatal": True, "lat": False, "lon": False},
            color="fatal", color_discrete_map={"Y": "#d62728", "N": "#1f77b4"},
            title="Shark attacks while surfing (USA, by state, all years)",
        )
        fig_map.update_geos(
            showland=True, landcolor="#e8e4d8", showlakes=True, lakecolor="#cfe3f7",
            showsubunits=True, subunitcolor="#9a9a9a", showcountries=True,
        )
        fig_map.update_traces(marker=dict(size=9, opacity=0.8, line=dict(width=0.5, color="white")))
        st.plotly_chart(fig_map, width="stretch")

        state_counts = surf_sharks["state"].value_counts().reset_index()
        state_counts.columns = ["state", "count"]
        st.plotly_chart(px.bar(state_counts, x="state", y="count", title="Shark attacks while surfing by state"),
                         width="stretch")

# ------------------------------------------------------------- Model tab
with tab_model:
    st.subheader("Severity prediction model")
    st.caption(
        "A random forest classifier trained on the NEISS data. Predicts mild/moderate/severe from age, "
        "sex, season, body part, and diagnosis. The moderate class has only 21 cases, so its accuracy "
        "is limited."
    )

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.slider("Age", 2, 90, 30)
        sex = c1.selectbox("Sex", neiss["sex"].dropna().unique())
        season = c2.selectbox("Season", neiss["season"].dropna().unique())
        body_part_group = c2.selectbox("Body part", sorted(neiss["body_part_group"].dropna().unique()))
        diagnosis_group = c3.selectbox("Diagnosis category", sorted(neiss["diagnosis_group"].dropna().unique()))
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([{
            "age_years": age, "sex": sex, "season": season,
            "body_part_group": body_part_group, "diagnosis_group": diagnosis_group,
        }])[FEATURES]
        proba = pipeline.predict_proba(input_df)[0]
        classes = pipeline.named_steps["clf"].classes_
        result = pd.DataFrame({"severity": classes, "probability": proba}).sort_values("probability", ascending=False)
        st.plotly_chart(px.bar(result, x="severity", y="probability", title="Prediction"), width="stretch")

    st.divider()
    st.subheader("Model evaluation (test set)")
    report = metrics_raw["classification_report"]
    report_df = pd.DataFrame(report).T.round(2)
    st.dataframe(report_df)

    importances = pipeline.named_steps["clf"].feature_importances_
    feat_names = [n.split("__", 1)[-1].replace("_", " ") for n in pipeline.named_steps["preprocess"].get_feature_names_out()]
    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False).head(15)
    st.plotly_chart(px.bar(imp_df, x="importance", y="feature", orientation="h", title="Feature importance"),
                     width="stretch")
