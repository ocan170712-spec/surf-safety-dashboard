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

st.set_page_config(page_title="サーフィン事故と海の安全 ダッシュボード", layout="wide", page_icon="🏄")

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

st.title("🏄 サーフィン事故と海の安全 ダッシュボード")
st.caption(
    "データ範囲: 米国のみ。① NEISS(米国救急外来サーベイランス, サーフィン受傷 2010-2025年, "
    f"{len(neiss):,}件) ② Global Shark Attack File(米国分, サーフィン関連 "
    f"{sharks['surf_related'].sum():,}件, 1900年代〜2016年)。"
)

with st.sidebar:
    st.header("フィルター")
    st.caption("NEISSデータ(概要・詳細分析タブ)")
    year_min, year_max = int(neiss["year"].min()), int(neiss["year"].max())
    year_range = st.slider("対象年(NEISS)", year_min, year_max, (year_min, year_max))

    st.divider()
    st.caption("サメ被害データ(詳細分析タブの地図)")
    state_options = sorted(sharks.loc[sharks["surf_related"], "state"].dropna().unique())
    selected_states = st.multiselect("州(シャーク被害マップ)", options=state_options, default=state_options)

neiss_f = neiss[(neiss["year"] >= year_range[0]) & (neiss["year"] <= year_range[1])]

tab_overview, tab_detail, tab_model = st.tabs(["概要", "詳細分析", "予測モデル"])

# ---------------------------------------------------------------- Overview
with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("NEISS 受診件数", f"{len(neiss_f):,}")
    severe_pct = (neiss_f["severity"] == "Severe").mean() * 100
    col2.metric("重症(入院・死亡)割合", f"{severe_pct:.1f}%")
    col3.metric("サーフィン中のサメ被害", f"{int(sharks['surf_related'].sum()):,}")
    fatal_surf = sharks.loc[sharks["surf_related"], "fatal"].eq("Y").sum()
    col4.metric("うち死亡", f"{int(fatal_surf)}")

    yearly = neiss_f.groupby("year").size().reset_index(name="件数")
    fig_yearly = px.line(yearly, x="year", y="件数", markers=True,
                          title="NEISS: 年別サーフィン受傷件数の推移(米国)")
    fig_yearly.update_traces(line_color=OCEAN_COLORS[0])
    st.plotly_chart(fig_yearly, width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        sev_counts = neiss_f["severity"].value_counts().reindex(["Mild", "Moderate", "Severe"])
        sev_df = sev_counts.rename_axis("重症度").reset_index(name="件数")
        fig_sev = px.bar(sev_df, x="重症度", y="件数", color="重症度",
                          color_discrete_map=SEVERITY_COLORS, title="重症度の内訳")
        st.plotly_chart(fig_sev, width="stretch")
    with col_b:
        season_counts = neiss_f["season"].value_counts()
        fig_season = px.pie(season_counts, values=season_counts.values, names=season_counts.index,
                             title="季節別の受傷件数")
        st.plotly_chart(fig_season, width="stretch")

    st.markdown(
        "**読み方**: 全体の9割以上は治療後に帰宅する軽症ですが、"
        f"約{severe_pct:.0f}%は入院や死亡に至る重症です。件数は夏に集中しておらず、"
        "季節を問わずサーファーが活動していることがうかがえます。"
    )

# ------------------------------------------------------------- Detail tab
with tab_detail:
    st.subheader("NEISS: 重症度に影響する要因")

    filter_col1, filter_col2 = st.columns(2)
    seasons = filter_col1.multiselect("季節", options=neiss_f["season"].unique(),
                                       default=list(neiss_f["season"].unique()))
    age_groups = filter_col2.multiselect("年齢層", options=neiss_f["age_group"].dropna().unique(),
                                          default=list(neiss_f["age_group"].dropna().unique()))
    filtered = neiss_f[neiss_f["season"].isin(seasons) & neiss_f["age_group"].isin(age_groups)]

    col_c, col_d = st.columns(2)
    with col_c:
        body_part_sev = filtered.groupby(["body_part_group", "severity"], observed=True).size().reset_index(name="件数")
        fig_bp = px.bar(body_part_sev, x="body_part_group", y="件数", color="severity",
                         color_discrete_map=SEVERITY_COLORS, title="部位グループ x 重症度", barmode="stack")
        st.plotly_chart(fig_bp, width="stretch")
    with col_d:
        age_sev = filtered.groupby(["age_group", "severity"], observed=True).size().reset_index(name="件数")
        fig_age = px.bar(age_sev, x="age_group", y="件数", color="severity",
                          color_discrete_map=SEVERITY_COLORS, title="年齢層 x 重症度", barmode="stack")
        st.plotly_chart(fig_age, width="stretch")

    st.markdown("#### 統計的検定: 季節と重症度に関係はある?")
    contingency = pd.crosstab(filtered["season"], filtered["severity"])
    if contingency.shape[0] > 1 and contingency.shape[1] > 1:
        chi2, p_value, dof, _ = chi2_contingency(contingency)
        verdict = "統計的に有意な関連が見られます" if p_value < 0.05 else "統計的に有意な関連は見られません"
        st.write(f"カイ二乗検定: p値 = {p_value:.3f} → {verdict}(有意水準5%)")
    else:
        st.write("選択したフィルタではデータが不足しており検定できません。")

    st.markdown("#### 年齢層 x 季節: 重症化率のヒートマップ")
    rate = (
        filtered.assign(is_severe=(filtered["severity"] == "Severe").astype(int))
        .pivot_table(index="age_group", columns="season", values="is_severe", aggfunc="mean", observed=True)
        * 100
    )
    if not rate.empty:
        fig_heat = px.imshow(rate.round(1), text_auto=True, color_continuous_scale="Sunset",
                              labels=dict(color="重症化率(%)"), title="年齢層 x 季節ごとの重症化率(%)")
        st.plotly_chart(fig_heat, width="stretch")

    st.divider()
    st.subheader("重症化に関わる要因(ロジスティック回帰・オッズ比)")
    st.caption(
        "診断結果を使わず、事故発生時点でわかる要因(年齢・性別・季節・受傷部位)だけで"
        "「重症(入院・死亡)になる確率」を説明するモデルです。オッズ比が1より大きいほど重症化しやすく、"
        "信頼区間が1をまたぐ場合は統計的に有意とは言えません。基準カテゴリ: 性別=Female, 季節=Fall, 受傷部位=Arm/Hand。"
    )
    logit_df = neiss_f.dropna(subset=["age_years", "sex", "season", "body_part_group", "severity"])
    try:
        odds_df = fit_severity_logit(logit_df)
        fig_forest = px.scatter(
            odds_df, x="odds_ratio", y="term", log_x=True,
            error_x=odds_df["ci_high"] - odds_df["odds_ratio"],
            error_x_minus=odds_df["odds_ratio"] - odds_df["ci_low"],
            title="重症化オッズ比(95%信頼区間)",
            labels={"odds_ratio": "オッズ比(対数軸)", "term": ""},
        )
        fig_forest.add_vline(x=1, line_dash="dash", line_color="gray")
        fig_forest.update_traces(marker=dict(size=10, color=OCEAN_COLORS[0]))
        st.plotly_chart(fig_forest, width="stretch")
    except Exception as e:
        st.warning(f"選択したフィルタではモデルを推定できませんでした({e})。")

    st.divider()
    st.subheader("シャーク被害マップ(米国, サーフィン関連)")
    st.caption("州名しか記録されていないため、点は各州の中心付近にランダムに散らしています。正確な発生地点ではありません。")

    surf_sharks = sharks[sharks["surf_related"]].dropna(subset=["state"]).copy()
    surf_sharks = surf_sharks[surf_sharks["state"].isin(STATE_CENTROIDS)]
    surf_sharks = surf_sharks[surf_sharks["state"].isin(selected_states)]

    if surf_sharks.empty:
        st.info("選択した州にはサーフィン関連のサメ被害データがありません。サイドバーの州選択を見直してください。")
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
            title="サーフィン中のサメ被害(米国, 州別・年代不問)",
        )
        fig_map.update_geos(
            showland=True, landcolor="#e8e4d8", showlakes=True, lakecolor="#cfe3f7",
            showsubunits=True, subunitcolor="#9a9a9a", showcountries=True,
        )
        fig_map.update_traces(marker=dict(size=9, opacity=0.8, line=dict(width=0.5, color="white")))
        st.plotly_chart(fig_map, width="stretch")

        state_counts = surf_sharks["state"].value_counts().reset_index()
        state_counts.columns = ["州", "件数"]
        st.plotly_chart(px.bar(state_counts, x="州", y="件数", title="州別 サーフィン中のサメ被害件数"),
                         width="stretch")

# ------------------------------------------------------------- Model tab
with tab_model:
    st.subheader("重症度予測モデル")
    st.caption(
        "NEISSデータでランダムフォレスト分類器を学習。年齢・性別・季節・受傷部位・診断内容から、"
        "軽症/中等症/重症を予測します。中等症はデータが21件と非常に少なく、精度は限定的です。"
    )

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.slider("年齢", 2, 90, 30)
        sex = c1.selectbox("性別", neiss["sex"].dropna().unique())
        season = c2.selectbox("季節", neiss["season"].dropna().unique())
        body_part_group = c2.selectbox("受傷部位", sorted(neiss["body_part_group"].dropna().unique()))
        diagnosis_group = c3.selectbox("診断分類", sorted(neiss["diagnosis_group"].dropna().unique()))
        submitted = st.form_submit_button("予測する")

    if submitted:
        input_df = pd.DataFrame([{
            "age_years": age, "sex": sex, "season": season,
            "body_part_group": body_part_group, "diagnosis_group": diagnosis_group,
        }])[FEATURES]
        proba = pipeline.predict_proba(input_df)[0]
        classes = pipeline.named_steps["clf"].classes_
        result = pd.DataFrame({"重症度": classes, "確率": proba}).sort_values("確率", ascending=False)
        st.plotly_chart(px.bar(result, x="重症度", y="確率", title="予測結果"), width="stretch")

    st.divider()
    st.subheader("モデル評価(テストデータ)")
    report = metrics_raw["classification_report"]
    report_df = pd.DataFrame(report).T.round(2)
    st.dataframe(report_df)

    importances = pipeline.named_steps["clf"].feature_importances_
    feat_names = [n.split("__", 1)[-1].replace("_", " ") for n in pipeline.named_steps["preprocess"].get_feature_names_out()]
    imp_df = pd.DataFrame({"特徴量": feat_names, "重要度": importances}).sort_values("重要度", ascending=False).head(15)
    st.plotly_chart(px.bar(imp_df, x="重要度", y="特徴量", orientation="h", title="特徴量の重要度"),
                     width="stretch")
