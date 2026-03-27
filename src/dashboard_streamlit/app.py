import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import io

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import matplotlib.pyplot as plt
import xgboost


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Global Governance Risk Explorer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Constants
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
DEFAULT_PATH = APP_DIR.parents[1] / "data" / "raw" / "actual" / "predictions_ssp.csv"
MODELS_DIR = APP_DIR.parents[1] / "models"

POVERTY_LABELS = {
    "poverty_3usd": "PHR at 3 USD/day [% of population]",
    "poverty_8_3usd": "PHR at 8.3 USD/day [% of population]",
    "poverty_10usd": "PHR at 10 USD/day [% of population]",
}

SCENARIO_LABELS = {
    "SSP1": "Sustainability",
    "SSP2": "Middle of the Road",
    "SSP3": "Regional Rivalry",
    "SSP4": "Inequality",
    "SSP5": "Fossil-fueled Development",
}

SCENARIO_SHORT = {
    "SSP1": "Sustainability-oriented pathway",
    "SSP2": "Baseline / middle-of-the-road pathway",
    "SSP3": "Fragmented world with regional rivalry",
    "SSP4": "High inequality across and within countries",
    "SSP5": "Rapid growth driven by fossil-fueled development",
}

SCENARIO_DESCRIPTIONS = {
    "SSP1": (
        "SSP1 — Sustainability: The world shifts toward inclusive development, stronger institutions, "
        "better education, and environmentally conscious growth. Poverty headcount ratios tend to evolve more "
        "favorably because vulnerability is reduced through cooperation, investment in human capital, "
        "and lower structural inequality."
    ),
    "SSP2": (
        "SSP2 — Middle of the Road: Development follows historical patterns without major breakthroughs "
        "or major collapse. Poverty headcount ratios improve in some places but persist in others, making this a "
        "useful baseline scenario for comparing more optimistic and more adverse futures."
    ),
    "SSP3": (
        "SSP3 — Regional Rivalry: Countries become more fragmented, geopolitical tensions increase, and "
        "cooperation weakens. Lower investment, slower economic development, and institutional strain can "
        "lead to higher vulnerability, weaker resilience, and higher poverty headcount ratios."
    ),
    "SSP4": (
        "SSP4 — Inequality: Development is highly uneven, with globally connected and well-resourced groups "
        "benefiting while vulnerable populations lag behind. Poverty headcount ratios can remain elevated because inequality "
        "limits adaptive capacity and concentrates fragility in already disadvantaged regions."
    ),
    "SSP5": (
        "SSP5 — Fossil-fueled Development: Rapid economic growth and technological progress continue, but "
        "development remains resource- and energy-intensive. Some countries may reduce poverty through growth, "
        "yet long-term vulnerabilities can remain substantial under unsustainable pathways."
    ),
}

MODEL_FEATURES = [
    "gdp_per_capita",
    "hdi",
    "control_of_corruption",
    "employment_agriculture",
    "gini",
]

DASHBOARD_TO_MODEL_TARGET = {
    "poverty_3usd": "poverty_3",
    "poverty_8_3usd": "poverty_8_30",
    "poverty_10usd": "poverty_10",
}

FEATURE_LABELS = {
    "gdp_per_capita": "GDP per capita [USD/capita]",
    "population": "Population [people]",
    "hdi": "HDI [index]",
    "control_of_corruption": "Control of Corruption [index]",
    "employment_agriculture": "Employment in agriculture [% of total employment]",
    "gini": "Gini index [0–100]",
}


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
      .block-container {
        padding-top: 3.0rem;
        padding-bottom: 2rem;
      }

      section[data-testid="stSidebar"] {
        padding-top: 1.1rem;
      }

      div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(40,40,40,0.10);
        padding: 14px 16px;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(15,23,42,0.06);
      }

      .header-title {
        font-size: 2.05rem;
        font-weight: 750;
        line-height: 1.2;
        margin-bottom: 0.35rem;
      }

      .header-subtitle {
        font-size: 1rem;
        color: rgba(31,41,55,0.72);
        margin-bottom: 0.7rem;
      }

      .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.25rem;
        margin-bottom: 0.4rem;
      }

      .pill {
        display: inline-block;
        padding: 0.36rem 0.72rem;
        border-radius: 999px;
        font-size: 0.84rem;
        border: 1px solid rgba(40,40,40,0.10);
        background: rgba(255,255,255,0.95);
      }

      .section-note {
        padding: 0.8rem 1rem;
        border-radius: 14px;
        background: rgba(248,250,252,0.95);
        border: 1px solid rgba(40,40,40,0.08);
        margin-bottom: 0.75rem;
      }

      .footer-box {
        margin-top: 1.25rem;
        padding: 0.8rem 1rem;
        border-radius: 14px;
        background: rgba(248,250,252,0.95);
        border: 1px solid rgba(40,40,40,0.08);
        font-size: 0.92rem;
        color: rgba(31,41,55,0.72);
      }

      .tiny-note {
        font-size: 0.78rem;
        color: rgba(31,41,55,0.62);
        line-height: 1.35;
        margin-top: -0.1rem;
        margin-bottom: 0.2rem;
      }

      .tiny-alert {
        font-size: 0.79rem;
        color: #9a6700;
        background: rgba(255, 244, 214, 0.9);
        border: 1px solid rgba(154, 103, 0, 0.18);
        border-radius: 10px;
        padding: 0.35rem 0.55rem;
        margin-top: 0.15rem;
        margin-bottom: 0.25rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)

    # Long-format model output with real predictors
    if {"country_name", "country_code", "poverty_threshold", "predicted_poverty"}.issubset(df.columns):
        df = df.rename(
            columns={
                "country_name": "country",
                "country_code": "iso3",
            }
        )

        df["year"] = df["year"].astype(int)
        df["scenario"] = df["scenario"].astype(str)
        if "approach" in df.columns:
            df["approach"] = df["approach"].astype(str).str.strip().str.upper()

        threshold_map = {
            "poverty_3": "poverty_3usd",
            "poverty_8_30": "poverty_8_3usd",
            "poverty_10": "poverty_10usd",
        }
        df["poverty_threshold"] = df["poverty_threshold"].replace(threshold_map)

        feature_cols = [
            "gdp_per_capita",
            "population",
            "hdi",
            "control_of_corruption",
            "employment_agriculture",
            "gini",
        ]
        existing_feature_cols = [c for c in feature_cols if c in df.columns]

        index_cols = ["iso3", "country", "scenario", "year", "approach"] + existing_feature_cols

        df = df.pivot_table(
            index=index_cols,
            columns="poverty_threshold",
            values="predicted_poverty",
            aggfunc="first",
        ).reset_index()

        df.columns.name = None

        if "approach" in df.columns:
            df = df[df["approach"] == "B"].copy()

    else:
        df["year"] = df["year"].astype(int)
        df["scenario"] = df["scenario"].astype(str)
        if "approach" in df.columns:
            df["approach"] = df["approach"].astype(str).str.strip().str.upper()
            df = df[df["approach"] == "B"].copy()

    # Clean numeric columns defensively
    numeric_candidate_cols = [
        "gdp_per_capita",
        "population",
        "hdi",
        "control_of_corruption",
        "employment_agriculture",
        "gini",
        "poverty_3usd",
        "poverty_8_3usd",
        "poverty_10usd",
    ]

    for col in numeric_candidate_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(r"^\[", "", regex=True)
                .str.replace(r"\]$", "", regex=True)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data
def add_continent(df_in: pd.DataFrame) -> pd.DataFrame:
    gm = px.data.gapminder()[["iso_alpha", "continent"]].drop_duplicates()
    out = df_in.merge(gm, left_on="iso3", right_on="iso_alpha", how="left")
    out["continent"] = out["continent"].fillna("Other")
    out = out.drop(columns=["iso_alpha"])
    return out


# -----------------------------
# SHAP / Model helpers
# -----------------------------
@st.cache_resource
def load_xgb_model(target: str):
    model_path = MODELS_DIR / f"XGBoost_{target}.pkl"
    return joblib.load(model_path)


def get_model_input(df_subset: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in MODEL_FEATURES if c not in df_subset.columns]
    if missing:
        raise ValueError(f"Missing model features in dashboard data: {missing}")

    X = df_subset[MODEL_FEATURES].copy()

    for col in X.columns:
        X[col] = (
            X[col]
            .astype(str)
            .str.extract(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', expand=False)
        )
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.astype("float64").dropna()
    return X


@st.cache_data(show_spinner=False)
def compute_shap_overview(df_subset: pd.DataFrame, poverty_metric: str, sample_size: int = 300):
    target = DASHBOARD_TO_MODEL_TARGET[poverty_metric]
    model = load_xgb_model(target)

    X = get_model_input(df_subset)
    if X.empty:
        return None

    if len(X) > sample_size:
        X = X.sample(sample_size, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X)

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.array(base_value).flatten()[0])

    shap_values = shap.Explanation(
        values=shap_values_raw,
        base_values=np.repeat(base_value, len(X)),
        data=X.values,
        feature_names=X.columns.tolist(),
    )

    importance = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)

    return {
        "X": X,
        "shap_values": shap_values,
        "importance": importance,
        "base_value": base_value,
    }


@st.cache_data(show_spinner=False)
def compute_single_country_shap(row_df: pd.DataFrame, poverty_metric: str):
    target = DASHBOARD_TO_MODEL_TARGET[poverty_metric]
    model = load_xgb_model(target)

    X = get_model_input(row_df)
    if X.empty:
        return None

    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X)

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.array(base_value).flatten()[0])

    explanation = shap.Explanation(
        values=shap_values_raw[0],
        base_values=base_value,
        data=X.iloc[0].values,
        feature_names=X.columns.tolist(),
    )

    prediction = float(model.predict(X)[0])

    contrib_df = pd.DataFrame({
        "feature": X.columns,
        "feature_value": X.iloc[0].values,
        "shap_value": explanation.values,
    })

    contrib_df["abs_shap"] = contrib_df["shap_value"].abs()
    contrib_df = contrib_df.sort_values("abs_shap", ascending=False).drop(columns="abs_shap")

    return {
        "prediction": prediction,
        "explanation": explanation,
        "contrib_df": contrib_df,
    }


def render_shap_summary_plot(shap_values, max_display=10):
    plt.figure(figsize=(10, 5.8))
    shap.summary_plot(
        shap_values.values,
        features=shap_values.data,
        feature_names=shap_values.feature_names,
        show=False,
        max_display=max_display
    )
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf


def render_shap_waterfall_plot(explanation, max_display=10):
    plt.figure()
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 5.5)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf


# -----------------------------
# Load data
# -----------------------------
df = load_data(DEFAULT_PATH)
df = add_continent(df)

ID_COLS = [c for c in ["iso3", "country", "scenario", "year", "continent"] if c in df.columns]
POVERTY_COLS = [c for c in df.columns if c.startswith("poverty_")]
FEATURE_COLS = [c for c in df.columns if c not in ID_COLS + POVERTY_COLS + ["approach"]]


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.title("⚙️ Controls")

    scenario_options = sorted(df["scenario"].dropna().unique().tolist())
    default_scenario = "SSP1" if "SSP1" in scenario_options else scenario_options[0]

    scenario = st.segmented_control(
        "Scenario",
        options=scenario_options,
        default=default_scenario,
        key="scenario_selector",
        help=(
            "Shared Socioeconomic Pathways (SSPs) describe alternative global futures with different "
            "patterns of development, inequality, cooperation, and sustainability."
        ),
    )

    for s in scenario_options:
        st.caption(f"{s} — {SCENARIO_SHORT.get(s, '')}")

    available_years = sorted(df["year"].dropna().unique().tolist())

    year = st.select_slider(
        "Year",
        options=available_years,
        value=available_years[0],
        help="Select one of the available projection years in the dataset.",
    )

    st.markdown(
        """
        <div class="tiny-note">
        Disclaimer: For years above <b>2050</b>, some predictor variables are extrapolated.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if year > 2050:
        st.markdown(
            """
            <div class="tiny-alert">
            Selected year is above 2050 — some input variables are extrapolated.
            </div>
            """,
            unsafe_allow_html=True,
        )

    poverty_metric = st.selectbox(
        "Poverty line",
        options=POVERTY_COLS,
        index=0,
        format_func=lambda x: POVERTY_LABELS.get(x, x),
        help="Select the poverty line for which the predicted poverty headcount ratio is shown across the dashboard.",
    )

    map_scale = st.selectbox(
        "Map color scale",
        options=["Continuous", "Categorical risk bands"],
        index=0,
    )

    map_theme = st.segmented_control(
        "Map theme",
        options=["Light", "Dark"],
        default="Light",
        help="Switch only the map between a light or dark visual style.",
    )

    continents = ["Global"] + sorted(df["continent"].dropna().unique().tolist())
    region_focus = st.selectbox("Region focus", options=continents, index=0)

    show_top_n = st.slider("Show top N countries in rankings", 5, 30, 10, 1)

    highlight_top_map = st.toggle(
        "Highlight top countries on map",
        value=True,
        help="Emphasize the highest displayed values in the current view with a stronger border outline.",
    )

    st.divider()
    st.caption(
        "The quick data peek uses the predictor variables underlying the displayed model outputs."
    )


# -----------------------------
# Helpers
# -----------------------------
def pop_weighted_mean(d: pd.DataFrame, value_col: str, weight_col: str = "population") -> float | None:
    if weight_col not in d.columns:
        return None
    x = d[[value_col, weight_col]].dropna()
    if x.empty:
        return None
    w = x[weight_col].astype(float).values
    v = x[value_col].astype(float).values
    if np.sum(w) == 0:
        return float(np.nanmean(v))
    return float(np.sum(v * w) / np.sum(w))


selected_metric_label = POVERTY_LABELS.get(poverty_metric, poverty_metric)
selected_scenario_label = f"{scenario} – {SCENARIO_LABELS.get(scenario, scenario)}"


# -----------------------------
# Filtered data
# -----------------------------
base = df[(df["scenario"] == scenario) & (df["year"] == year)].copy()
if region_focus != "Global":
    base = base[base["continent"] == region_focus].copy()

base_metric = base.dropna(subset=[poverty_metric]).copy()


# -----------------------------
# Header section
# -----------------------------
st.markdown(
    f"""
    <div class="header-title">🌍 Global Governance Risk Explorer</div>
    <div class="header-subtitle">
      Policy-oriented exploration of poverty headcount ratios (PHR) under alternative SSP futures.
    </div>
    <div class="badge-row">
      <span class="pill">Scenario: {selected_scenario_label}</span>
      <span class="pill">Year: {year}</span>
      <span class="pill">Metric: {selected_metric_label}</span>
      <span class="pill">Region: {region_focus}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("About this dashboard and how to interpret it", expanded=False):
    st.markdown(
        f"""
This dashboard combines SSP-based scenario projections with a selected poverty headcount ratio indicator to support policy-oriented exploration across countries and regions.

**How to read the views**
- The **Risk Map** highlights the geographic distribution of the predicted poverty headcount ratio for the chosen scenario and year.
- **Scenario Comparison** shows how the selected poverty headcount ratio differs across SSP futures.
- **Regional Trends** aggregates country-level outcomes into continent-level trajectories.
- **Explainability** shows which model input variables are most influential for the selected view and how they affect a single-country prediction.

**What is shown**
- The predicted outcome is the **Poverty Headcount Ratio** at the selected poverty line, measured as **% of population**.

**Detailed SSP descriptions**
- **SSP1 — Sustainability:** {SCENARIO_DESCRIPTIONS["SSP1"].replace("SSP1 — Sustainability: ", "")}
- **SSP2 — Middle of the Road:** {SCENARIO_DESCRIPTIONS["SSP2"].replace("SSP2 — Middle of the Road: ", "")}
- **SSP3 — Regional Rivalry:** {SCENARIO_DESCRIPTIONS["SSP3"].replace("SSP3 — Regional Rivalry: ", "")}
- **SSP4 — Inequality:** {SCENARIO_DESCRIPTIONS["SSP4"].replace("SSP4 — Inequality: ", "")}
- **SSP5 — Fossil-fueled Development:** {SCENARIO_DESCRIPTIONS["SSP5"].replace("SSP5 — Fossil-fueled Development: ", "")}

**Model inputs included in the dataset**
- GDP per capita
- Population
- HDI
- Control of corruption
- Employment in agriculture
- Gini inequality index

**Extrapolated values**
- For projection years **above 2050**, some input variables are **extrapolated** and should be interpreted with that limitation in mind.

**Explainability note**
- The Explainability tab uses **SHAP values** from the final **XGBoost** models for the selected poverty line.
- Only the variables that were actually used to train the model are included in the SHAP analysis.
        """
    )


# -----------------------------
# KPI section
# -----------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

mean_val = float(base_metric[poverty_metric].mean()) if not base_metric.empty else np.nan
median_val = float(base_metric[poverty_metric].median()) if not base_metric.empty else np.nan
pw_mean = pop_weighted_mean(base_metric, poverty_metric)

high_risk_threshold = 25.0
high_risk_share = (
    float((base_metric[poverty_metric] > high_risk_threshold).mean() * 100.0)
    if not base_metric.empty else np.nan
)

kpi1.metric("Average headcount ratio [%]", f"{mean_val:,.2f}" if np.isfinite(mean_val) else "—")
kpi2.metric("Median headcount ratio [%]", f"{median_val:,.2f}" if np.isfinite(median_val) else "—")
kpi3.metric("Population-weighted average [%]", f"{pw_mean:,.2f}" if (pw_mean is not None and np.isfinite(pw_mean)) else "—")
kpi4.metric(
    f"Share above {high_risk_threshold:.0f}% headcount [% of countries]",
    f"{high_risk_share:,.1f}" if np.isfinite(high_risk_share) else "—"
)


# -----------------------------
# Download current view
# -----------------------------
download_df = base_metric.copy()
if not download_df.empty:
    csv_bytes = download_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download current filtered view as CSV",
        data=csv_bytes,
        file_name=f"ssp_poverty_headcount_view_{scenario}_{year}.csv",
        mime="text/csv",
    )


# -----------------------------
# Map theme config
# -----------------------------
if map_theme == "Dark":
    land_color = "rgb(33, 39, 53)"
    ocean_color = "rgb(18, 24, 38)"
    lake_color = "rgb(18, 24, 38)"
    country_line = "rgba(255,255,255,0.22)"
    coast_line = "rgba(255,255,255,0.35)"
    border_line = "rgba(255,255,255,0.18)"
    top_outline_color = "rgba(255,255,255,0.95)"
    paper_bg = "rgba(0,0,0,0)"
    map_continuous_scale = [
        [0.0, "#0B3C5D"],
        [0.2, "#328CC1"],
        [0.4, "#D9B310"],
        [0.7, "#F18F01"],
        [1.0, "#C73E1D"],
    ]
    map_template = "plotly_dark"
else:
    land_color = "rgb(245, 247, 250)"
    ocean_color = "rgb(232, 240, 248)"
    lake_color = "rgb(232, 240, 248)"
    country_line = "rgba(80,90,110,0.28)"
    coast_line = "rgba(80,90,110,0.35)"
    border_line = "rgba(80,90,110,0.22)"
    top_outline_color = "rgba(25,25,25,0.95)"
    paper_bg = "rgba(255,255,255,0)"
    map_continuous_scale = [
        [0.0, "#DCEAF7"],
        [0.2, "#9CC9E3"],
        [0.4, "#F4D06F"],
        [0.7, "#F29E4C"],
        [1.0, "#C8553D"],
    ]
    map_template = "plotly_white"


# -----------------------------
# Tabs
# -----------------------------
tab_map, tab_compare, tab_trends, tab_shap = st.tabs(
    ["🗺️ Risk Map", "📊 Scenario Comparison", "📈 Regional Trends", "🧠 Explainability"]
)


# -----------------------------
# MAP TAB
# -----------------------------
with tab_map:
    st.markdown(
        """
        <div class="section-note">
        <b>Map view:</b> Explore the selected SSP scenario spatially. The map shows the predicted poverty headcount ratio at the selected poverty line, measured as % of population.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([2.35, 1])

    with left:
        if base_metric.empty:
            st.warning("No data available for the selected filters.")
        else:
            map_df = base_metric.copy()

            if map_scale == "Categorical risk bands":
                bins = [-np.inf, 5, 15, 30, 50, np.inf]
                labels = ["Very Low", "Low", "Moderate", "High", "Extreme"]

                map_df["headcount_band"] = pd.cut(map_df[poverty_metric], bins=bins, labels=labels)

                fig = px.choropleth(
                    map_df,
                    locations="iso3",
                    color="headcount_band",
                    hover_name="country",
                    hover_data={
                        poverty_metric: ":.2f",
                        "continent": True,
                        "iso3": True,
                    },
                    category_orders={"headcount_band": labels},
                    title=f"{selected_metric_label} — {selected_scenario_label} — {year}",
                    projection="natural earth",
                    color_discrete_sequence=px.colors.sequential.Plasma_r[:len(labels)],
                )
            else:
                fig = px.choropleth(
                    map_df,
                    locations="iso3",
                    color=poverty_metric,
                    hover_name="country",
                    hover_data={
                        poverty_metric: ":.2f",
                        "continent": True,
                        "iso3": True,
                    },
                    title=f"{selected_metric_label} — {selected_scenario_label} — {year}",
                    color_continuous_scale=map_continuous_scale,
                    projection="natural earth",
                )

            fig.update_traces(
                marker_line_color=border_line,
                marker_line_width=0.6,
                hovertemplate=(
                    "<b>%{hovertext}</b><br>"
                    "ISO3: %{location}<br>"
                    f"{selected_metric_label}: %{{z:.2f}} % of population<extra></extra>"
                )
            )

            if highlight_top_map and not map_df.empty:
                top_outline = (
                    map_df.sort_values(poverty_metric, ascending=False)
                    .head(min(show_top_n, 10))
                    .copy()
                )

                if not top_outline.empty:
                    fig.add_trace(
                        go.Choropleth(
                            locations=top_outline["iso3"],
                            z=[1] * len(top_outline),
                            locationmode="ISO-3",
                            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                            showscale=False,
                            marker_line_color=top_outline_color,
                            marker_line_width=2.4,
                            hoverinfo="skip",
                            name="Top countries",
                        )
                    )

            if region_focus != "Global":
                fig.update_geos(
                    fitbounds="locations",
                    visible=False,
                    showcountries=True,
                    countrycolor=country_line,
                    showcoastlines=True,
                    coastlinecolor=coast_line,
                    showland=True,
                    landcolor=land_color,
                    showocean=True,
                    oceancolor=ocean_color,
                    showlakes=True,
                    lakecolor=lake_color,
                    showframe=False,
                    bgcolor="rgba(0,0,0,0)",
                )
            else:
                fig.update_geos(
                    showcoastlines=True,
                    coastlinecolor=coast_line,
                    showcountries=True,
                    countrycolor=country_line,
                    showland=True,
                    landcolor=land_color,
                    showocean=True,
                    oceancolor=ocean_color,
                    showlakes=True,
                    lakecolor=lake_color,
                    showframe=False,
                    bgcolor="rgba(0,0,0,0)",
                )

            fig.update_layout(
                template=map_template,
                height=690,
                margin=dict(l=0, r=0, t=68, b=0),
                title=dict(x=0.02, font=dict(size=22)),
                paper_bgcolor=paper_bg,
                plot_bgcolor=paper_bg,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=0.01,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(0,0,0,0.08)",
                    borderwidth=0,
                ),
            )

            if map_scale == "Continuous":
                fig.update_layout(
                    coloraxis_colorbar=dict(
                        title=selected_metric_label,
                        thickness=16,
                        len=0.72,
                        bgcolor="rgba(0,0,0,0.0)",
                    )
                )

            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("📌 Policy Focus View")
        if base_metric.empty:
            st.info("No ranking available for the current filters.")
        else:
            top = (
                base_metric
                .sort_values(poverty_metric, ascending=False)
                .head(show_top_n)[["country", "iso3", poverty_metric]]
            )
            top = top.rename(columns={poverty_metric: selected_metric_label})
            st.dataframe(top, use_container_width=True, hide_index=True)

        st.subheader("🔎 Quick data peek")

        quick_country_options = ["All countries"] + sorted(base_metric["country"].dropna().unique().tolist())
        quick_country = st.selectbox(
            "Filter quick peek by country",
            options=quick_country_options,
            index=0,
            key="quick_peek_country",
        )

        peek_df = base_metric.copy()
        if quick_country != "All countries":
            peek_df = peek_df[peek_df["country"] == quick_country].copy()

        preferred_feature_order = [
            "gdp_per_capita",
            "population",
            "hdi",
            "control_of_corruption",
            "employment_agriculture",
            "gini",
        ]
        preview_cols = ["country", "iso3", "continent", poverty_metric]
        preview_cols += [c for c in preferred_feature_order if c in peek_df.columns]
        existing_preview_cols = [c for c in preview_cols if c in peek_df.columns]

        preview_display = peek_df[existing_preview_cols].head(12).copy()
        preview_display = preview_display.rename(columns={
            poverty_metric: selected_metric_label,
            **FEATURE_LABELS,
        })

        st.dataframe(
            preview_display,
            use_container_width=True,
            hide_index=True,
        )


# -----------------------------
# COMPARISON TAB
# -----------------------------
with tab_compare:
    st.markdown(
        """
        <div class="section-note">
        <b>Scenario comparison:</b> Compare how the selected poverty headcount ratio differs across SSP futures in the chosen year.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Scenario comparison at selected year")

    comp = df[df["year"] == year].copy()
    if region_focus != "Global":
        comp = comp[comp["continent"] == region_focus].copy()
    comp = comp.dropna(subset=[poverty_metric])

    c1, c2 = st.columns([1.4, 1])

    with c1:
        fig = px.box(
            comp,
            x="scenario",
            y=poverty_metric,
            points="outliers",
            title=f"Distribution of {selected_metric_label} across scenarios ({year})",
        )
        fig.update_layout(
            template="plotly_white",
            height=460,
            margin=dict(l=0, r=0, t=60, b=0),
            title=dict(x=0.02),
            yaxis_title=selected_metric_label,
            xaxis_title="Scenario",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        agg_rows = []
        for s in scenario_options:
            d = comp[comp["scenario"] == s]
            agg_rows.append(
                {
                    "scenario": s,
                    "mean": float(d[poverty_metric].mean()) if not d.empty else np.nan,
                    "pop_weighted_mean": pop_weighted_mean(d, poverty_metric),
                }
            )
        agg = pd.DataFrame(agg_rows)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=agg["scenario"], y=agg["mean"], name="Mean"))
        if agg["pop_weighted_mean"].notna().any():
            fig.add_trace(go.Bar(x=agg["scenario"], y=agg["pop_weighted_mean"], name="Pop-weighted mean"))

        fig.update_layout(
            template="plotly_white",
            barmode="group",
            title=f"Scenario averages — {selected_metric_label} ({year})",
            height=460,
            margin=dict(l=0, r=0, t=60, b=0),
            yaxis_title=selected_metric_label,
            xaxis_title="Scenario",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Country spotlight")
    country_options = sorted(df["country"].dropna().unique().tolist())
    default_country = "Angola" if "Angola" in country_options else country_options[0]

    selected_country = st.selectbox(
        "Pick a country",
        options=country_options,
        index=country_options.index(default_country),
    )

    spotlight = df[df["country"] == selected_country].copy()
    if region_focus != "Global":
        spotlight = spotlight[spotlight["continent"] == region_focus].copy()

    fig = px.line(
        spotlight,
        x="year",
        y=poverty_metric,
        color="scenario",
        markers=True,
        title=f"{selected_country}: {selected_metric_label} over time by scenario",
    )
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=0, r=0, t=60, b=0),
        title=dict(x=0.02),
        yaxis_title=selected_metric_label,
        xaxis_title="Year",
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# TRENDS TAB
# -----------------------------
with tab_trends:
    st.markdown(
        """
        <div class="section-note">
        <b>Regional trends:</b> Track continent-level developments over time and compare how trajectories differ across SSP scenarios.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Regional trend lines (continent aggregates)")

    trend = df.copy()
    if region_focus != "Global":
        trend = trend[trend["continent"] == region_focus].copy()

    trend = trend.dropna(subset=[poverty_metric])

    if "population" in trend.columns:
        g = trend.dropna(subset=["population"]).copy()
        g["w"] = g["population"].astype(float)
        g["v"] = g[poverty_metric].astype(float)

        regional = (
            g.groupby(["continent", "scenario", "year"], as_index=False)
            .apply(lambda x: np.sum(x["v"] * x["w"]) / np.sum(x["w"]) if np.sum(x["w"]) else np.nan)
            .rename(columns={None: "headcount_pw_mean"})
        )
        ycol = "headcount_pw_mean"
        yname = "Population-weighted poverty headcount ratio [% of population]"
    else:
        regional = trend.groupby(["continent", "scenario", "year"], as_index=False)[poverty_metric].mean()
        ycol = poverty_metric
        yname = "Mean poverty headcount ratio [% of population]"

    c1, c2 = st.columns([1.15, 1])

    with c1:
        conts = sorted(regional["continent"].dropna().unique().tolist())
        default_cont = region_focus if region_focus != "Global" else ("Europe" if "Europe" in conts else conts[0])

        chosen_cont = st.selectbox("Spotlight continent", conts, index=conts.index(default_cont))

        sub = regional[regional["continent"] == chosen_cont].copy()
        fig = px.line(
            sub,
            x="year",
            y=ycol,
            color="scenario",
            markers=True,
            title=f"{chosen_cont}: {yname} over time",
        )
        fig.update_layout(
            template="plotly_white",
            height=460,
            margin=dict(l=0, r=0, t=60, b=0),
            title=dict(x=0.02),
            yaxis_title=yname,
            xaxis_title="Year",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        chosen_scenario = st.selectbox(
            "Scenario for multi-continent view",
            scenario_options,
            index=min(1, len(scenario_options) - 1),
        )

        sub = regional[regional["scenario"] == chosen_scenario].copy()
        fig = px.line(
            sub,
            x="year",
            y=ycol,
            color="continent",
            markers=True,
            title=f"{chosen_scenario}: {yname} by continent",
        )
        fig.update_layout(
            template="plotly_white",
            height=460,
            margin=dict(l=0, r=0, t=60, b=0),
            title=dict(x=0.02),
            yaxis_title=yname,
            xaxis_title="Year",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Continent mapping is derived from Plotly’s built-in Gapminder country mapping.")


# -----------------------------
# EXPLAINABILITY TAB
# -----------------------------
with tab_shap:
    st.markdown(
        """
        <div class="section-note">
        <b>Explainability:</b> This view uses SHAP values from the final XGBoost model to show which variables drive the predicted poverty headcount ratios.
        The overview reflects the currently selected dashboard subset. A single-country explanation can be generated on demand.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_top1, c_top2 = st.columns([1, 1])
    with c_top1:
        shap_sample_size = st.slider(
            "Sample size for overview SHAP",
            min_value=50,
            max_value=500,
            value=250,
            step=50,
            help="To keep the dashboard responsive, the SHAP overview uses a sample of the currently filtered observations."
        )
    with c_top2:
        show_beeswarm = st.toggle(
            "Show SHAP summary plot",
            value=True,
            help="The beeswarm plot gives a richer overview, but it is visually denser than the bar chart."
        )

    try:
        shap_result = compute_shap_overview(
            base_metric,
            poverty_metric=poverty_metric,
            sample_size=shap_sample_size,
        )
    except Exception as e:
        shap_result = None
        st.error(f"SHAP overview could not be computed: {e}")

    if shap_result is None:
        st.info("No SHAP explanation available for the current filter selection.")
    else:
        st.caption(
            f"Overview explanation for {selected_metric_label} under {selected_scenario_label}, year {year}, region {region_focus}."
        )
        st.caption(
            "Higher mean absolute SHAP values indicate variables that contribute more strongly to the model’s predictions within the currently selected subset."
        )

        c1, c2 = st.columns([1, 1.35])

        with c1:
            imp_display = shap_result["importance"].copy()
            imp_display["feature"] = imp_display["feature"].replace(FEATURE_LABELS)

            fig_imp = px.bar(
                imp_display,
                x="mean_abs_shap",
                y="feature",
                orientation="h",
                title="Variable importance (mean absolute SHAP value)",
            )
            fig_imp.update_layout(
                template="plotly_white",
                height=430,
                margin=dict(l=0, r=0, t=60, b=0),
                xaxis_title="Average contribution magnitude",
                yaxis_title="",
            )
            fig_imp.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig_imp, use_container_width=True)

        with c2:
            if show_beeswarm:
                buf = render_shap_summary_plot(shap_result["shap_values"], max_display=10)
                st.image(buf, use_container_width=True)

    st.divider()

    st.subheader("Country-level explanation")
    st.caption("Explains one country for the currently selected scenario, year, region, and poverty line.")
    st.caption("Positive SHAP values increase the predicted poverty headcount ratio, while negative SHAP values decrease it relative to the model baseline.")

    country_options = sorted(base_metric["country"].dropna().unique().tolist())

    if not country_options:
        st.info("No country available for the current filter selection.")
    else:
        selected_country_shap = st.selectbox(
            "Select country",
            options=country_options,
            key="shap_country_selector",
        )

        country_row = base_metric[base_metric["country"] == selected_country_shap].head(1).copy()

        if st.button("Explain selected country"):
            try:
                single_result = compute_single_country_shap(country_row, poverty_metric)
            except Exception as e:
                single_result = None
                st.error(f"Country-level SHAP could not be computed: {e}")

            if single_result is None:
                st.warning("No valid feature data available for this country.")
            else:
                st.metric(
                    "Predicted poverty headcount ratio [% of population]",
                    f"{single_result['prediction']:.2f}"
                )

                wc1, wc2 = st.columns([1.35, 1])

                with wc1:
                    buf = render_shap_waterfall_plot(single_result["explanation"], max_display=10)
                    st.image(buf, use_container_width=True)

                with wc2:
                    contrib_display = single_result["contrib_df"].copy()
                    contrib_display["feature"] = contrib_display["feature"].replace(FEATURE_LABELS)
                    contrib_display["feature_value"] = contrib_display["feature_value"].round(3)
                    contrib_display["shap_value"] = contrib_display["shap_value"].round(3)
                    contrib_display = contrib_display.rename(columns={
                        "feature": "Feature",
                        "feature_value": "Feature value",
                        "shap_value": "SHAP contribution",
                    })

                    st.dataframe(
                        contrib_display,
                        use_container_width=True,
                        hide_index=True,
                    )


# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <div class="footer-box">
    <b>Project note:</b> This dashboard is designed as an interactive policy exploration tool for SSP-based poverty headcount ratio analysis.
    For years above 2050, some inputs are extrapolated and should be interpreted accordingly.
    </div>
    """,
    unsafe_allow_html=True,
)