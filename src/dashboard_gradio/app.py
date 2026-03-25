import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import gradio as gr
import plotly.express as px
import plotly.graph_objects as go

from functools import lru_cache
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# Optional SHAP (graceful fallback)
# -----------------------------
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_CSV = "C:\\Users\\miche\\Desktop\\SSP Risk Explorer\\project-ssp-risk-explorer\\data\\raw\\dummy\\dashboard_ready_dummy_ssp_poverty_multiline.csv"
SCENARIOS = ["SSP1", "SSP2", "SSP3", "SSP5"]
PREFERRED_YEARS = [2030, 2050, 2100]


# -----------------------------
# Data loading + enrichment
# -----------------------------
@lru_cache(maxsize=4)
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["year"] = df["year"].astype(int)
    df["scenario"] = df["scenario"].astype(str)

    # Add continent mapping using plotly's built-in gapminder (offline)
    gm = px.data.gapminder()[["iso_alpha", "continent"]].drop_duplicates()
    df = df.merge(gm, left_on="iso3", right_on="iso_alpha", how="left")
    df["continent"] = df["continent"].fillna("Other")
    df = df.drop(columns=["iso_alpha"])
    return df


def get_columns(df: pd.DataFrame):
    id_cols = ["iso3", "country", "scenario", "year"]
    poverty_cols = [c for c in df.columns if c.startswith("poverty_")]
    feature_cols = [c for c in df.columns if c not in id_cols + poverty_cols + ["continent"]]
    return id_cols, poverty_cols, feature_cols


def pop_weighted_mean(d: pd.DataFrame, value_col: str, weight_col: str = "population"):
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


def year_options(df: pd.DataFrame):
    years = sorted(df["year"].unique().tolist())
    preferred = [y for y in PREFERRED_YEARS if y in years]
    return preferred if preferred else years


# -----------------------------
# Figure builders (FAST)
# -----------------------------
def build_map(df: pd.DataFrame, scenario: str, year: int, metric: str, region: str, map_scale: str):
    d = df[(df["scenario"] == scenario) & (df["year"] == year)].copy()
    if region != "Global":
        d = d[d["continent"] == region].copy()
    d = d.dropna(subset=[metric])

    if d.empty:
        return go.Figure().update_layout(
            title=dict(text="No data for selected filters", x=0.02),
            height=620
        )

    if map_scale == "Categorical risk bands":
        bins = [-np.inf, 5, 15, 30, 50, np.inf]
        labels = ["Very Low", "Low", "Moderate", "High", "Extreme"]
        d["risk_band"] = pd.cut(d[metric], bins=bins, labels=labels)

        fig = px.choropleth(
            d,
            locations="iso3",
            color="risk_band",
            hover_name="country",
            category_orders={"risk_band": labels},
        )
        fig.update_layout(title=dict(text=f"Risk bands — {scenario} {year} ({metric})", x=0.02))
    else:
        fig = px.choropleth(
            d,
            locations="iso3",
            color=metric,
            hover_name="country",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(title=dict(text=f"{metric} — {scenario} {year}", x=0.02))

    fig.update_layout(height=620, margin=dict(l=0, r=0, t=60, b=0))
    return fig


def build_ranking_table(df: pd.DataFrame, scenario: str, year: int, metric: str, region: str, top_n: int):
    d = df[(df["scenario"] == scenario) & (df["year"] == year)].copy()
    if region != "Global":
        d = d[d["continent"] == region].copy()
    d = d.dropna(subset=[metric])
    if d.empty:
        return pd.DataFrame(columns=["country", "iso3", metric])
    return d.sort_values(metric, ascending=False).head(top_n)[["country", "iso3", metric]]


def build_kpis(df: pd.DataFrame, scenario: str, year: int, metric: str, region: str):
    d = df[(df["scenario"] == scenario) & (df["year"] == year)].copy()
    if region != "Global":
        d = d[d["continent"] == region].copy()
    d = d.dropna(subset=[metric])

    if d.empty:
        return "—", "—", "—", "—"

    mean_val = float(d[metric].mean())
    median_val = float(d[metric].median())
    pw = pop_weighted_mean(d, metric)
    high_thr = 25.0
    share_high = float((d[metric] > high_thr).mean() * 100.0)

    return (
        f"{mean_val:,.2f}%",
        f"{median_val:,.2f}%",
        f"{pw:,.2f}%" if (pw is not None and np.isfinite(pw)) else "—",
        f"{share_high:,.1f}% (> {high_thr:.0f}%)"
    )


def build_scenario_boxplot(df: pd.DataFrame, year: int, metric: str, region: str):
    d = df[df["year"] == year].copy()
    if region != "Global":
        d = d[d["continent"] == region].copy()
    d = d.dropna(subset=[metric])
    if d.empty:
        return go.Figure().update_layout(title=dict(text="No data", x=0.02), height=460)

    fig = px.box(d, x="scenario", y=metric, points="outliers")
    fig.update_layout(
        title=dict(text=f"Distribution of {metric} across scenarios ({year})", x=0.02),
        height=460, margin=dict(l=0, r=0, t=60, b=0)
    )
    return fig


def build_scenario_averages(df: pd.DataFrame, year: int, metric: str, region: str):
    d = df[df["year"] == year].copy()
    if region != "Global":
        d = d[d["continent"] == region].copy()
    d = d.dropna(subset=[metric])

    rows = []
    for s in SCENARIOS:
        ds = d[d["scenario"] == s]
        rows.append({
            "scenario": s,
            "mean": float(ds[metric].mean()) if not ds.empty else np.nan,
            "pw_mean": pop_weighted_mean(ds, metric)
        })
    agg = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["scenario"], y=agg["mean"], name="Mean"))
    if agg["pw_mean"].notna().any():
        fig.add_trace(go.Bar(x=agg["scenario"], y=agg["pw_mean"], name="Pop-weighted mean"))

    fig.update_layout(
        barmode="group",
        title=dict(text=f"Scenario averages — {metric} ({year})", x=0.02),
        height=460, margin=dict(l=0, r=0, t=60, b=0)
    )
    return fig


def build_country_spotlight(df: pd.DataFrame, country: str, metric: str, region: str):
    d = df[df["country"] == country].copy()
    if region != "Global":
        d = d[d["continent"] == region].copy()
    d = d.dropna(subset=[metric])
    if d.empty:
        return go.Figure().update_layout(title=dict(text="No data", x=0.02), height=420)

    fig = px.line(d, x="year", y=metric, color="scenario", markers=True)
    fig.update_layout(
        title=dict(text=f"{country}: {metric} over time by scenario", x=0.02),
        height=420, margin=dict(l=0, r=0, t=60, b=0)
    )
    return fig


def build_regional_trends(df: pd.DataFrame, metric: str, region_focus: str, spotlight_continent: str, scenario_for_multicont: str):
    d = df.copy()
    if region_focus != "Global":
        d = d[d["continent"] == region_focus].copy()
    d = d.dropna(subset=[metric])

    if d.empty:
        empty = go.Figure().update_layout(title=dict(text="No data", x=0.02), height=460)
        return empty, empty

    if "population" in d.columns:
        g = d.dropna(subset=["population"]).copy()
        g["w"] = g["population"].astype(float)
        g["v"] = g[metric].astype(float)
        regional = (
            g.groupby(["continent", "scenario", "year"], as_index=False)
            .apply(lambda x: np.sum(x["v"] * x["w"]) / np.sum(x["w"]) if np.sum(x["w"]) else np.nan)
            .rename(columns={None: "risk_pw_mean"})
        )
        ycol = "risk_pw_mean"
        ylabel = "Pop-weighted mean risk (%)"
    else:
        regional = d.groupby(["continent", "scenario", "year"], as_index=False)[metric].mean()
        ycol = metric
        ylabel = "Mean risk (%)"

    sub1 = regional[regional["continent"] == spotlight_continent].copy()
    fig1 = px.line(sub1, x="year", y=ycol, color="scenario", markers=True)
    fig1.update_layout(
        title=dict(text=f"{spotlight_continent}: {ylabel} over time", x=0.02),
        height=460, margin=dict(l=0, r=0, t=60, b=0)
    )

    sub2 = regional[regional["scenario"] == scenario_for_multicont].copy()
    fig2 = px.line(sub2, x="year", y=ycol, color="continent", markers=True)
    fig2.update_layout(
        title=dict(text=f"{scenario_for_multicont}: {ylabel} by continent", x=0.02),
        height=460, margin=dict(l=0, r=0, t=60, b=0)
    )

    return fig1, fig2


# -----------------------------
# SHAP (SLOW, button-triggered)
# -----------------------------
def build_explainability(df: pd.DataFrame, metric: str, region_scope: str):
    if not SHAP_AVAILABLE:
        return (
            "SHAP is unavailable in this environment (often due to NumPy/Numba version mismatch). "
            "Install compatible versions (e.g., numpy<=2.3) to enable SHAP.",
            None,
            None
        )

    try:
        d = df.copy()
        if region_scope != "Global":
            d = d[d["continent"] == region_scope].copy()

        _, poverty_cols, feature_cols = get_columns(d)
        if metric not in poverty_cols:
            return "Selected metric not found.", None, None

        d = d.dropna(subset=[metric]).copy()
        if len(d) < 200:
            return "Not enough rows for a stable SHAP demo. Try Global scope.", None, None

        X = d[feature_cols].replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True))
        y = d[metric].astype(float).values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=7)

        model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=7,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        sample_n = min(600, len(X_test))
        X_shap = X_test.sample(sample_n, random_state=7)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)

        import matplotlib.pyplot as plt

        def fig_to_rgb_array(fig):
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            return rgba[..., :3].copy()

        fig1 = plt.figure(figsize=(8.5, 5.2))
        shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=min(12, len(feature_cols)))
        bar_img = fig_to_rgb_array(fig1)
        plt.close(fig1)

        fig2 = plt.figure(figsize=(8.5, 5.2))
        shap.summary_plot(shap_values, X_shap, show=False, max_display=min(12, len(feature_cols)))
        swarm_img = fig_to_rgb_array(fig2)
        plt.close(fig2)

        y_pred = model.predict(X_test)
        mae = float(np.mean(np.abs(y_test - y_pred)))

        msg = f"✅ SHAP computed. Quick model check: MAE = {mae:,.3f} (sample={sample_n})"
        return msg, bar_img, swarm_img

    except Exception as e:
        return f"❌ SHAP failed: {type(e).__name__}: {e}", None, None


def compute_shap(csv_path, metric, xai_scope, region_focus):
    csv_path = (csv_path or DEFAULT_CSV).strip()
    if not os.path.exists(csv_path):
        return f"❌ CSV not found: {csv_path}", None, None

    df = load_data(csv_path)
    region_scope = "Global" if xai_scope == "Global" else region_focus
    return build_explainability(df, metric, region_scope)


# -----------------------------
# Main update function (FAST; no SHAP)
# -----------------------------
def update_dashboard(csv_path, scenario, year, metric, map_scale, region, top_n, spotlight_country, trend_spotlight_cont, trend_scenario):
    csv_path = (csv_path or DEFAULT_CSV).strip()
    if not os.path.exists(csv_path):
        return (
            f"❌ CSV not found: {csv_path}",
            "—", "—", "—", "—",
            go.Figure(), pd.DataFrame(),
            go.Figure(), go.Figure(), go.Figure(),
            go.Figure(), go.Figure()
        )

    df = load_data(csv_path)
    _, poverty_cols, _ = get_columns(df)
    if poverty_cols and metric not in poverty_cols:
        metric = poverty_cols[0]

    k1, k2, k3, k4 = build_kpis(df, scenario, year, metric, region)

    fig_map = build_map(df, scenario, year, metric, region, map_scale)
    ranking = build_ranking_table(df, scenario, year, metric, region, int(top_n))

    fig_box = build_scenario_boxplot(df, year, metric, region)
    fig_avg = build_scenario_averages(df, year, metric, region)
    fig_country = build_country_spotlight(df, spotlight_country, metric, region)

    fig_trend1, fig_trend2 = build_regional_trends(df, metric, region, trend_spotlight_cont, trend_scenario)

    status = f"✅ Loaded: {os.path.basename(csv_path)} | View: {scenario} · {year} · {metric} · Region: {region}"
    return status, k1, k2, k3, k4, fig_map, ranking, fig_box, fig_avg, fig_country, fig_trend1, fig_trend2


# -----------------------------
# Build UI (Controls right, content left)
# -----------------------------
def build_app():
    df0 = load_data(DEFAULT_CSV) if os.path.exists(DEFAULT_CSV) else None

    if df0 is not None:
        _, poverty_cols0, _ = get_columns(df0)
        years0 = year_options(df0)
        continents0 = ["Global"] + sorted(df0["continent"].unique().tolist())
        countries0 = sorted(df0["country"].unique().tolist())
        metric0 = poverty_cols0[0] if poverty_cols0 else "poverty_dummy"
        year0 = years0[0]
        cont0 = "Global"
        country0 = "Angola" if "Angola" in countries0 else countries0[0]
        trend_spotlight0 = "Europe" if "Europe" in df0["continent"].unique() else sorted(df0["continent"].unique().tolist())[0]
        all_conts = sorted(df0["continent"].unique().tolist())
    else:
        poverty_cols0 = ["poverty_dummy"]
        years0 = PREFERRED_YEARS
        continents0 = ["Global"]
        countries0 = ["—"]
        metric0 = poverty_cols0[0]
        year0 = years0[0]
        cont0 = "Global"
        country0 = "—"
        trend_spotlight0 = "Global"
        all_conts = ["Global"]

    # Small CSS to make the right panel feel like a compact control dock
    css = """
    .control-dock {
      border: 1px solid rgba(0,0,0,0.08);
      border-radius: 16px;
      padding: 14px 14px 8px 14px;
      background: rgba(255,255,255,0.55);
      backdrop-filter: blur(6px);
    }
    """

    with gr.Blocks(title="SSP Risk Explorer — Gradio", theme=gr.themes.Soft(), css=css) as demo:
        gr.Markdown(
            """
            # 🌍 SSP Risk Explorer — Poverty Multiline
            **Controls are docked on the right** so the content stays front and center.  
            SHAP runs **only when you click** the button.
            """
        )

        with gr.Row(equal_height=True):
            # ---------------- Left: Content ----------------
            with gr.Column(scale=5):
                status = gr.Markdown("")

                with gr.Row():
                    kpi1 = gr.Textbox(label="Avg risk (%)", value="—")
                    kpi2 = gr.Textbox(label="Median risk (%)", value="—")
                    kpi3 = gr.Textbox(label="Pop-weighted avg (%)", value="—")
                    kpi4 = gr.Textbox(label="Share high-risk", value="—")

                with gr.Tabs():
                    with gr.Tab("🗺️ Map"):
                        with gr.Row():
                            fig_map = gr.Plot(label="Map", scale=3)
                            ranking = gr.Dataframe(label="Country ranking", scale=2, interactive=False)

                    with gr.Tab("📊 Scenario Comparison"):
                        with gr.Row():
                            fig_box = gr.Plot(label="Distribution by scenario", scale=1)
                            fig_avg = gr.Plot(label="Scenario averages", scale=1)

                        gr.Markdown("### Country spotlight")
                        fig_country = gr.Plot(label="Country trend by scenario")

                    with gr.Tab("📈 Regional Trends"):
                        with gr.Row():
                            fig_trend1 = gr.Plot(label="Spotlight continent trend", scale=1)
                            fig_trend2 = gr.Plot(label="Multi-continent trend", scale=1)

                        gr.Markdown(
                            "Note: continent mapping uses Plotly’s built-in Gapminder dataset; unmatched countries are labeled 'Other'."
                        )

                    with gr.Tab("🧠 Explainability (SHAP)"):
                        xai_msg = gr.Markdown("Click **Compute SHAP** to generate explainability plots.")
                        with gr.Row():
                            bar_img = gr.Image(label="SHAP importance (bar)", type="numpy")
                            swarm_img = gr.Image(label="SHAP summary (beeswarm)", type="numpy")

                        if not SHAP_AVAILABLE:
                            gr.Markdown(
                                "⚠️ **SHAP is not available in your environment right now.** "
                                "To enable it, pin NumPy to ≤ 2.3 (and compatible numba/shap)."
                            )

            # ---------------- Right: Controls dock ----------------
            with gr.Column(scale=2, elem_classes=["control-dock"]):
                gr.Markdown("## ⚙️ Controls")

                csv_path = gr.Textbox(label="CSV path", value=DEFAULT_CSV)
                scenario = gr.Radio(SCENARIOS, value="SSP2", label="Scenario")
                year = gr.Radio(years0, value=year0, label="Year")

                metric = gr.Dropdown(poverty_cols0, value=metric0, label="Risk metric (poverty line)")
                map_scale = gr.Radio(["Continuous", "Categorical risk bands"], value="Continuous", label="Map scale")

                region = gr.Dropdown(continents0, value=cont0, label="Region focus")
                top_n = gr.Slider(5, 30, value=10, step=1, label="Top N countries")

                gr.Markdown("### Spotlight & Trends")
                spotlight_country = gr.Dropdown(countries0, value=country0, label="Country spotlight")

                trend_spotlight_cont = gr.Dropdown(
                    all_conts,
                    value=trend_spotlight0 if df0 is not None else all_conts[0],
                    label="Spotlight continent",
                )
                trend_scenario = gr.Dropdown(SCENARIOS, value="SSP2", label="Trend scenario (multi-continent)")

                gr.Markdown("### Explainability")
                xai_scope = gr.Dropdown(["Global", "Use selected region focus"], value="Global", label="Model scope")
                compute_btn = gr.Button("Compute SHAP (slow)", variant="primary")

        # FAST updates for everything except SHAP
        fast_inputs = [
            csv_path, scenario, year, metric, map_scale, region, top_n,
            spotlight_country, trend_spotlight_cont, trend_scenario
        ]
        fast_outputs = [
            status, kpi1, kpi2, kpi3, kpi4,
            fig_map, ranking,
            fig_box, fig_avg, fig_country,
            fig_trend1, fig_trend2
        ]

        for comp in fast_inputs:
            comp.change(fn=update_dashboard, inputs=fast_inputs, outputs=fast_outputs)

        demo.load(fn=update_dashboard, inputs=fast_inputs, outputs=fast_outputs)

        # SHAP only when button clicked
        compute_btn.click(
            fn=compute_shap,
            inputs=[csv_path, metric, xai_scope, region],
            outputs=[xai_msg, bar_img, swarm_img]
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
