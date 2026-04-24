"""
Global Health & Disease Analytics Dashboard
============================================
Team: Vecha Jatadhar | Pravalika Revelli (X25161261) | Marla Srija (X24295256)

Interactive Streamlit dashboard connecting to PostgreSQL for all data.
Run with:  streamlit run dashboard/app.py
"""

import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import PG_CONN_STR, N_CLUSTERS
from src.visualisations import (
    choropleth_life_expectancy,
    scatter_expenditure_vs_lifeexp,
    bar_feature_importance,
    scatter_actual_vs_predicted,
    scatter_clusters,
    heatmap_correlation,
    box_life_exp_by_cluster,
    bar_covid_by_continent,
    plot_regression_coefficients,
)

logging.basicConfig(level=logging.WARNING)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Global Health Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 20px; color: white;
        text-align: center; margin: 4px;
    }
    .metric-card h3 { font-size: 2rem; margin: 0; }
    .metric-card p  { margin: 0; opacity: 0.85; font-size: 0.9rem; }
    .section-header {
        font-size: 1.3rem; font-weight: 700;
        border-left: 5px solid #667eea; padding-left: 12px;
        margin: 20px 0 10px 0;
    }
    [data-testid="stSidebar"] { background: #1a1a2e; }
    [data-testid="stSidebar"] * { color: #e0e0e0; }
</style>
""", unsafe_allow_html=True)

# ─── DB connection ────────────────────────────────────────────────────────────
@st.cache_resource
def get_engine():
    return create_engine(PG_CONN_STR)

@st.cache_data(ttl=300)
def load_profiles() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM analysis_country_profiles", get_engine())

@st.cache_data(ttl=300)
def load_who_ts(indicator: str) -> pd.DataFrame:
    q = f"""
        SELECT w.country_code, d.country_name, w.year, w.value
        FROM fact_who_indicators w
        JOIN dim_country d USING (country_code)
        WHERE indicator_label = '{indicator}'
        ORDER BY country_code, year
    """
    return pd.read_sql(q, get_engine())

@st.cache_data(ttl=300)
def load_disease() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM fact_disease_stats", get_engine())

@st.cache_data(ttl=300)
def load_corr() -> pd.DataFrame:
    df = load_profiles()
    numeric = df.select_dtypes(include=np.number).drop(
        columns=["pca_x", "pca_y", "country_cluster", "population"], errors="ignore"
    )
    return numeric.corr()

# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 🏥 Global Health Analytics")
    st.markdown("**NCI MSc Data Analytics**  \nSemester 2 — 2025/26")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📊 Overview",
            "🗺️ World Map",
            "💰 Health Expenditure",
            "🤖 ML Models",
            "🌍 Clustering",
            "📈 Time Series",
            "🦠 COVID-19",
            "🔗 Correlations",
        ],
    )
    st.markdown("---")

    # Global filters
    df_all = load_profiles()
    continents = sorted(df_all["continent"].dropna().unique().tolist())
    sel_continents = st.multiselect("Filter by Continent", continents, default=continents)
    df = df_all[df_all["continent"].isin(sel_continents)] if sel_continents else df_all

    st.markdown("---")
    st.markdown(
        "**Team Members**\n"
        "- Vecha Jatadhar\n"
        "- Pravalika Revelli (X25161261)\n"
        "- Marla Srija (X24295256)"
    )

# ═════════════════════════════════════════════════════════════════════════════
#  PAGES
# ═════════════════════════════════════════════════════════════════════════════

# ── Overview ──────────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("🏥 Global Health & Disease Analytics")
    st.markdown(
        "An end-to-end analytics pipeline integrating **WHO GHO**, **World Bank**, "
        "and **disease.sh** APIs to uncover global health patterns and predict life expectancy."
    )
    st.markdown("---")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <h3>{df['country_code'].nunique()}</h3>
            <p>Countries Analysed</p></div>""", unsafe_allow_html=True)
    with c2:
        mean_le = df["life_expectancy_both"].mean()
        st.markdown(f"""<div class="metric-card">
            <h3>{mean_le:.1f}</h3>
            <p>Avg Life Expectancy (yrs)</p></div>""", unsafe_allow_html=True)
    with c3:
        mean_exp = df["health_expenditure_per_capita"].mean()
        st.markdown(f"""<div class="metric-card">
            <h3>${mean_exp:,.0f}</h3>
            <p>Avg Health Expenditure/Capita</p></div>""", unsafe_allow_html=True)
    with c4:
        mean_u5 = df["under5_mortality_rate"].mean()
        st.markdown(f"""<div class="metric-card">
            <h3>{mean_u5:.1f}</h3>
            <p>Avg Under-5 Mortality Rate</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Research Questions</div>', unsafe_allow_html=True)
    rq1, rq2, rq3 = st.columns(3)
    with rq1:
        st.info("**RQ1** What socioeconomic factors are the strongest predictors of life expectancy globally?")
    with rq2:
        st.info("**RQ2** How does healthcare investment correlate with country health outcomes?")
    with rq3:
        st.info("**RQ3** Can ML models cluster countries by health profile and forecast future life expectancy?")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">Life Expectancy by Continent</div>', unsafe_allow_html=True)
        cont_avg = df.groupby("continent")["life_expectancy_both"].mean().reset_index().sort_values("life_expectancy_both")
        fig = px.bar(cont_avg, x="life_expectancy_both", y="continent", orientation="h",
                     color="life_expectancy_both", color_continuous_scale="YlGn",
                     labels={"life_expectancy_both": "Life Expectancy (years)", "continent": ""},
                     title="Average Life Expectancy by Continent")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">GDP vs Life Expectancy</div>', unsafe_allow_html=True)
        plot_df = df.dropna(subset=["gdp_per_capita", "life_expectancy_both"])
        fig2 = px.scatter(plot_df, x="gdp_per_capita", y="life_expectancy_both",
                          color="continent", hover_name="country_name", log_x=True,
                          trendline="ols",
                          labels={"gdp_per_capita": "GDP per Capita (log USD)",
                                  "life_expectancy_both": "Life Expectancy (yrs)"},
                          title="GDP vs Life Expectancy (log scale)")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)


# ── World Map ─────────────────────────────────────────────────────────────────
elif page == "🗺️ World Map":
    st.title("🗺️ Global Health Map")

    indicator_map = {
        "Life Expectancy (Both Sexes)":       "life_expectancy_both",
        "Under-5 Mortality Rate":             "under5_mortality_rate",
        "Infant Mortality Rate":              "infant_mortality_rate",
        "Obesity Prevalence":                 "obesity_prevalence",
        "Health Expenditure per Capita (USD)":"health_expenditure_per_capita",
        "GDP per Capita (USD)":               "gdp_per_capita",
        "Physicians per 1,000":               "physicians_per_1000",
        "Tuberculosis Incidence":             "tuberculosis_incidence",
    }
    selected = st.selectbox("Select indicator to map:", list(indicator_map.keys()))
    col_name = indicator_map[selected]

    scale = "YlGn" if "expectancy" in col_name or "gdp" in col_name or "expenditure" in col_name else "YlOrRd_r"
    fig = px.choropleth(
        df.dropna(subset=[col_name]),
        locations="country_code",
        color=col_name,
        hover_name="country_name",
        color_continuous_scale=scale,
        title=f"World Map — {selected}",
        projection="natural earth",
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    n_show = st.slider("Top/Bottom N countries", 5, 20, 10)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Top {n_show}")
        top_df = df[["country_name", col_name]].dropna().sort_values(col_name, ascending=False).head(n_show)
        st.dataframe(top_df.reset_index(drop=True), use_container_width=True)
    with col2:
        st.subheader(f"Bottom {n_show}")
        bot_df = df[["country_name", col_name]].dropna().sort_values(col_name).head(n_show)
        st.dataframe(bot_df.reset_index(drop=True), use_container_width=True)


# ── Health Expenditure ────────────────────────────────────────────────────────
elif page == "💰 Health Expenditure":
    st.title("💰 Health Expenditure Analysis")
    fig = scatter_expenditure_vs_lifeexp(df)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Hospital Beds vs Life Expectancy")
        fig2 = px.scatter(
            df.dropna(subset=["hospital_beds_per_1000", "life_expectancy_both"]),
            x="hospital_beds_per_1000", y="life_expectancy_both",
            color="continent", hover_name="country_name", trendline="ols",
            labels={"hospital_beds_per_1000": "Hospital Beds per 1,000",
                    "life_expectancy_both": "Life Expectancy (yrs)"},
        )
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.subheader("Physicians per 1,000 vs Life Expectancy")
        fig3 = px.scatter(
            df.dropna(subset=["physicians_per_1000", "life_expectancy_both"]),
            x="physicians_per_1000", y="life_expectancy_both",
            color="continent", hover_name="country_name", trendline="ols",
            labels={"physicians_per_1000": "Physicians per 1,000",
                    "life_expectancy_both": "Life Expectancy (yrs)"},
        )
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)


# ── ML Models ─────────────────────────────────────────────────────────────────
elif page == "🤖 ML Models":
    st.title("🤖 Machine Learning Results")
    st.info(
        "Models were trained on the integrated PostgreSQL country profiles. "
        "Run `python run_pipeline.py` to refresh model results."
    )

    tab1, tab2 = st.tabs(["Random Forest", "Linear Regression"])

    with tab1:
        st.subheader("Random Forest Regressor — Life Expectancy Prediction")
        try:
            import pickle, pathlib
            rf_path = pathlib.Path(__file__).parent.parent / "data" / "processed" / "rf_results.pkl"
            with open(rf_path, "rb") as f:
                rf_res = pickle.load(f)

            m = rf_res["metrics"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R²", f"{m['r2']:.4f}")
            c2.metric("RMSE", f"{m['rmse']:.3f} yrs")
            c3.metric("MAE",  f"{m['mae']:.3f} yrs")
            c4.metric("CV R² (mean)", f"{m['cv_r2_mean']:.4f} ± {m['cv_r2_std']:.4f}")

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(bar_feature_importance(rf_res["importances"]), use_container_width=True)
            with col2:
                st.plotly_chart(
                    scatter_actual_vs_predicted(rf_res["y_test"], rf_res["y_pred"]),
                    use_container_width=True
                )
        except FileNotFoundError:
            st.warning("Run the ML pipeline first: `python run_pipeline.py`")

    with tab2:
        st.subheader("Multiple Linear Regression — Interpretable Coefficients")
        try:
            import pickle, pathlib
            lr_path = pathlib.Path(__file__).parent.parent / "data" / "processed" / "lr_results.pkl"
            with open(lr_path, "rb") as f:
                lr_res = pickle.load(f)
            m = lr_res["metrics"]
            c1, c2, c3 = st.columns(3)
            c1.metric("R²",   f"{m['r2']:.4f}")
            c2.metric("RMSE", f"{m['rmse']:.3f} yrs")
            c3.metric("MAE",  f"{m['mae']:.3f} yrs")
            st.plotly_chart(plot_regression_coefficients(lr_res["coef_df"]), use_container_width=True)
            st.subheader("Coefficient Table")
            st.dataframe(lr_res["coef_df"].round(4), use_container_width=True)
        except FileNotFoundError:
            st.warning("Run the ML pipeline first: `python run_pipeline.py`")


# ── Clustering ────────────────────────────────────────────────────────────────
elif page == "🌍 Clustering":
    st.title("🌍 Country Health Clusters (K-Means)")
    st.plotly_chart(scatter_clusters(df), use_container_width=True)
    st.plotly_chart(box_life_exp_by_cluster(df), use_container_width=True)

    if "country_cluster" in df.columns:
        st.subheader("Cluster Summary Statistics")
        cluster_num_cols = [
            "life_expectancy_both", "gdp_per_capita",
            "health_expenditure_per_capita", "under5_mortality_rate",
            "physicians_per_1000",
        ]
        summary = df.groupby("country_cluster")[
            [c for c in cluster_num_cols if c in df.columns]
        ].mean().round(2)
        st.dataframe(summary, use_container_width=True)


# ── Time Series ───────────────────────────────────────────────────────────────
elif page == "📈 Time Series":
    st.title("📈 Life Expectancy Trends & Forecast")
    ts_df = load_who_ts("life_expectancy_both")
    countries = sorted(ts_df["country_name"].dropna().unique())
    selected_country = st.selectbox("Select country", countries,
                                     index=countries.index("Ireland") if "Ireland" in countries else 0)

    ts_sub = ts_df[ts_df["country_name"] == selected_country].sort_values("year")
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=ts_sub["year"], y=ts_sub["value"],
                                mode="lines+markers", name=selected_country,
                                line=dict(color="#2196F3", width=2)))
    fig_ts.update_layout(title=f"Life Expectancy Trend — {selected_country}",
                         xaxis_title="Year", yaxis_title="Life Expectancy (Years)",
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_ts, use_container_width=True)

    # Multi-country comparison
    st.markdown("---")
    st.subheader("Compare Countries")
    compare_list = st.multiselect("Select countries to compare", countries,
                                   default=countries[:5] if len(countries) >= 5 else countries)
    if compare_list:
        cmp_df = ts_df[ts_df["country_name"].isin(compare_list)]
        fig_cmp = px.line(cmp_df, x="year", y="value", color="country_name",
                           markers=True,
                           labels={"value": "Life Expectancy (yrs)", "year": "Year",
                                   "country_name": "Country"},
                           title="Life Expectancy Comparison")
        fig_cmp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cmp, use_container_width=True)


# ── COVID-19 ──────────────────────────────────────────────────────────────────
elif page == "🦠 COVID-19":
    st.title("🦠 COVID-19 Impact Analysis")
    dis_df = load_disease()
    full_df = df.merge(dis_df, on="country_code", how="left")

    st.plotly_chart(bar_covid_by_continent(full_df), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Deaths per Million vs Life Expectancy")
        fig = px.scatter(
            full_df.dropna(subset=["deaths_per_1m_y", "life_expectancy_both"]),
            x="life_expectancy_both", y="deaths_per_1m_y",
            color="continent", hover_name="country_name", log_y=True,
            labels={"deaths_per_1m_y": "COVID Deaths per Million (log)",
                    "life_expectancy_both": "Life Expectancy (yrs)"},
            title="Life Expectancy vs COVID Mortality",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Case Fatality Rate vs Health Expenditure")
        fig2 = px.scatter(
            full_df.dropna(subset=["case_fatality_rate_y", "health_expenditure_per_capita"]),
            x="health_expenditure_per_capita", y="case_fatality_rate_y",
            color="continent", hover_name="country_name", log_x=True,
            labels={"case_fatality_rate_y": "Case Fatality Rate (%)",
                    "health_expenditure_per_capita": "Health Expenditure/Capita (log)"},
            title="Health Expenditure vs CFR",
        )
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top 20 Countries by Deaths per Million")
    top20 = full_df.dropna(subset=["deaths_per_1m_y"]).sort_values("deaths_per_1m_y", ascending=False).head(20)
    fig3 = px.bar(top20, x="country_name", y="deaths_per_1m_y", color="continent",
                   labels={"deaths_per_1m_y": "Deaths per Million", "country_name": "Country"})
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       xaxis_tickangle=-45)
    st.plotly_chart(fig3, use_container_width=True)


# ── Correlations ──────────────────────────────────────────────────────────────
elif page == "🔗 Correlations":
    st.title("🔗 Correlation Analysis")
    corr = load_corr()
    st.plotly_chart(heatmap_correlation(corr), use_container_width=True)

    st.markdown("---")
    st.subheader("Top Correlations with Life Expectancy")
    if "life_expectancy_both" in corr.columns:
        le_corr = corr["life_expectancy_both"].drop("life_expectancy_both").sort_values()
        fig = px.bar(
            x=le_corr.index.str.replace("_", " ").str.title(),
            y=le_corr.values,
            color=le_corr.values,
            color_continuous_scale="RdYlGn",
            labels={"x": "Indicator", "y": "Pearson r"},
            title="Pearson Correlation with Life Expectancy",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          xaxis_tickangle=-40, coloraxis_showscale=False)
        fig.add_hline(y=0, line_dash="dash", line_color="grey")
        st.plotly_chart(fig, use_container_width=True)
