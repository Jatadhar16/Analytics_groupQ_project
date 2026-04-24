"""
Visualisations Module
=====================
All Plotly figures used in both the Streamlit dashboard and the project report.
Each function returns a standalone plotly.graph_objects.Figure.

Design principles applied (per Few & Tufte):
  - Maximise data-ink ratio
  - Avoid chartjunk / 3-D effects
  - Use colour only to encode information (sequential/diverging palettes)
  - Provide tooltips for interactivity
  - Consistent typography and axis labelling
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Colour palette (accessible, sequential) ──────────────────────────────────
PALETTE      = px.colors.sequential.Viridis
DIV_PALETTE  = px.colors.diverging.RdYlGn
CLUSTER_COLS = px.colors.qualitative.Bold
BG           = "rgba(0,0,0,0)"
FONT         = dict(family="Inter, Arial, sans-serif", size=13, color="#333333")


# ══════════════════════════════════════════════════════════════════════════════
#  1. Choropleth — Global Life Expectancy Map
# ══════════════════════════════════════════════════════════════════════════════

def choropleth_life_expectancy(df: pd.DataFrame) -> go.Figure:
    """
    World choropleth of life expectancy at birth (both sexes).
    Encoding: sequential green palette (low → high).
    """
    fig = px.choropleth(
        df,
        locations="country_code",
        color="life_expectancy_both",
        hover_name="country_name",
        hover_data={
            "life_expectancy_both": ":.1f",
            "gdp_per_capita":       ":,.0f",
            "health_expenditure_per_capita": ":,.0f",
            "country_code": False,
        },
        color_continuous_scale="YlGn",
        range_color=[50, 85],
        title="Global Life Expectancy at Birth (Years)",
        labels={"life_expectancy_both": "Life Expectancy (yrs)"},
    )
    fig.update_layout(
        font=FONT,
        paper_bgcolor=BG,
        geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
        coloraxis_colorbar=dict(title="Years", thickness=15),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  2. Scatter — Health Expenditure vs Life Expectancy
# ══════════════════════════════════════════════════════════════════════════════

def scatter_expenditure_vs_lifeexp(df: pd.DataFrame) -> go.Figure:
    """
    Preston curve: health expenditure per capita (log) vs life expectancy.
    Colour encodes continent; size encodes population.
    """
    plot_df = df.dropna(subset=["health_expenditure_per_capita", "life_expectancy_both"])
    plot_df = plot_df[plot_df["health_expenditure_per_capita"] > 0]

    fig = px.scatter(
        plot_df,
        x="health_expenditure_per_capita",
        y="life_expectancy_both",
        color="continent",
        size="population",
        size_max=40,
        hover_name="country_name",
        hover_data={
            "gdp_per_capita": ":,.0f",
            "health_expenditure_per_capita": ":,.0f",
            "life_expectancy_both": ":.1f",
            "population": ":,",
        },
        log_x=True,
        trendline="ols",
        title="Health Expenditure vs. Life Expectancy (Preston Curve)",
        labels={
            "health_expenditure_per_capita": "Health Expenditure per Capita (USD, log)",
            "life_expectancy_both": "Life Expectancy at Birth (Years)",
        },
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_layout(font=FONT, paper_bgcolor=BG, plot_bgcolor=BG,
                      legend_title="Continent")
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  3. Bar — Feature Importance (Random Forest)
# ══════════════════════════════════════════════════════════════════════════════

def bar_feature_importance(importances: pd.Series) -> go.Figure:
    """
    Horizontal bar chart of RF feature importances.
    Encoding: colour intensity proportional to importance.
    """
    top = importances.head(10).sort_values()
    fig = go.Figure(go.Bar(
        x=top.values,
        y=top.index.str.replace("_", " ").str.title(),
        orientation="h",
        marker=dict(
            color=top.values,
            colorscale="Blues",
            showscale=False,
        ),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Top 10 Predictors of Life Expectancy (Random Forest)",
        xaxis_title="Feature Importance (Gini)",
        yaxis_title="",
        font=FONT,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        margin=dict(l=180, r=20, t=60, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  4. Scatter — Actual vs Predicted Life Expectancy
# ══════════════════════════════════════════════════════════════════════════════

def scatter_actual_vs_predicted(y_test, y_pred, model_name: str = "Random Forest") -> go.Figure:
    residuals = np.array(y_test) - np.array(y_pred)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test, y=y_pred,
        mode="markers",
        marker=dict(color=residuals, colorscale="RdYlGn", size=8,
                    colorbar=dict(title="Residual"), showscale=True),
        hovertemplate="Actual: %{x:.1f}<br>Predicted: %{y:.1f}<extra></extra>",
        name="Countries",
    ))
    mn, mx = min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx], mode="lines",
        line=dict(dash="dash", color="grey"), name="Perfect fit",
    ))
    fig.update_layout(
        title=f"Actual vs Predicted Life Expectancy — {model_name}",
        xaxis_title="Actual Life Expectancy (Years)",
        yaxis_title="Predicted Life Expectancy (Years)",
        font=FONT, paper_bgcolor=BG, plot_bgcolor=BG,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  5. PCA + Cluster Scatter
# ══════════════════════════════════════════════════════════════════════════════

def scatter_clusters(df: pd.DataFrame) -> go.Figure:
    """
    2-D PCA projection coloured by K-Means cluster.
    """
    if "pca_x" not in df.columns:
        return go.Figure()

    cluster_labels = {
        0: "High-Income, High-Health",
        1: "Upper-Middle Development",
        2: "Mid-Development",
        3: "Lower-Middle Development",
        4: "Low-Income, Low-Health",
    }
    df = df.copy()
    df["cluster_label"] = df["country_cluster"].map(cluster_labels).fillna("Unknown")

    fig = px.scatter(
        df.dropna(subset=["pca_x", "pca_y"]),
        x="pca_x", y="pca_y",
        color="cluster_label",
        hover_name="country_name",
        hover_data={
            "life_expectancy_both": ":.1f",
            "gdp_per_capita": ":,.0f",
            "pca_x": False, "pca_y": False,
        },
        title="Country Health Clusters (PCA Projection)",
        labels={"pca_x": "PCA Component 1", "pca_y": "PCA Component 2"},
        color_discrete_sequence=CLUSTER_COLS,
    )
    fig.update_layout(font=FONT, paper_bgcolor=BG, plot_bgcolor=BG,
                      legend_title="Cluster")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  6. Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def heatmap_correlation(corr_matrix: pd.DataFrame) -> go.Figure:
    cols = [
        "life_expectancy_both", "gdp_per_capita",
        "health_expenditure_per_capita", "physicians_per_1000",
        "hospital_beds_per_1000", "adult_literacy_rate",
        "under5_mortality_rate", "infant_mortality_rate",
        "obesity_prevalence", "tuberculosis_incidence",
    ]
    cols = [c for c in cols if c in corr_matrix.columns]
    sub  = corr_matrix.loc[cols, cols]
    labels = [c.replace("_", " ").title() for c in cols]

    fig = go.Figure(go.Heatmap(
        z=sub.values,
        x=labels, y=labels,
        colorscale="RdBu",
        zmid=0,
        zmin=-1, zmax=1,
        hovertemplate="Row: %{y}<br>Col: %{x}<br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(title="Pearson r"),
    ))
    fig.update_layout(
        title="Pearson Correlation Matrix — Health Indicators",
        font=FONT, paper_bgcolor=BG,
        width=700, height=700,
        xaxis=dict(tickangle=-45),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  7. Line — ARIMA Forecast
# ══════════════════════════════════════════════════════════════════════════════

def line_arima_forecast(arima_result: dict) -> go.Figure:
    if not arima_result:
        return go.Figure()

    series   = arima_result["series"].dropna()
    forecast = arima_result["forecast"]
    country  = arima_result["country"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index.year, y=series.values,
        mode="lines+markers", name="Historical",
        line=dict(color="#2196F3", width=2),
        marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=forecast.index.year, y=forecast.values,
        mode="lines+markers", name="Forecast (ARIMA)",
        line=dict(color="#FF5722", dash="dash", width=2),
        marker=dict(size=5, symbol="diamond"),
    ))
    fig.update_layout(
        title=f"Life Expectancy Forecast — {country} (ARIMA)",
        xaxis_title="Year",
        yaxis_title="Life Expectancy (Years)",
        font=FONT, paper_bgcolor=BG, plot_bgcolor=BG,
        legend=dict(x=0.02, y=0.98),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  8. Box — Life Expectancy Distribution per Cluster
# ══════════════════════════════════════════════════════════════════════════════

def box_life_exp_by_cluster(df: pd.DataFrame) -> go.Figure:
    df = df.dropna(subset=["life_expectancy_both", "country_cluster"])
    df["cluster_label"] = df["country_cluster"].apply(lambda x: f"Cluster {int(x)}")

    fig = px.box(
        df.sort_values("country_cluster"),
        x="cluster_label",
        y="life_expectancy_both",
        color="cluster_label",
        points="all",
        hover_name="country_name",
        title="Life Expectancy Distribution by Health Cluster",
        labels={"life_expectancy_both": "Life Expectancy (Years)", "cluster_label": "Cluster"},
        color_discrete_sequence=CLUSTER_COLS,
    )
    fig.update_layout(font=FONT, paper_bgcolor=BG, plot_bgcolor=BG, showlegend=False)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  9. Multi-panel COVID overview
# ══════════════════════════════════════════════════════════════════════════════

def bar_covid_by_continent(df: pd.DataFrame) -> go.Figure:
    required_cols = [
        "continent",
        "cases_per_1m",
        "deaths_per_1m",
        "case_fatality_rate",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        fig = go.Figure()
        fig.add_annotation(
            text=(
                "COVID-19 continent chart unavailable.<br>"
                f"Missing columns: {', '.join(missing_cols)}"
            ),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="#666666"),
            align="center",
        )
        fig.update_layout(
            title="COVID-19 Impact by Continent",
            font=FONT,
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    cont = (
        df.groupby("continent")[["cases_per_1m", "deaths_per_1m", "case_fatality_rate"]]
        .mean(numeric_only=True)
        .reset_index()
        .dropna()
    )

    if cont.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="COVID-19 data is empty after filtering.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="#666666"),
        )
        fig.update_layout(
            title="COVID-19 Impact by Continent",
            font=FONT,
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Cases per Million",
            "Deaths per Million",
            "Case Fatality Rate (%)",
        ],
    )

    for i, col in enumerate(["cases_per_1m", "deaths_per_1m", "case_fatality_rate"]):
        fig.add_trace(
            go.Bar(
                x=cont["continent"],
                y=cont[col],
                marker_color=CLUSTER_COLS[i],
                name=col.replace("_", " ").title(),
                showlegend=False,
            ),
            row=1,
            col=i + 1,
        )

    fig.update_layout(
        title="COVID-19 Impact by Continent",
        font=FONT,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        height=400,
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  10. Coefficient Plot (Linear Regression)
# ══════════════════════════════════════════════════════════════════════════════

def plot_regression_coefficients(coef_df: pd.DataFrame) -> go.Figure:
    df = coef_df[coef_df["feature"] != "intercept"].copy()
    df["significant"] = df["p_value"] < 0.05
    df["colour"] = df["significant"].map({True: "#1976D2", False: "#BDBDBD"})

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["feature"].str.replace("_", " ").str.title(),
        y=df["coefficient"],
        marker_color=df["colour"].tolist(),
        error_y=dict(type="data", array=df["std_err"].tolist(), visible=True),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "β = %{y:.4f}<br>"
            "p-value = %{customdata:.4f}<extra></extra>"
        ),
        customdata=df["p_value"].tolist(),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_layout(
        title="Linear Regression Coefficients (Standardised) — Life Expectancy",
        xaxis_title="Feature",
        yaxis_title="Coefficient (β)",
        font=FONT, paper_bgcolor=BG, plot_bgcolor=BG,
        xaxis_tickangle=-35,
        annotations=[dict(
            x=1, y=1.05, xref="paper", yref="paper",
            text="Blue = significant (p < 0.05)", showarrow=False,
            font=dict(size=11, color="#1976D2"),
        )],
    )
    return fig
