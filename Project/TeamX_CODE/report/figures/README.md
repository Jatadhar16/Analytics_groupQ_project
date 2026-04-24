# Figures for LaTeX Report

Place your exported Plotly PNG images here:

- `feature_importance.png`  — RF feature importance bar chart
- `preston_curve.png`       — Health expenditure vs life expectancy scatter
- `choropleth_le.png`       — World choropleth of life expectancy
- `arima_forecast.png`      — ARIMA forecasts for IRL/DEU/KEN
- `clusters_pca.png`        — PCA cluster scatter
- `correlation_heatmap.png` — Pearson correlation heatmap

Generate these with Notebook 04_visualisation.ipynb:
    fig.write_image('report/figures/FILENAME.png', width=900, height=550, scale=2)
