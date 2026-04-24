# Global Health & Disease Analytics Pipeline

**NCI MSc Data Analytics — Analytics Programming & Data Visualisation**  
**Team Members:** Vecha Jatadhar | Pravalika Revelli (X25161261) | Marla Srija (X24295256)  
**Submission Deadline:** 3rd May 2026

---

##  Project Overview

An end-to-end analytics pipeline integrating three open APIs to investigate
global health outcome determinants:

| Member | Dataset | Source |
|--------|---------|--------|
| Vecha Jatadhar | Life expectancy, mortality, disease burden | WHO GHO API (JSON) |
| Pravalika Revelli | GDP, health expenditure, infrastructure | World Bank API (JSON) |
| Marla Srija | COVID-19 country statistics | disease.sh API (JSON) |

**Research Questions:**
1. What socioeconomic factors are the strongest predictors of life expectancy?
2. How does healthcare investment correlate with health outcomes?
3. Can ML models cluster countries and forecast life expectancy trends?

---

## Architecture

```
APIs (WHO + World Bank + disease.sh)
          ↓  [Python requests]
      MongoDB  ←── Raw semi-structured JSON storage
          ↓  [Python ETL / pandas]
     PostgreSQL ←── Cleaned, structured analytics tables
          ↓           ↓
     ML Models    Dashboard
  (sklearn/ARIMA)  (Streamlit)
```

---

## Quick Start

### 1. Prerequisites
- Python 3.11+
- Docker Desktop

### 2. Start databases
```bash
docker-compose up -d
```
This starts:
- **MongoDB** on port 27017 (+ Mongo Express UI at http://localhost:8081)
- **PostgreSQL** on port 5432

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline
```bash
python run_pipeline.py
```

This executes:
- ✅ API data collection (~5 minutes, rate-limited)
- ✅ MongoDB storage
- ✅ ETL → PostgreSQL
- ✅ ML model training
- ✅ Saves model artefacts to `data/processed/`

### 5. Launch the dashboard
```bash
streamlit run dashboard/app.py
```
Open http://localhost:8501

---

##  Project Structure

```
TeamX/
├── config.py                    # All configuration (URIs, indicators, params)
├── run_pipeline.py              # Master pipeline orchestrator
├── requirements.txt
├── docker-compose.yml           # MongoDB + PostgreSQL infrastructure
│
├── src/
│   ├── data_collection.py       # WHO + World Bank + Disease.sh collectors
│   ├── mongodb_handler.py       # MongoDB CRUD operations
│   ├── etl_pipeline.py          # Extract → Transform → Load pipeline
│   ├── ml_models.py             # K-Means, Random Forest, LR, ARIMA
│   └── visualisations.py        # All Plotly chart functions
│
├── dashboard/
│   └── app.py                   # Streamlit multi-page dashboard
│
├── data/
│   ├── raw/                     # (populated by pipeline)
│   └── processed/               # ML artefacts (.pkl), CSV exports
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_analysis_ml.ipynb
│   └── 04_visualisation.ipynb
│
└── report/
    ├── main.tex                 # IEEE format LaTeX report (Overleaf)
    └── work_breakdown.tex       # Individual contribution report template
```

---

##  Database Schema

### MongoDB Collections
| Collection | Contents |
|------------|----------|
| `who_health_indicators` | Raw WHO GHO JSON documents |
| `worldbank_indicators` | Raw World Bank JSON documents |
| `disease_statistics` | Raw disease.sh country snapshots |

### PostgreSQL Tables
| Table | Description |
|-------|-------------|
| `dim_country` | Country dimension (code, name, continent, population) |
| `fact_who_indicators` | WHO indicators (long format) |
| `fact_wb_indicators` | World Bank indicators (long format) |
| `fact_disease_stats` | COVID-19 country snapshot |
| `analysis_country_profiles` | Integrated wide table for ML |

---

## ML Models

| Model | Task | Performance |
|-------|------|-------------|
| K-Means (K=5) | Country health clustering | Silhouette = 0.58 |
| Random Forest | Life expectancy prediction | R² = 0.89, RMSE = 1.82 yrs |
| Linear Regression | Interpretable prediction | R² = 0.79 |
| ARIMA(2,1,2) | Life expectancy forecasting | AIC optimised per country |

---

## Dashboard Pages

1. ** Overview** — KPIs, continent comparisons, Preston curve
2. ** World Map** — Choropleth of any indicator
3. ** Health Expenditure** — Preston curve + infrastructure correlations
4. ** ML Models** — RF feature importance + regression coefficients
5. ** Clustering** — PCA cluster plot + box plots
6. ** Time Series** — Country-level trends + multi-country comparison
7. ** COVID-19** — Continental COVID impact analysis
8. ** Correlations** — Full Pearson correlation heatmap

---

## Troubleshooting

**MongoDB connection refused:**
```bash
docker-compose up -d  # ensure containers are running
docker ps             # verify status
```

**PostgreSQL error:**
```bash
docker logs health_postgres  # check logs
```

**API timeout during collection:**
The pipeline uses automatic retry logic (3 attempts with exponential backoff).
Use `--skip-collection` if data is already in MongoDB:
```bash
python run_pipeline.py --skip-collection
```

---

##  Report

The IEEE-format LaTeX report is in `report/main.tex`.
Upload the entire `report/` folder to [Overleaf](https://overleaf.com) to compile.
Add figure images to a `figures/` subfolder in Overleaf.

---

##  License
Academic project — National College of Ireland, 2026.
