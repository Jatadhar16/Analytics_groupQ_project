"""
Machine Learning Models
=======================
Safe ML pipeline for Global Health Analytics.

Fixes:
- Handles missing values properly
- Prevents n_samples=0 error
- Skips models safely if target/features are unavailable
- Keeps clustering, RF, Linear Regression, ARIMA, correlation
"""

import logging
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from scipy import stats
from sqlalchemy import create_engine, text

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from config import PG_CONN_STR, N_CLUSTERS, RANDOM_STATE, TEST_SIZE

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class HealthAnalyticsML:

    FEATURES = [
        "gdp_per_capita",
        "health_expenditure_per_capita",
        "hospital_beds_per_1000",
        "physicians_per_1000",
        "measles_immunization_pct",
        "adult_literacy_rate",
        "urban_population_pct",
        "basic_water_access_pct",
        "obesity_prevalence",
        "tobacco_smoking_prevalence",
        "tuberculosis_incidence",
    ]

    TARGET = "life_expectancy_both"

    def __init__(self):
        self.engine = create_engine(PG_CONN_STR)
        self.results = {}

    def load_profiles(self) -> pd.DataFrame:
        df = pd.read_sql("SELECT * FROM analysis_country_profiles", self.engine)
        logger.info(f"Loaded {len(df)} country profiles from PostgreSQL")
        return df

    def load_time_series(self, indicator: str = "life_expectancy_both") -> pd.DataFrame:
        q = text("""
            SELECT country_code, year, value
            FROM fact_who_indicators
            WHERE indicator_label = :ind
            ORDER BY country_code, year
        """)
        with self.engine.connect() as conn:
            df = pd.read_sql(q, conn, params={"ind": indicator})
        return df

    def _prepare_supervised_data(self, df: pd.DataFrame, features: list):
        available = [f for f in features if f in df.columns]

        if self.TARGET not in df.columns:
            logger.warning(f"Target column missing: {self.TARGET}")
            return None, None, []

        if not available:
            logger.warning("No available ML feature columns found")
            return None, None, []

        data = df[available + [self.TARGET]].copy()

        for col in available + [self.TARGET]:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        # Keep rows with target only
        data = data.dropna(subset=[self.TARGET])

        if data.empty:
            logger.warning("No rows with valid target value")
            return None, None, []

        # Remove columns that are fully empty
        usable_features = [
            col for col in available
            if data[col].notna().sum() > 0
        ]

        if not usable_features:
            logger.warning("No usable feature columns after checking missing values")
            return None, None, []

        X = data[usable_features].copy()
        y = data[self.TARGET].copy()

        # Fill feature missing values using median, then 0 fallback
        X = X.fillna(X.median(numeric_only=True))
        X = X.fillna(0)

        return X, y, usable_features

    def run_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Running K-Means clustering")

        cluster_features = [
            "health_system_index",
            "socioeconomic_index",
            "life_expectancy_both",
            "under5_mortality_rate",
            "gdp_per_capita",
            "health_expenditure_per_capita",
        ]

        available = [f for f in cluster_features if f in df.columns]

        if not available:
            logger.warning("Skipping clustering - no cluster features found")
            df["country_cluster"] = np.nan
            return df

        X = df[available].copy()

        for col in available:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        X = X.fillna(X.median(numeric_only=True)).fillna(0)

        if len(X) < 2:
            logger.warning("Skipping clustering - not enough rows")
            df["country_cluster"] = np.nan
            return df

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        safe_k = min(N_CLUSTERS, len(X))
        safe_k = max(1, safe_k)

        inertias = {}
        for k in range(2, min(9, len(X) + 1)):
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            km.fit(X_scaled)
            inertias[k] = km.inertia_

        kmeans = KMeans(n_clusters=safe_k, random_state=RANDOM_STATE, n_init=20)

        df = df.copy()
        df["country_cluster"] = kmeans.fit_predict(X_scaled)

        if X_scaled.shape[1] >= 2:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(X_scaled)
            df["pca_x"] = coords[:, 0]
            df["pca_y"] = coords[:, 1]
            pca_variance = pca.explained_variance_ratio_
        else:
            df["pca_x"] = X_scaled[:, 0]
            df["pca_y"] = 0
            pca_variance = np.array([1.0, 0.0])

        cluster_summary = df.groupby("country_cluster")[available + ["country_name"]].agg(
            {**{f: "mean" for f in available}, "country_name": "count"}
        ).rename(columns={"country_name": "n_countries"})

        self.results["clustering"] = {
            "model": kmeans,
            "scaler": scaler,
            "inertias": inertias,
            "cluster_summary": cluster_summary,
            "pca_variance": pca_variance,
            "features": available,
        }

        logger.info(f"Clustering complete | {safe_k} clusters")
        return df

    def run_random_forest(self, df: pd.DataFrame) -> dict:
        logger.info("Training Random Forest")

        X, y, available_features = self._prepare_supervised_data(df, self.FEATURES)

        if X is None or y is None or len(X) < 5:
            logger.warning("Skipping Random Forest - not enough usable rows")
            self.results["random_forest"] = {
                "skipped": True,
                "reason": "Not enough usable rows/features",
            }
            return self.results["random_forest"]

        test_size = TEST_SIZE if len(X) >= 10 else 0.3

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=RANDOM_STATE,
        )

        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        cv = min(5, len(X))
        if cv >= 2:
            cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="r2")
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std())
        else:
            cv_mean = 0.0
            cv_std = 0.0

        importances = pd.Series(
            rf.feature_importances_,
            index=available_features,
        ).sort_values(ascending=False)

        metrics = {
            "r2": float(r2_score(y_test, y_pred)) if len(y_test) > 1 else 0.0,
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "cv_r2_mean": cv_mean,
            "cv_r2_std": cv_std,
        }

        self.results["random_forest"] = {
            "model": rf,
            "best_params": {
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_split": 2,
            },
            "metrics": metrics,
            "importances": importances,
            "y_test": y_test,
            "y_pred": y_pred,
            "X_test": X_test,
        }

        logger.info(f"RF | R2={metrics['r2']:.4f} | RMSE={metrics['rmse']:.4f}")
        return self.results["random_forest"]

    def run_linear_regression(self, df: pd.DataFrame) -> dict:
        logger.info("Running Linear Regression")

        reg_features = [
            "gdp_per_capita",
            "health_expenditure_per_capita",
            "physicians_per_1000",
            "adult_literacy_rate",
            "basic_water_access_pct",
            "urban_population_pct",
        ]

        X, y, available = self._prepare_supervised_data(df, reg_features)

        if X is None or y is None or len(X) < 5:
            logger.warning("Skipping Linear Regression - not enough usable rows")
            self.results["linear_regression"] = {
                "skipped": True,
                "reason": "Not enough usable rows/features",
            }
            return self.results["linear_regression"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=TEST_SIZE if len(X) >= 10 else 0.3,
            random_state=RANDOM_STATE,
        )

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)

        n, p = len(X), len(available)

        try:
            X_full = np.hstack([np.ones((n, 1)), X_scaled])
            beta = np.linalg.lstsq(X_full, y.values, rcond=None)[0]
            y_hat = X_full @ beta
            resid = y.values - y_hat

            dof = max(n - p - 1, 1)
            s2 = resid.T @ resid / dof
            cov_b = s2 * np.linalg.pinv(X_full.T @ X_full)
            se = np.sqrt(np.diag(cov_b))
            t_stat = beta / se
            p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=dof))

            coef_df = pd.DataFrame({
                "feature": ["intercept"] + available,
                "coefficient": beta,
                "std_err": se,
                "t_stat": t_stat,
                "p_value": p_val,
            })

        except Exception as exc:
            logger.warning(f"Coefficient statistics skipped: {exc}")
            coef_df = pd.DataFrame({
                "feature": ["intercept"] + available,
                "coefficient": [lr.intercept_] + list(lr.coef_),
            })

        metrics = {
            "r2": float(r2_score(y_test, y_pred)) if len(y_test) > 1 else 0.0,
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
        }

        self.results["linear_regression"] = {
            "model": lr,
            "scaler": scaler,
            "coef_df": coef_df,
            "metrics": metrics,
            "y_test": y_test,
            "y_pred": y_pred,
        }

        logger.info(f"LR | R2={metrics['r2']:.4f} | RMSE={metrics['rmse']:.4f}")
        return self.results["linear_regression"]

    def run_arima(self, ts_df: pd.DataFrame, country_code: str = "IRL", n_forecast: int = 10) -> dict:
        if not HAS_STATSMODELS:
            logger.warning("statsmodels not installed - skipping ARIMA")
            self.results["arima"] = {
                "skipped": True,
                "reason": "statsmodels not installed",
            }
            return self.results["arima"]

        logger.info(f"Running ARIMA for {country_code}")

        if ts_df.empty:
            logger.warning("Skipping ARIMA - empty time series table")
            self.results["arima"] = {
                "skipped": True,
                "reason": "Empty time series",
            }
            return self.results["arima"]

        series = (
            ts_df[ts_df["country_code"] == country_code]
            .set_index("year")["value"]
            .sort_index()
        )

        series = pd.to_numeric(series, errors="coerce").dropna()

        if len(series) < 6:
            logger.warning(f"Skipping ARIMA - not enough data for {country_code}")
            self.results["arima"] = {
                "skipped": True,
                "reason": f"Not enough time-series data for {country_code}",
            }
            return self.results["arima"]

        try:
            series.index = pd.to_datetime(series.index.astype(int), format="%Y")
            series = series.asfreq("YS")

            adf_result = adfuller(series.dropna())
            is_stationary = adf_result[1] < 0.05
            d = 0 if is_stationary else 1

            model = ARIMA(series.dropna(), order=(1, d, 1))
            fitted = model.fit()
            forecast = fitted.forecast(steps=n_forecast)

            forecast_years = pd.date_range(
                start=series.index[-1] + pd.DateOffset(years=1),
                periods=n_forecast,
                freq="YS",
            )

            self.results["arima"] = {
                "country": country_code,
                "series": series,
                "fitted": fitted,
                "forecast": pd.Series(forecast.values, index=forecast_years),
                "adf_pvalue": adf_result[1],
                "is_stationary": is_stationary,
                "aic": fitted.aic,
                "bic": fitted.bic,
            }

            logger.info(f"ARIMA complete | AIC={fitted.aic:.2f}")
            return self.results["arima"]

        except Exception as exc:
            logger.warning(f"ARIMA skipped due to error: {exc}")
            self.results["arima"] = {
                "skipped": True,
                "reason": str(exc),
            }
            return self.results["arima"]

    def run_correlation_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric = df.select_dtypes(include=np.number).drop(
            columns=["pca_x", "pca_y", "country_cluster"],
            errors="ignore",
        )

        corr = numeric.corr(method="pearson")
        self.results["correlation_matrix"] = corr

        logger.info("Correlation analysis complete")
        return corr

    def run_all(self) -> dict:
        df = self.load_profiles()
        ts_df = self.load_time_series()

        df = self.run_clustering(df)

        self.run_random_forest(df)
        self.run_linear_regression(df)
        self.run_arima(ts_df)
        self.run_correlation_analysis(df)

        cluster_map = df[["country_code", "country_cluster"]].dropna()

        with self.engine.connect() as conn:
            for _, row in cluster_map.iterrows():
                conn.execute(
                    text("""
                        UPDATE analysis_country_profiles
                        SET country_cluster = :c
                        WHERE country_code = :cc
                    """),
                    {
                        "c": int(row["country_cluster"]),
                        "cc": row["country_code"],
                    },
                )
            conn.commit()

        logger.info("Cluster labels persisted to PostgreSQL")

        self.df = df
        return self.results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ml = HealthAnalyticsML()
    results = ml.run_all()

    if "random_forest" in results and not results["random_forest"].get("skipped"):
        rf = results["random_forest"]
        print("\nTop 5 most important features for life expectancy:")
        print(rf["importances"].head())
        print(f"\nRandom Forest R2: {rf['metrics']['r2']:.4f}")
    else:
        print("\nRandom Forest skipped.")
        print(results.get("random_forest", {}))