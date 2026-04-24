"""
ETL Pipeline
============
Extract  → Read raw JSON documents from MongoDB collections
Transform → Clean, normalise, impute, feature-engineer, and merge datasets
Load      → Insert structured DataFrames into PostgreSQL tables

Design rationale
----------------
* MongoDB acts as a raw data lake (schema-flexible, write-once).
* Transformation runs entirely in-memory using pandas, enabling full
  auditability of each step.
* PostgreSQL stores the clean, analysis-ready tables with foreign-key
  relationships and typed columns.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine, text

from config import PG_CONN_STR, TARGET_YEAR
from src.mongodb_handler import MongoHandler

logger = logging.getLogger(__name__)


class ETLPipeline:

    def __init__(self):
        self.mongo   = MongoHandler()
        self.engine  = create_engine(PG_CONN_STR)
        self._create_pg_schema()

    # ══════════════════════════════════════════════════════════════════════════
    #  SCHEMA CREATION
    # ══════════════════════════════════════════════════════════════════════════

    def _create_pg_schema(self):
        ddl = """
        -- Country dimension table
        CREATE TABLE IF NOT EXISTS dim_country (
            country_code  CHAR(3)      PRIMARY KEY,
            country_name  VARCHAR(120),
            continent     VARCHAR(60),
            population    BIGINT
        );

        -- WHO health indicators (long format)
        CREATE TABLE IF NOT EXISTS fact_who_indicators (
            id             SERIAL PRIMARY KEY,
            country_code   CHAR(3)        REFERENCES dim_country(country_code),
            year           SMALLINT,
            indicator_code VARCHAR(40),
            indicator_label VARCHAR(80),
            value          DOUBLE PRECISION,
            low            DOUBLE PRECISION,
            high           DOUBLE PRECISION,
            UNIQUE (country_code, year, indicator_code)
        );

        -- World Bank indicators (long format)
        CREATE TABLE IF NOT EXISTS fact_wb_indicators (
            id             SERIAL PRIMARY KEY,
            country_code   CHAR(3)        REFERENCES dim_country(country_code),
            year           SMALLINT,
            indicator_code VARCHAR(40),
            indicator_label VARCHAR(80),
            value          DOUBLE PRECISION,
            UNIQUE (country_code, year, indicator_code)
        );

        -- Disease / COVID-19 snapshot
        CREATE TABLE IF NOT EXISTS fact_disease_stats (
            country_code          CHAR(3)  PRIMARY KEY REFERENCES dim_country(country_code),
            cases                 BIGINT,
            deaths                BIGINT,
            recovered             BIGINT,
            active                BIGINT,
            cases_per_1m          DOUBLE PRECISION,
            deaths_per_1m         DOUBLE PRECISION,
            tests_per_1m          DOUBLE PRECISION,
            case_fatality_rate    DOUBLE PRECISION,
            vaccinated            BIGINT
        );

        -- Integrated analysis table (wide, one row per country, TARGET_YEAR)
        CREATE TABLE IF NOT EXISTS analysis_country_profiles (
            country_code              CHAR(3) PRIMARY KEY,
            country_name              VARCHAR(120),
            continent                 VARCHAR(60),
            population                BIGINT,

            -- WHO indicators
            life_expectancy_both      DOUBLE PRECISION,
            life_expectancy_male      DOUBLE PRECISION,
            life_expectancy_female    DOUBLE PRECISION,
            under5_mortality_rate     DOUBLE PRECISION,
            infant_mortality_rate     DOUBLE PRECISION,
            obesity_prevalence        DOUBLE PRECISION,
            ncd_mortality_rate        DOUBLE PRECISION,
            tuberculosis_incidence    DOUBLE PRECISION,
            tobacco_smoking_prevalence DOUBLE PRECISION,

            -- World Bank indicators
            gdp_per_capita            DOUBLE PRECISION,
            health_expenditure_per_capita DOUBLE PRECISION,
            hospital_beds_per_1000    DOUBLE PRECISION,
            physicians_per_1000       DOUBLE PRECISION,
            measles_immunization_pct  DOUBLE PRECISION,
            adult_literacy_rate       DOUBLE PRECISION,
            urban_population_pct      DOUBLE PRECISION,
            basic_water_access_pct    DOUBLE PRECISION,

            -- Disease stats
            cases_per_1m              DOUBLE PRECISION,
            deaths_per_1m             DOUBLE PRECISION,
            case_fatality_rate        DOUBLE PRECISION,

            -- ML features (engineered)
            health_system_index       DOUBLE PRECISION,
            socioeconomic_index       DOUBLE PRECISION,
            health_outcome_score      DOUBLE PRECISION,
            country_cluster           SMALLINT
        );
        """
        with self.engine.connect() as conn:
            for stmt in ddl.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(text(stmt))
            conn.commit()
        logger.info("PostgreSQL schema created / verified")

    # ══════════════════════════════════════════════════════════════════════════
    #  EXTRACT
    # ══════════════════════════════════════════════════════════════════════════

    def extract(self):
        logger.info("Extracting raw data from MongoDB …")
        self.who_df     = self.mongo.get_who_dataframe()
        self.wb_df      = self.mongo.get_worldbank_dataframe()
        self.disease_df = self.mongo.get_disease_dataframe()
        logger.info(
            f"Extracted | WHO={len(self.who_df)} | WB={len(self.wb_df)} | Disease={len(self.disease_df)}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  TRANSFORM
    # ══════════════════════════════════════════════════════════════════════════

    def _clean_who(self) -> pd.DataFrame:
        df = self.who_df.copy()
        # Keep only real ISO-3 country codes (exclude WHO regional aggregates)
        df = df[df["country_code"].str.len() == 3]
        df["value"]  = pd.to_numeric(df["value"],  errors="coerce")
        df["year"]   = pd.to_numeric(df["year"],   errors="coerce").astype("Int64")
        df["low"]    = pd.to_numeric(df["low"],    errors="coerce")
        df["high"]   = pd.to_numeric(df["high"],   errors="coerce")
        df = df.dropna(subset=["country_code", "year", "value"])
        # Clip obvious outliers
        df = df[df["value"] >= 0]
        logger.info(f"WHO cleaned: {len(df)} rows")
        return df

    def _clean_worldbank(self) -> pd.DataFrame:
        df = self.wb_df.copy()
        df = df[df["country_code"].str.len() == 3]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["year"]  = pd.to_numeric(df["year"],  errors="coerce").astype("Int64")
        df = df.dropna(subset=["country_code", "year", "value"])
        df = df[df["value"] >= 0]
        logger.info(f"World Bank cleaned: {len(df)} rows")
        return df

    def _clean_disease(self) -> pd.DataFrame:
        df = self.disease_df.copy()
        numeric_cols = [
            "cases", "deaths", "recovered", "active",
            "cases_per_1m", "deaths_per_1m", "tests_per_1m",
            "case_fatality_rate", "vaccinated",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["country_code"])
        df = df[df["country_code"].str.len() == 3]
        logger.info(f"Disease cleaned: {len(df)} rows")
        return df

    def _build_dim_country(self, who_c, wb_c, dis_c) -> pd.DataFrame:
        """Build country dimension from all sources."""
        wb_names = wb_c[["country_code", "country_name"]].drop_duplicates("country_code")
        dis_meta = dis_c[["country_code", "country", "continent", "population"]].rename(
            columns={"country": "dis_name"}
        ).drop_duplicates("country_code")
        dim = wb_names.merge(dis_meta, on="country_code", how="outer")
        dim["country_name"] = dim["country_name"].combine_first(dim.get("dis_name"))
        dim = dim[["country_code", "country_name", "continent", "population"]].drop_duplicates("country_code")
        logger.info(f"dim_country: {len(dim)} countries")
        return dim

    def _build_wide_profile(self, who_c, wb_c, dis_c, dim_c) -> pd.DataFrame:
        """
        Build wide analysis table: one row per country for TARGET_YEAR.
        Missing values are forward/backward filled across years,
        then median-imputed per continent group.
        """
        # WHO pivot
        who_t = who_c[who_c["year"] == TARGET_YEAR].pivot_table(
            index="country_code", columns="indicator_label", values="value", aggfunc="mean"
        ).reset_index()

        # If TARGET_YEAR rows are sparse, use most recent available value
        if len(who_t) < 30:
            who_sorted = who_c.sort_values("year", ascending=False)
            who_t = who_sorted.drop_duplicates(
                ["country_code", "indicator_label"]
            ).pivot_table(
                index="country_code", columns="indicator_label", values="value", aggfunc="mean"
            ).reset_index()

        # WB pivot
        wb_t = wb_c[wb_c["year"] == TARGET_YEAR].pivot_table(
            index="country_code", columns="indicator_label", values="value", aggfunc="mean"
        ).reset_index()

        if len(wb_t) < 30:
            wb_sorted = wb_c.sort_values("year", ascending=False)
            wb_t = wb_sorted.drop_duplicates(
                ["country_code", "indicator_label"]
            ).pivot_table(
                index="country_code", columns="indicator_label", values="value", aggfunc="mean"
            ).reset_index()

        who_t.columns.name = None
        wb_t.columns.name  = None

        # Merge all
        prof = dim_c.merge(who_t,   on="country_code", how="left")
        prof = prof.merge(wb_t,      on="country_code", how="left")
        dis_sub = dis_c[[
            "country_code", "cases_per_1m", "deaths_per_1m",
            "case_fatality_rate"
        ]].drop_duplicates("country_code")
        prof = prof.merge(dis_sub, on="country_code", how="left")

        # Continent-level median imputation
        numeric_cols = prof.select_dtypes(include=np.number).columns.tolist()
        for col in numeric_cols:
            prof[col] = prof.groupby("continent")[col].transform(
                lambda x: x.fillna(x.median())
            )
        # Global median as fallback
        prof[numeric_cols] = prof[numeric_cols].fillna(prof[numeric_cols].median())

        # Feature engineering
        scaler = MinMaxScaler()
        health_sys_cols = [c for c in [
            "health_expenditure_per_capita", "hospital_beds_per_1000",
            "physicians_per_1000", "measles_immunization_pct", "basic_water_access_pct"
        ] if c in prof.columns]
        socio_cols = [c for c in [
            "gdp_per_capita", "adult_literacy_rate", "urban_population_pct"
        ] if c in prof.columns]
        outcome_cols = [c for c in [
            "life_expectancy_both", "under5_mortality_rate", "infant_mortality_rate"
        ] if c in prof.columns]

        if health_sys_cols:
            scaled = scaler.fit_transform(prof[health_sys_cols].fillna(0))
            prof["health_system_index"] = scaled.mean(axis=1)
        if socio_cols:
            scaled = scaler.fit_transform(prof[socio_cols].fillna(0))
            prof["socioeconomic_index"] = scaled.mean(axis=1)
        if outcome_cols:
            scaled_out = scaler.fit_transform(prof[outcome_cols].fillna(0))
            # Life expectancy is positive, mortality is negative
            le_idx = outcome_cols.index("life_expectancy_both") if "life_expectancy_both" in outcome_cols else None
            if le_idx is not None:
                score = scaled_out[:, le_idx].copy()
                for i, col in enumerate(outcome_cols):
                    if "mortality" in col:
                        score -= scaled_out[:, i]
                prof["health_outcome_score"] = score
            else:
                prof["health_outcome_score"] = scaled_out.mean(axis=1)

        logger.info(f"Wide profile built: {prof.shape}")
        return prof

    def transform(self):
        logger.info("Running transformation pipeline …")
        self.who_clean     = self._clean_who()
        self.wb_clean      = self._clean_worldbank()
        self.disease_clean = self._clean_disease()
        self.dim_country   = self._build_dim_country(
            self.who_clean, self.wb_clean, self.disease_clean
        )
        self.country_profiles = self._build_wide_profile(
            self.who_clean, self.wb_clean, self.disease_clean, self.dim_country
        )
        logger.info("Transformation complete")

    # ══════════════════════════════════════════════════════════════════════════
    #  LOAD
    # ══════════════════════════════════════════════════════════════════════════

    def _pg_upsert(self, df: pd.DataFrame, table: str, conflict_col: str = None):
        """Naive truncate-insert (idempotent for reruns)."""
        with self.engine.connect() as conn:
            conn.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
            conn.commit()
        df.to_sql(table, self.engine, if_exists="append", index=False, method="multi", chunksize=500)
        logger.info(f"Loaded {len(df):>6} rows → {table}")

    def load(self):
        logger.info("Loading into PostgreSQL …")

        # dim_country
        self._pg_upsert(
            self.dim_country[["country_code", "country_name", "continent", "population"]],
            "dim_country"
        )

        # fact_who_indicators
        who_load = self.who_clean[[
            "country_code", "year", "indicator_code", "indicator_label", "value", "low", "high"
        ]].copy()
        # keep only countries in dim
        valid = set(self.dim_country["country_code"].dropna())
        who_load = who_load[who_load["country_code"].isin(valid)]
        self._pg_upsert(who_load, "fact_who_indicators")

        # fact_wb_indicators
        wb_load = self.wb_clean[[
            "country_code", "year", "indicator_code", "indicator_label", "value"
        ]].copy()
        wb_load = wb_load[wb_load["country_code"].isin(valid)]
        self._pg_upsert(wb_load, "fact_wb_indicators")

        # fact_disease_stats
        dis_cols = [
            "country_code", "cases", "deaths", "recovered", "active",
            "cases_per_1m", "deaths_per_1m", "tests_per_1m",
            "case_fatality_rate", "vaccinated"
        ]
        dis_load = self.disease_clean[[c for c in dis_cols if c in self.disease_clean.columns]].copy()
        dis_load = dis_load[dis_load["country_code"].isin(valid)]
        self._pg_upsert(dis_load, "fact_disease_stats")

        # analysis_country_profiles
        profile_cols = [c for c in self.country_profiles.columns
                        if c in self._get_profile_schema_cols()]
        self._pg_upsert(self.country_profiles[profile_cols], "analysis_country_profiles")

        logger.info("PostgreSQL load complete")

    def _get_profile_schema_cols(self):
        return [
            "country_code", "country_name", "continent", "population",
            "life_expectancy_both", "life_expectancy_male", "life_expectancy_female",
            "under5_mortality_rate", "infant_mortality_rate", "obesity_prevalence",
            "ncd_mortality_rate", "tuberculosis_incidence", "tobacco_smoking_prevalence",
            "gdp_per_capita", "health_expenditure_per_capita",
            "hospital_beds_per_1000", "physicians_per_1000",
            "measles_immunization_pct", "adult_literacy_rate",
            "urban_population_pct", "basic_water_access_pct",
            "cases_per_1m", "deaths_per_1m", "case_fatality_rate",
            "health_system_index", "socioeconomic_index",
            "health_outcome_score", "country_cluster",
        ]

    # ══════════════════════════════════════════════════════════════════════════
    #  RUN FULL PIPELINE
    # ══════════════════════════════════════════════════════════════════════════

    def run(self):
        self.extract()
        self.transform()
        self.load()
        logger.info("ETL pipeline finished successfully")
        return self.country_profiles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    etl = ETLPipeline()
    profiles = etl.run()
    print(profiles.describe())
