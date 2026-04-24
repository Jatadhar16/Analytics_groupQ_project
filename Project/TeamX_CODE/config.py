"""
Configuration file for Global Health Analytics Project
Team: Vecha Jatadhar | Pravalika Revelli (X25161261) | Marla Srija (X24295256)
"""

# ─── MongoDB Configuration ───────────────────────────────────────────────────
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB = "global_health_db"
MONGO_COLLECTIONS = {
    "who_raw": "who_health_indicators",
    "worldbank_raw": "worldbank_indicators",
    "disease_raw": "disease_statistics",
}

# ─── PostgreSQL Configuration ─────────────────────────────────────────────────
PG_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "global_health_analytics",
    "user": "postgres",
    "password": "postgres",
}
PG_CONN_STR = "postgresql://postgres:pola@localhost:5432/global_health_analytics"

# ─── WHO GHO API ─────────────────────────────────────────────────────────────
WHO_BASE_URL = "https://ghoapi.azureedge.net/api"
WHO_INDICATORS = {
    "WHOSIS_000001": "life_expectancy_both",
    "WHOSIS_000002": "life_expectancy_male",
    "WHOSIS_000015": "life_expectancy_female",
    "MDG_0000000001": "under5_mortality_rate",
    "MDG_0000000003": "infant_mortality_rate",
    "NCD_BMI_30A":    "obesity_prevalence",
    "SDGPM25":        "air_pollution_pm25",
    "WHS4_100":       "tuberculosis_incidence",
    "MORT_500":       "ncd_mortality_rate",
    "SA_0000001462":  "tobacco_smoking_prevalence",
}

# ─── World Bank API ───────────────────────────────────────────────────────────
WB_BASE_URL = "https://api.worldbank.org/v2"
WB_INDICATORS = {
    "SH.XPD.CHEX.PC.CD": "health_expenditure_per_capita",
    "NY.GDP.PCAP.CD":    "gdp_per_capita",
    "SP.POP.TOTL":       "population_total",
    "SH.MED.BEDS.ZS":   "hospital_beds_per_1000",
    "SH.MED.PHYS.ZS":   "physicians_per_1000",
    "SH.IMM.MEAS":       "measles_immunization_pct",
    "SH.DYN.NMRT":       "neonatal_mortality_rate",
    "SE.ADT.LITR.ZS":    "adult_literacy_rate",
    "SP.URB.TOTL.IN.ZS": "urban_population_pct",
    "SH.H2O.BASW.ZS":    "basic_water_access_pct",
}

# ─── Disease.sh API ───────────────────────────────────────────────────────────
DISEASE_BASE_URL = "https://disease.sh/v3/covid-19"

# ─── Analysis Settings ────────────────────────────────────────────────────────
YEARS = list(range(2010, 2023))
TARGET_YEAR = 2019            # pre-COVID baseline for main analysis
N_CLUSTERS = 5                # K-Means clusters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ─── Dashboard ────────────────────────────────────────────────────────────────
DASHBOARD_TITLE = "Global Health & Disease Analytics Dashboard"
DASHBOARD_PORT = 8501
