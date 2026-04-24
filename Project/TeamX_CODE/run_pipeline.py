"""
run_pipeline.py
===============
Master orchestrator — runs the entire Global Health Analytics pipeline.

Usage:
    python run_pipeline.py [--skip-collection] [--skip-ml]
"""

import argparse
import logging
import pickle
import pathlib
import sys

# Windows-safe logging: removed fancy Unicode symbols
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", mode="w", encoding="utf-8"),
    ],
)

logger = logging.getLogger("pipeline")

ARTEFACT_DIR = pathlib.Path("data/processed")
ARTEFACT_DIR.mkdir(parents=True, exist_ok=True)


def run_collection_and_storage() -> dict:
    """Step 1 & 2: Collect from APIs and store in MongoDB."""
    from src.data_collection import collect_all_data
    from src.mongodb_handler import MongoHandler

    logger.info("--- STEP 1: Data Collection ---")
    raw = collect_all_data()

    logger.info("--- STEP 2: MongoDB Storage ---")
    mongo = MongoHandler()

    stats = {}
    stats["who"] = mongo.upsert_who_records(raw["who"])
    stats["worldbank"] = mongo.upsert_worldbank_records(raw["worldbank"])
    stats["disease"] = mongo.upsert_disease_records(raw["disease"]["country_summaries"])

    mongo.close()

    hist_path = ARTEFACT_DIR / "global_historical.pkl"
    with open(hist_path, "wb") as f:
        pickle.dump(raw["disease"].get("global_historical", {}), f)

    logger.info(f"MongoDB stats: {stats}")
    return stats


def run_etl():
    """Step 3: ETL pipeline to PostgreSQL."""
    from src.etl_pipeline import ETLPipeline

    logger.info("--- STEP 3: ETL Pipeline ---")

    etl = ETLPipeline()
    profiles = etl.run()

    profiles.to_csv(ARTEFACT_DIR / "country_profiles.csv", index=False)

    logger.info("Country profiles saved to data/processed/country_profiles.csv")
    return profiles


def run_ml() -> dict:
    """Step 4: Run ML models and persist artefacts."""
    from src.ml_models import HealthAnalyticsML

    logger.info("--- STEP 4: Machine Learning ---")

    ml = HealthAnalyticsML()
    results = ml.run_all()

    for key in ["random_forest", "linear_regression", "clustering", "arima"]:
        if key in results:
            out_path = ARTEFACT_DIR / f"{key.replace(' ', '_')}_results.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(results[key], f)

            logger.info(f"Saved {key} artefact -> {out_path}")

    if "random_forest" in results:
        m = results["random_forest"]["metrics"]
        logger.info(
            f"Random Forest | R2={m['r2']:.4f} | RMSE={m['rmse']:.4f} yrs | "
            f"CV-R2={m['cv_r2_mean']:.4f}+/-{m['cv_r2_std']:.4f}"
        )

    if "linear_regression" in results:
        m = results["linear_regression"]["metrics"]
        logger.info(
            f"Linear Regression | R2={m['r2']:.4f} | RMSE={m['rmse']:.4f} yrs"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Global Health Analytics Pipeline")

    parser.add_argument(
        "--skip-collection",
        action="store_true",
        help="Skip API collection and MongoDB storage"
    )

    parser.add_argument(
        "--skip-ml",
        action="store_true",
        help="Skip ML model training"
    )

    args = parser.parse_args()

    logger.info("=== Global Health Analytics Pipeline - START ===")
    logger.info(
        "Team: Vecha Jatadhar | Pravalika Revelli (X25161261) | Marla Srija (X24295256)"
    )

    try:
        if not args.skip_collection:
            run_collection_and_storage()

        profiles = run_etl()

        if not args.skip_ml:
            results = run_ml()

        logger.info("=== Pipeline Complete ===")
        logger.info("Next step: launch dashboard with: streamlit run dashboard/app.py")

    except Exception as exc:
        logger.exception(f"Pipeline failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()