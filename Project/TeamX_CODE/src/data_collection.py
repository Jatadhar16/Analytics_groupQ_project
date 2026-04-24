"""
Data Collection Module
======================
Fetches semi-structured JSON data from WHO, World Bank, and Disease.sh APIs.
"""

import time
import logging
import requests
from typing import Dict, List, Optional, Any

from config import (
    WHO_BASE_URL, WHO_INDICATORS,
    WB_BASE_URL, WB_INDICATORS,
    DISEASE_BASE_URL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)


def _get(url: str, params: dict = None, retries: int = 1) -> Optional[Any]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0",
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=40)
            resp.raise_for_status()
            return resp.json()

        except Exception as exc:
            logger.warning(f"Attempt {attempt}/{retries} failed for {url}: {exc}")
            try:
                logger.warning(f"Response preview: {resp.text[:300]}")
            except Exception:
                pass
            time.sleep(1)

    logger.error(f"All retries exhausted for {url}")
    return None


class WHOCollector:
    """Collects indicator data from WHO GHO API."""

    def __init__(self):
        self.base = WHO_BASE_URL
        self.indicators = WHO_INDICATORS

    def fetch_indicator(self, code: str, label: str) -> List[dict]:
        url = f"{self.base}/{code}"

        # Do NOT use $filter or $select here.
        # WHO API is failing JSON parsing with those query params on your system.
        data = _get(url, params=None, retries=1)

        if not data or "value" not in data:
            logger.warning(f"WHO skipped {code} - no valid data returned")
            return []

        records = []

        for item in data.get("value", []):
            year_raw = item.get("TimeDimensionValue")

            try:
                year = int(year_raw)
            except (TypeError, ValueError):
                continue

            if year < 2010 or year > 2022:
                continue

            records.append({
                "indicator_code": code,
                "indicator_label": label,
                "country_code": item.get("SpatialDim"),
                "year": year,
                "value": item.get("NumericValue"),
                "low": item.get("Low"),
                "high": item.get("High"),
                "source": "WHO GHO API",
                "_raw": item,
            })

        logger.info(f"WHO | {code:<25} | {len(records):>6} records")
        return records

    def collect_all(self) -> List[dict]:
        all_records: List[dict] = []

        for code, label in self.indicators.items():
            records = self.fetch_indicator(code, label)
            all_records.extend(records)
            time.sleep(0.5)

        logger.info(f"WHO collection complete - {len(all_records)} total records")
        return all_records


class WorldBankCollector:
    """Collects World Bank indicator data."""

    def __init__(self):
        self.base = WB_BASE_URL
        self.indicators = WB_INDICATORS

    def fetch_indicator(self, code: str, label: str) -> List[dict]:
        url = f"{self.base}/country/all/indicator/{code}"

        params = {
            "format": "json",
            "date": "2010:2022",
            "per_page": 2000,
        }

        data = _get(url, params=params, retries=1)

        if not data or not isinstance(data, list) or len(data) < 2:
            logger.warning(f"World Bank skipped {code} - no valid data returned")
            return []

        meta = data[0] or {}
        items = data[1] or []

        records = []

        for item in items:
            country = item.get("country", {})

            try:
                year = int(item.get("date"))
            except (TypeError, ValueError):
                continue

            records.append({
                "indicator_code": code,
                "indicator_label": label,
                "country_code": item.get("countryiso3code") or country.get("id"),
                "country_name": country.get("value"),
                "year": year,
                "value": item.get("value"),
                "source": "World Bank API",
                "total_pages": meta.get("pages"),
                "_raw": item,
            })

        logger.info(f"WB | {code:<30} | {len(records):>6} records")
        return records

    def collect_all(self) -> List[dict]:
        all_records: List[dict] = []

        for code, label in self.indicators.items():
            records = self.fetch_indicator(code, label)
            all_records.extend(records)
            time.sleep(0.3)

        logger.info(f"World Bank collection complete - {len(all_records)} total records")
        return all_records


class DiseaseCollector:
    """Collects COVID-19 data from disease.sh API."""

    def __init__(self):
        self.base = DISEASE_BASE_URL

    def fetch_country_summary(self) -> List[dict]:
        data = _get(f"{self.base}/countries", {"sort": "cases"}, retries=1)

        if not data:
            logger.warning("Disease.sh country summary skipped - no valid data returned")
            return []

        records = []

        for item in data:
            cases = item.get("cases") or 0
            deaths = item.get("deaths") or 0

            records.append({
                "country": item.get("country"),
                "country_code": item.get("countryInfo", {}).get("iso3"),
                "continent": item.get("continent"),
                "cases": cases,
                "deaths": deaths,
                "recovered": item.get("recovered"),
                "active": item.get("active"),
                "critical": item.get("critical"),
                "cases_per_1m": item.get("casesPerOneMillion"),
                "deaths_per_1m": item.get("deathsPerOneMillion"),
                "tests": item.get("tests"),
                "tests_per_1m": item.get("testsPerOneMillion"),
                "population": item.get("population"),
                "vaccinated": item.get("oneDoseVaccinated"),
                "case_fatality_rate": (
                    round(deaths / cases * 100, 4) if cases > 0 else None
                ),
                "source": "disease.sh API",
                "_raw": item,
            })

        logger.info(f"Disease | country summaries | {len(records)} records")
        return records

    def fetch_global_historical(self, lastdays: int = 720) -> dict:
        data = _get(f"{self.base}/historical/all", {"lastdays": lastdays}, retries=1)

        if not data:
            logger.warning("Disease.sh global historical skipped - no valid data returned")
            return {}

        logger.info("Disease | global historical timeline fetched")

        return {
            "scope": "global",
            "lastdays": lastdays,
            "timeline": data,
            "source": "disease.sh API",
        }

    def fetch_country_historical(self, country: str, lastdays: int = 720) -> dict:
        data = _get(
            f"{self.base}/historical/{country}",
            {"lastdays": lastdays},
            retries=1,
        )

        if not data:
            return {}

        return {
            "country": country,
            "lastdays": lastdays,
            "timeline": data.get("timeline", {}),
            "source": "disease.sh API",
        }

    def collect_all(self) -> Dict[str, object]:
        summaries = self.fetch_country_summary()
        historical = self.fetch_global_historical()

        logger.info("Disease.sh collection complete")

        return {
            "country_summaries": summaries,
            "global_historical": historical,
        }


def collect_all_data() -> Dict[str, object]:
    logger.info("=== Starting data collection from all sources ===")

    who_data = WHOCollector().collect_all()
    wb_data = WorldBankCollector().collect_all()
    disease_data = DiseaseCollector().collect_all()

    logger.info("=== Data collection complete ===")

    return {
        "who": who_data,
        "worldbank": wb_data,
        "disease": disease_data,
    }


if __name__ == "__main__":
    results = collect_all_data()

    print(f"\nWHO records:        {len(results['who'])}")
    print(f"World Bank records: {len(results['worldbank'])}")
    print(f"Disease countries:  {len(results['disease']['country_summaries'])}")