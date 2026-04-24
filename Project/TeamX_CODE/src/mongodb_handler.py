"""
MongoDB Handler
===============
Stores raw semi-structured JSON documents collected from APIs.
MongoDB is chosen for this stage because:
  - Schema-less storage preserves original JSON structure (including nested _raw fields)
  - Ideal for heterogeneous API responses before normalisation
  - Scales horizontally for large volumes of raw documents
"""

import logging
from typing import Any, Dict, List, Optional

from pymongo import MongoClient, ASCENDING, errors

from config import MONGO_URI, MONGO_DB, MONGO_COLLECTIONS

logger = logging.getLogger(__name__)


class MongoHandler:
    """
    Thin wrapper around PyMongo providing insert, query and bulk-upsert
    operations for the three raw collections.
    """

    def __init__(self):
        self.client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        self.db     = self.client[MONGO_DB]
        self._ensure_indexes()
        logger.info(f"Connected to MongoDB: {MONGO_URI} | DB: {MONGO_DB}")

    def _ensure_indexes(self):
        """Create compound indexes to speed up queries and prevent duplicates."""
        try:
            who_col = self.db[MONGO_COLLECTIONS["who_raw"]]
            who_col.create_index(
                [("indicator_code", ASCENDING), ("country_code", ASCENDING), ("year", ASCENDING)],
                unique=True, background=True
            )

            wb_col = self.db[MONGO_COLLECTIONS["worldbank_raw"]]
            wb_col.create_index(
                [("indicator_code", ASCENDING), ("country_code", ASCENDING), ("year", ASCENDING)],
                unique=True, background=True
            )

            dis_col = self.db[MONGO_COLLECTIONS["disease_raw"]]
            dis_col.create_index(
                [("country_code", ASCENDING)],
                unique=True, background=True
            )
            logger.info("MongoDB indexes ensured")
        except errors.OperationFailure as e:
            logger.warning(f"Index creation warning: {e}")

    # ── Insert helpers ────────────────────────────────────────────────────────

    def upsert_who_records(self, records: List[dict]) -> Dict[str, int]:
        col = self.db[MONGO_COLLECTIONS["who_raw"]]
        inserted, updated, skipped = 0, 0, 0
        for rec in records:
            if not rec.get("country_code") or rec.get("value") is None:
                skipped += 1
                continue
            filter_doc = {
                "indicator_code": rec["indicator_code"],
                "country_code":   rec["country_code"],
                "year":           rec["year"],
            }
            try:
                result = col.update_one(filter_doc, {"$set": rec}, upsert=True)
                if result.upserted_id:
                    inserted += 1
                else:
                    updated += 1
            except errors.DuplicateKeyError:
                skipped += 1
        logger.info(f"WHO → MongoDB  | inserted={inserted} updated={updated} skipped={skipped}")
        return {"inserted": inserted, "updated": updated, "skipped": skipped}

    def upsert_worldbank_records(self, records: List[dict]) -> Dict[str, int]:
        col = self.db[MONGO_COLLECTIONS["worldbank_raw"]]
        inserted, updated, skipped = 0, 0, 0
        for rec in records:
            if not rec.get("country_code") or rec.get("value") is None:
                skipped += 1
                continue
            filter_doc = {
                "indicator_code": rec["indicator_code"],
                "country_code":   rec["country_code"],
                "year":           rec["year"],
            }
            try:
                result = col.update_one(filter_doc, {"$set": rec}, upsert=True)
                if result.upserted_id:
                    inserted += 1
                else:
                    updated += 1
            except errors.DuplicateKeyError:
                skipped += 1
        logger.info(f"WorldBank → MongoDB | inserted={inserted} updated={updated} skipped={skipped}")
        return {"inserted": inserted, "updated": updated, "skipped": skipped}

    def upsert_disease_records(self, records: List[dict]) -> Dict[str, int]:
        col = self.db[MONGO_COLLECTIONS["disease_raw"]]
        inserted, updated, skipped = 0, 0, 0
        for rec in records:
            if not rec.get("country_code"):
                skipped += 1
                continue
            filter_doc = {"country_code": rec["country_code"]}
            try:
                result = col.update_one(filter_doc, {"$set": rec}, upsert=True)
                if result.upserted_id:
                    inserted += 1
                else:
                    updated += 1
            except errors.DuplicateKeyError:
                skipped += 1
        logger.info(f"Disease → MongoDB  | inserted={inserted} updated={updated} skipped={skipped}")
        return {"inserted": inserted, "updated": updated, "skipped": skipped}

    # ── Read helpers ──────────────────────────────────────────────────────────

    def get_who_dataframe(self) -> "pd.DataFrame":
        import pandas as pd
        col = self.db[MONGO_COLLECTIONS["who_raw"]]
        docs = list(col.find({}, {"_id": 0, "_raw": 0}))
        return pd.DataFrame(docs)

    def get_worldbank_dataframe(self) -> "pd.DataFrame":
        import pandas as pd
        col = self.db[MONGO_COLLECTIONS["worldbank_raw"]]
        docs = list(col.find({}, {"_id": 0, "_raw": 0}))
        return pd.DataFrame(docs)

    def get_disease_dataframe(self) -> "pd.DataFrame":
        import pandas as pd
        col = self.db[MONGO_COLLECTIONS["disease_raw"]]
        docs = list(col.find({}, {"_id": 0, "_raw": 0}))
        return pd.DataFrame(docs)

    def get_collection_stats(self) -> Dict[str, int]:
        return {
            name: self.db[col].count_documents({})
            for name, col in MONGO_COLLECTIONS.items()
        }

    def close(self):
        self.client.close()
        logger.info("MongoDB connection closed")
