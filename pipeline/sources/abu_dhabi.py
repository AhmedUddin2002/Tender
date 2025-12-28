from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

AD_LIST_URL = "https://www.adgpg.gov.ae/SCAPI/ADGEs/AlMaqtaa/Tender/List"
AD_DETAILS_URL_TPL = (
    "https://www.adgpg.gov.ae/SCAPI/ADGEs/AlMaqtaa/Tender/Details/{tender_id}?recommendedLimit=0"
)
AD_DEFAULT_LIMIT = 100
AD_DEFAULT_WORKERS = 5
AD_REQUESTS_TIMEOUT = 20
AD_SLEEP_BETWEEN_REQUESTS = 0.2

STATUS_OPEN = "OPEN"


class AbuDhabiScraper:
    def __init__(self, retries: int = 3, backoff_factor: float = 0.5, session: Optional[requests.Session] = None):
        self.session = session or self._build_session(retries=retries, backoff_factor=backoff_factor)

    @staticmethod
    def _build_session(retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=("GET", "POST"),
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/117.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.adgpg.gov.ae/en/For-Suppliers/Public-Tenders",
            }
        )
        return session

    def fetch_list(self, offset: int, limit: int, status: str = STATUS_OPEN) -> dict:
        payload = {
            "status": status,
            "offset": offset,
            "limit": limit,
            "Category": "",
            "Entity": "",
            "Sorting": "LAST_CREATED",
            "DueDate": "",
            "Name": "",
        }
        resp = self.session.post(AD_LIST_URL, data=payload, timeout=AD_REQUESTS_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def fetch_detail(self, tender_id: str) -> Optional[dict]:
        url = AD_DETAILS_URL_TPL.format(tender_id=tender_id)
        try:
            resp = self.session.get(url, timeout=AD_REQUESTS_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("[AD] Failed to fetch details for %s: %s", tender_id, exc)
            return None

    def merge_list_and_detail(self, list_item: dict, detail_resp: Optional[dict]) -> dict:
        record = {}
        for key in (
            "TenderID",
            "TenderName",
            "TenderNumber",
            "EntityName",
            "entityId",
            "InternalStatus",
            "DueDate",
            "TenderDetails",
            "BiddingOpenDate",
            "DueDays",
        ):
            record[key] = list_item.get(key)

        record["TenderURL"] = (
            f"https://www.adgpg.gov.ae/en/For-Suppliers/Public-Tenders?id={record['TenderID']}?recommendedLimit=3"
            if record.get("TenderID")
            else None
        )

        if detail_resp and "TenderDetails" in detail_resp:
            details = detail_resp["TenderDetails"]
            record.update(
                {
                    "EstimatedValue": details.get("EstimatedValue"),
                    "DiscoveryUrl": details.get("DiscoveryUrl"),
                    "EventId": details.get("EventId"),
                    "EventType": details.get("EventType"),
                    "CategoryDescriptionEn": details.get("CategoryDescriptionEn"),
                    "CategoryDescriptionAr": details.get("CategoryDescriptionAr"),
                    "CategoryIds": details.get("categoryId"),
                    "TenderStatus": details.get("TenderStatus"),
                    "Attachments": details.get("Attachments"),
                    "RecommendedTenders": details.get("RecommendedTenders"),
                }
            )
        else:
            for key in (
                "EstimatedValue",
                "DiscoveryUrl",
                "EventId",
                "EventType",
                "CategoryDescriptionEn",
                "CategoryDescriptionAr",
                "CategoryIds",
                "TenderStatus",
                "Attachments",
                "RecommendedTenders",
            ):
                record[key] = None

        return record

    def scrape(self, limit: int = AD_DEFAULT_LIMIT, workers: int = AD_DEFAULT_WORKERS) -> pd.DataFrame:
        logger.info("[AD] Starting Abu Dhabi scraping...")
        all_records = []
        offset = 0
        total_expected = None

        while True:
            try:
                response_json = self.fetch_list(offset=offset, limit=limit)
            except Exception as exc:
                logger.error("[AD] List request failed: %s", exc)
                break

            tender_list: Iterable[dict] = response_json.get("TenderList") or []
            if total_expected is None:
                total_expected = response_json.get("TenderCount")
                logger.info("[AD] Total API count: %s", total_expected)

            if not tender_list:
                break

            page_records = []
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {executor.submit(self.fetch_detail, tender.get("TenderID")): tender for tender in tender_list}
                for future in as_completed(future_map):
                    list_item = future_map[future]
                    detail = None
                    try:
                        detail = future.result()
                    except Exception:
                        logger.exception("[AD] Detail retrieval failed for %s", list_item.get("TenderID"))
                    merged = self.merge_list_and_detail(list_item, detail)
                    page_records.append(merged)
                    time.sleep(AD_SLEEP_BETWEEN_REQUESTS)

            all_records.extend(page_records)
            offset += limit
            if len(tender_list) < limit:
                break

        df = pd.DataFrame(all_records)
        if not df.empty:
            df["Source"] = "AbuDhabi"
            df["tender_title"] = df["TenderName"].astype(str).str.strip()
            df["reference_no"] = df["TenderNumber"].astype(str).str.strip()

        logger.info("[AD] Scraped Abu Dhabi rows: %s", len(df))
        return df
