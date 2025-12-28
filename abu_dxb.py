from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from config import LATEST_COMBINED_PATH, raw_export_path, timestamped_filename
from pipeline.merge import merge_and_deduplicate
from pipeline.sources.abu_dhabi import AbuDhabiScraper
from pipeline.sources.dubai import DubaiScraper


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tender_pipeline")


def persist_dataframe(df: pd.DataFrame, timestamp_path: Path, latest_path: Path) -> None:
    if df.empty:
        logger.warning("Attempted to persist empty DataFrame: %s", timestamp_path.name)
        return
    df.to_csv(timestamp_path, index=False, encoding="utf-8-sig")
    df.to_csv(latest_path, index=False, encoding="utf-8-sig")
    logger.info("Saved %s rows to %s", len(df), timestamp_path)


def pipeline(headless: bool = True) -> Path:
    # Scrape Abu Dhabi
    abu_scraper = AbuDhabiScraper()
    df_abu = abu_scraper.scrape()
    abu_raw_path = raw_export_path("abu_dhabi")
    if not df_abu.empty:
        df_abu.to_csv(abu_raw_path, index=False, encoding="utf-8-sig")
        logger.info("[AD] Raw export saved → %s", abu_raw_path)

    # Scrape Dubai
    dubai_scraper = DubaiScraper(excel_path=raw_export_path("dubai", extension="xls"), headless=headless)
    df_dubai = dubai_scraper.scrape()
    if not df_dubai.empty:
        dubai_raw_csv = raw_export_path("dubai", extension="csv")
        df_dubai.to_csv(dubai_raw_csv, index=False, encoding="utf-8-sig")
        logger.info("[DXB] Raw export saved → %s", dubai_raw_csv)

    combined = merge_and_deduplicate(df_abu, df_dubai)
    combined_path = timestamped_filename("tenders_combined")
    persist_dataframe(combined, combined_path, LATEST_COMBINED_PATH)
    return combined_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape and merge UAE tenders")
    parser.add_argument("--headless", dest="headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--return-path", action="store_true", help="Print the processed CSV path")
    args = parser.parse_args()

    output_path = pipeline(headless=args.headless)
    if args.return_path:
        print(output_path)


if __name__ == "__main__":
    main()
