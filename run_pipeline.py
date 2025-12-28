from __future__ import annotations

from pathlib import Path

from abu_dxb import pipeline as scrape_pipeline
from cleaner import clean_tenders, persist_cleaned
from vectorstore import rebuild_collection


def run_pipeline(headless: bool = True) -> Path:
    combined_path = scrape_pipeline(headless=headless)
    cleaned_df = clean_tenders(combined_path)
    cleaned_path = persist_cleaned(cleaned_df)
    rebuild_collection(cleaned_df)
    return cleaned_path
