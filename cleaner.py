from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd
from deep_translator import GoogleTranslator

from config import LATEST_CLEAN_PATH, timestamped_filename


ARABIC_RANGE = ("\u0600", "\u06FF")
FINAL_COLUMNS: Iterable[str] = (
    "reference_no",
    "tender_title",
    "buyer_name",
    "description",
    "publication_date",
    "closing_date",
    "category",
    "currency",
    "source",
    "link",
)


@lru_cache(maxsize=1)
def translator() -> GoogleTranslator:
    return GoogleTranslator(source="auto", target="en")


def translate_to_english(text):
    if pd.isna(text) or not isinstance(text, str):
        return text
    try:
        if any(ARABIC_RANGE[0] <= char <= ARABIC_RANGE[1] for char in text):
            return translator().translate(text)
    except Exception:
        pass
    return text


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["reference_no"] = df.get("TenderNumber").fillna(df.get("Project Code"))
    df["tender_title"] = df.get("TenderName").fillna(df.get("Project Title"))
    df["buyer_name"] = (
        df.get("EntityName")
        .fillna(df.get("Buyer Organisation"))
        .fillna(df.get("Buyer Organisation.1"))
    )
    df["description"] = df.get("TenderDetails").fillna(df.get("Description"))
    df["publication_date"] = df.get("BiddingOpenDate").fillna(df.get("Publication Date"))
    df["closing_date"] = df.get("DueDate").fillna(df.get("Closing date/time"))
    df["category"] = (
        df.get("CategoryDescriptionEn")
        .fillna(df.get("Supply Category"))
        .fillna(df.get("Project Categories"))
    )
    df["currency"] = df.get("Response Currency")
    df["source"] = df.get("Source")
    df["link"] = df.get("TenderURL")
    return df


def clean_tenders(input_file: Path) -> pd.DataFrame:
    df = pd.read_csv(input_file, dtype=str)
    df = normalise_columns(df)
    for col in ("tender_title", "description", "category"):
        df[col] = df[col].apply(translate_to_english)

    df = df.drop_duplicates(subset=["reference_no", "tender_title"], keep="first")
    df = df[list(FINAL_COLUMNS)]
    return df


def persist_cleaned(df: pd.DataFrame) -> Path:
    output_path = timestamped_filename("tenders_clean")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    df.to_csv(LATEST_CLEAN_PATH, index=False, encoding="utf-8-sig")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean merged tender CSV")
    parser.add_argument("input", type=Path, help="Path to merged tender CSV")
    parser.add_argument("--print-path", action="store_true", help="Print the saved CSV path")
    args = parser.parse_args()

    cleaned_df = clean_tenders(args.input)
    saved_path = persist_cleaned(cleaned_df)
    if args.print_path:
        print(saved_path)


if __name__ == "__main__":
    main()
