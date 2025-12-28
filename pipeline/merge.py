from __future__ import annotations

import logging
from typing import Iterable, List

import pandas as pd

logger = logging.getLogger(__name__)


def drop_empty_frames(frames: Iterable[pd.DataFrame]) -> List[pd.DataFrame]:
    return [df for df in frames if df is not None and not df.empty]


def merge_and_deduplicate(*frames: pd.DataFrame) -> pd.DataFrame:
    valid_frames = drop_empty_frames(frames)
    if not valid_frames:
        logger.warning("No data frames provided to merge; returning empty DataFrame")
        return pd.DataFrame()

    merged = pd.concat(valid_frames, ignore_index=True)
    if {"tender_title", "reference_no"} <= set(merged.columns):
        before = len(merged)
        merged = merged.drop_duplicates(subset=["tender_title", "reference_no"], keep="first")
        logger.info("[MERGE] Duplicate rows removed: %s", before - len(merged))

    return merged
