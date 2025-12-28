from __future__ import annotations

import os
from typing import Any
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _get_setting(name: str, default: Any = None) -> Any:
    value = os.getenv(name)
    if value is not None:
        return value
    try:
        import streamlit as st  # type: ignore

        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # noqa: BLE001
        pass
    return default


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(_get_setting("DATA_DIR", BASE_DIR / "data"))
RAW_DATA_DIR = Path(_get_setting("RAW_DATA_DIR", DATA_DIR / "raw"))
PROCESSED_DATA_DIR = Path(_get_setting("PROCESSED_DATA_DIR", DATA_DIR / "processed"))
ARCHIVE_DATA_DIR = Path(_get_setting("ARCHIVE_DATA_DIR", DATA_DIR / "archive"))

LMSTUDIO_URL = _get_setting("LMSTUDIO_URL", "http://localhost:1234/v1/chat/completions")
LMSTUDIO_MODEL = _get_setting("LMSTUDIO_MODEL", "google/gemma-3-1b")
EMBEDDING_MODEL = _get_setting("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L12-v2")

AI_CACHE_PATH = Path(_get_setting("AI_CACHE_PATH", PROCESSED_DATA_DIR / "ai_cache.json"))
VECTOR_DB_PATH = Path(_get_setting("VECTOR_DB_PATH", PROCESSED_DATA_DIR / "vector_store"))

SMTP_HOST = _get_setting("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(_get_setting("SMTP_PORT", 587))
SMTP_USERNAME = _get_setting("SMTP_USERNAME")
SMTP_PASSWORD = _get_setting("SMTP_PASSWORD")
SMTP_USE_TLS = str(_get_setting("SMTP_USE_TLS", "true")).lower() != "false"
SMTP_FROM_ADDRESS = _get_setting("SMTP_FROM_ADDRESS", SMTP_USERNAME or "Tender Alerts <no-reply@tenders.local>")

DEFAULT_NOTIFICATION_RECIPIENTS = [
    email.strip()
    for email in str(_get_setting("NOTIFY_EMAILS", "")).split(",")
    if email.strip()
]
DEFAULT_NOTIFICATION_SUBJECT = _get_setting("NOTIFY_SUBJECT", "Top Priority UAE Tenders")
DEFAULT_NOTIFICATION_TIME = _get_setting("NOTIFY_TIME", "17:00")
DEFAULT_NOTIFICATION_TIMEZONE = _get_setting("NOTIFY_TIMEZONE", "Asia/Dubai")
DEFAULT_NOTIFICATION_KEYWORDS = _get_setting("NOTIFY_KEYWORDS", "ai, software, data")
NOTIFICATIONS_CONFIG_PATH = Path(
    _get_setting("NOTIFICATIONS_CONFIG_PATH", PROCESSED_DATA_DIR / "notifications.json")
)
NOTIFICATION_LOG_PATH = Path(
    _get_setting("NOTIFICATION_LOG_PATH", PROCESSED_DATA_DIR / "notifications.log")
)

# Ensure directories exist at import time for smoother DX
for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, ARCHIVE_DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def timestamped_filename(name: str, suffix: str = "csv", prefix: Optional[str] = None) -> Path:
    """Generate a timestamped filename within the processed directory."""
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    parts = [p for p in [prefix, name, stamp] if p]
    filename = "_".join(parts)
    if suffix:
        filename = f"{filename}.{suffix.lstrip('.')}"
    return PROCESSED_DATA_DIR / filename


LATEST_COMBINED_PATH = PROCESSED_DATA_DIR / "tenders_combined_latest.csv"
LATEST_CLEAN_PATH = PROCESSED_DATA_DIR / "tenders_clean_latest.csv"
FALLBACK_CLEAN_PATH = Path(_get_setting("FALLBACK_CLEAN_PATH", BASE_DIR / "tenders_clean.csv"))


# Raw artefacts naming helpers

def raw_export_path(source: str, extension: str = "csv") -> Path:
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    filename = f"{source}_{stamp}.{extension.lstrip('.')}"
    return RAW_DATA_DIR / filename
