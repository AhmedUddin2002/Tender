from __future__ import annotations

import atexit
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import streamlit as st

from apscheduler.jobstores.base import JobLookupError
from apscheduler.schedulers.background import BackgroundScheduler

from ai_utils import get_ai_score_and_summary
from config import LATEST_CLEAN_PATH, PLAYWRIGHT_BROWSERS_PATH
from mailer import run_scheduled_digest, send_priority_digest
from notifications import (
    NotificationSettings,
    append_notification_log,
    load_notification_settings,
    parse_recipient_input,
    parse_time_string,
    save_notification_settings,
    tail_notification_log,
    validate_addresses,
)
from run_pipeline import run_pipeline
from vectorstore import VectorResult, semantic_search


FALLBACK_DATA_PATH = Path("tenders_clean.csv")
SCHEDULER_JOB_ID = "daily_notification_digest"


def ensure_playwright_browsers() -> None:
    target_dir = PLAYWRIGHT_BROWSERS_PATH or Path(os.getenv("PLAYWRIGHT_BROWSERS_PATH", "/mount/tmp/playwright"))
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(target_dir))

    chromium_executable = target_dir / "chromium-1200" / "chrome-headless-shell" / "chrome-headless-shell"
    if chromium_executable.exists():
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["playwright", "install", "chromium"],
            check=True,
            env={**os.environ, "PLAYWRIGHT_BROWSERS_PATH": str(target_dir)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:  # noqa: BLE001
        st.warning(
            "Playwright browser installation failed; Dubai scraping may not work in this environment."
        )


ensure_playwright_browsers()


st.set_page_config(page_title="UAE Tender AI Dashboard", page_icon="🇦🇪", layout="wide")
st.title("🇦🇪 UAE Government Tender AI Dashboard")


def locate_dataset() -> Optional[Path]:
    session_path = st.session_state.get("data_path")
    if session_path:
        candidate = Path(session_path)
        if candidate.exists():
            return candidate
    for candidate in (LATEST_CLEAN_PATH, FALLBACK_DATA_PATH):
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path
    return None


def _shutdown_scheduler() -> None:
    scheduler: BackgroundScheduler | None = st.session_state.get("_notification_scheduler")
    if scheduler and scheduler.running:
        scheduler.shutdown(wait=False)


def get_scheduler() -> BackgroundScheduler:
    scheduler: BackgroundScheduler | None = st.session_state.get("_notification_scheduler")
    if scheduler and scheduler.running:
        return scheduler

    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.start()
    st.session_state["_notification_scheduler"] = scheduler

    if not st.session_state.get("_scheduler_shutdown_registered"):
        atexit.register(_shutdown_scheduler)
        st.session_state["_scheduler_shutdown_registered"] = True

    return scheduler


def get_notification_settings() -> NotificationSettings:
    if "notification_settings" not in st.session_state:
        st.session_state["notification_settings"] = load_notification_settings().to_dict()
    return NotificationSettings.from_dict(st.session_state["notification_settings"])


def update_notification_settings(settings: NotificationSettings) -> None:
    st.session_state["notification_settings"] = settings.to_dict()


def ensure_scheduler_job(settings: NotificationSettings) -> Tuple[bool, Optional[str], Optional[datetime]]:
    scheduler = get_scheduler()

    signature = (
        tuple(settings.recipients),
        tuple(settings.cc),
        settings.subject,
        settings.send_time,
        settings.timezone,
        settings.enabled,
        settings.keywords,
    )

    previous_signature = st.session_state.get("_scheduler_signature")
    existing_job = scheduler.get_job(SCHEDULER_JOB_ID)
    needs_sync = (
        previous_signature != signature
        or (settings.enabled and existing_job is None)
        or (not settings.enabled and existing_job is not None)
    )

    if not needs_sync:
        next_run = existing_job.next_run_time if existing_job else None
        return True, None, next_run

    try:
        scheduler.remove_job(SCHEDULER_JOB_ID)
    except JobLookupError:
        pass

    if not settings.enabled:
        st.session_state["_scheduler_signature"] = signature
        return True, "Notifications disabled; schedule cleared.", None

    try:
        hour, minute = parse_time_string(settings.send_time)
    except ValueError as exc:
        return False, f"Invalid send time: {exc}", None

    try:
        timezone = ZoneInfo(settings.timezone)
    except ZoneInfoNotFoundError:
        return False, f"Unknown timezone: {settings.timezone}", None

    try:
        job = scheduler.add_job(
            run_scheduled_digest,
            trigger="cron",
            id=SCHEDULER_JOB_ID,
            hour=hour,
            minute=minute,
            timezone=timezone,
            replace_existing=True,
            kwargs={"settings_dict": settings.to_dict()},
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"Failed to schedule job: {exc}", None

    st.session_state["_scheduler_signature"] = signature
    return True, None, job.next_run_time


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def ensure_ai_metadata(df: pd.DataFrame, keywords: str) -> bool:
    changed = False
    for column in ("ai_score", "ai_summary", "ai_keywords"):
        if column not in df.columns:
            df[column] = pd.NA
            changed = True

    for idx, row in df.iterrows():
        ai_score = row.get("ai_score")
        stored_keywords = row.get("ai_keywords")
        keyword_mismatch = pd.isna(stored_keywords) or str(stored_keywords) != keywords
        needs_refresh = pd.isna(ai_score) or keyword_mismatch
        if needs_refresh:
            score, summary = get_ai_score_and_summary(
                row.get("tender_title") or row.get("title") or "",
                row.get("description", ""),
                keywords,
            )
            df.at[idx, "ai_score"] = score
            df.at[idx, "ai_summary"] = summary
            df.at[idx, "ai_keywords"] = keywords
            changed = True

    df["ai_score"] = pd.to_numeric(df["ai_score"], errors="coerce").fillna(0).astype(int)
    df["ai_summary"] = df["ai_summary"].fillna("")
    df["priority"] = pd.cut(
        df["ai_score"],
        bins=[-1, 39, 69, 100],
        labels=["LOW", "MEDIUM", "HIGH"],
    ).astype(str)
    return changed


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    export_df = df.copy()
    for column in ("publication_date", "closing_date"):
        if column in export_df.columns:
            export_df[column] = export_df[column].dt.strftime("%Y-%m-%d")
    export_df.to_csv(path, index=False)


def display_metrics(df: pd.DataFrame) -> None:
    now = datetime.utcnow()
    soon_threshold = now + timedelta(days=7)

    if "closing_date" in df.columns:
        closing_series = pd.to_datetime(df["closing_date"], errors="coerce")
        try:
            closing_series = closing_series.dt.tz_localize(None)
        except TypeError:
            pass
        closing_soon = closing_series.between(now, soon_threshold).sum()
    else:
        closing_soon = 0
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tracked tenders", f"{len(df):,}")
    col2.metric("High priority", f"{(df['priority'] == 'HIGH').sum():,}")
    col3.metric("Median AI score", f"{int(df['ai_score'].median()):d}")
    col4.metric("Closing ≤ 7 days", f"{closing_soon:,}")


def apply_filters(
    df: pd.DataFrame,
    selected_sources: List[str],
    selected_categories: List[str],
    selected_priorities: List[str],
    publication_range: Tuple[datetime, datetime] | Tuple[None, None],
    min_score: int,
) -> pd.DataFrame:
    filtered = df.copy()
    if selected_sources:
        filtered = filtered[filtered["source"].isin(selected_sources)]
    if selected_categories:
        filtered = filtered[filtered["category"].isin(selected_categories)]
    if selected_priorities:
        filtered = filtered[filtered["priority"].isin(selected_priorities)]
    if publication_range and all(publication_range):
        start_date, end_date = publication_range
        filtered = filtered[
            filtered["publication_date"].dt.date.between(start_date, end_date)
        ]
    filtered = filtered[filtered["ai_score"] >= min_score]
    return filtered


def format_dates(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for column in ("publication_date", "closing_date"):
        if column in formatted.columns:
            formatted[column] = formatted[column].dt.strftime("%Y-%m-%d")
    return formatted


def render_semantic_results(results: List[VectorResult], data: pd.DataFrame) -> None:
    if not results:
        st.info("No semantic matches were found. Try expanding your description.")
        return

    lookup = (
        data.sort_values("ai_score", ascending=False)
        .drop_duplicates(subset=["reference_no"], keep="first")
        .set_index("reference_no")
    )

    rows = []
    for item in results:
        base = {
            "Similarity": f"{item.score * 100:.1f}%",
            "Reference": item.reference_no,
            "Title": item.tender_title,
            "Source": item.source,
            "Category": item.category,
        }
        if item.reference_no in lookup.index:
            match = lookup.loc[item.reference_no]
            base.update(
                {
                    "AI score": int(match["ai_score"]),
                    "Priority": match["priority"],
                    "Closing date": match["closing_date"],
                    "Summary": match.get("ai_summary", ""),
                }
            )
        rows.append(base)

    results_df = pd.DataFrame(rows)
    st.dataframe(results_df, use_container_width=True)


with st.sidebar:
    st.header("AI controls")
    keywords = st.text_input(
        "🔑 Relevance keywords",
        value=st.session_state.get("keywords", "ai, software, data"),
        help="Used to guide the AI relevance scoring.",
    )
    st.session_state["keywords"] = keywords
    headless = st.checkbox("Run scraper headless", value=True)
    if st.button("🔄 Run full pipeline", use_container_width=True):
        with st.spinner("Scraping, cleaning, embedding..."):
            output_path = run_pipeline(headless=headless)
        st.session_state["data_path"] = str(output_path)
        st.cache_data.clear()
        st.success("Pipeline completed")


data_path = locate_dataset()
if not data_path:
    st.warning("No tender data found. Run the pipeline to populate the dataset.")
    st.stop()

raw_df = load_data(str(data_path)).copy()

for column in ("publication_date", "closing_date"):
    if column in raw_df.columns:
        raw_df[column] = pd.to_datetime(raw_df[column], errors="coerce")

ai_updated = ensure_ai_metadata(raw_df, keywords)
if ai_updated:
    save_dataframe(raw_df, data_path)

with st.sidebar:
    st.header("Filters")
    source_options = sorted(raw_df["source"].dropna().unique().tolist())
    selected_sources = st.multiselect("Source", source_options, default=source_options)

    category_options = sorted(raw_df["category"].dropna().unique().tolist())
    selected_categories = st.multiselect("Category", category_options, default=category_options)

    priority_options = ["HIGH", "MEDIUM", "LOW"]
    selected_priorities = st.multiselect("Priority", priority_options, default=priority_options)

    publication_dates = raw_df["publication_date"].dropna()
    if not publication_dates.empty:
        default_start = publication_dates.min().date()
        default_end = publication_dates.max().date()
        publication_range = st.date_input(
            "Publication range",
            value=(default_start, default_end),
            help="Filter tenders by publication date.",
        )
    else:
        publication_range = (None, None)

    min_score = st.slider("Minimum AI score", min_value=0, max_value=100, value=0, step=5)


filtered_df = apply_filters(
    raw_df,
    selected_sources,
    selected_categories,
    selected_priorities,
    publication_range if isinstance(publication_range, tuple) else (None, None),
    min_score,
)

display_metrics(filtered_df)

st.subheader("🔴 High priority tenders")
high_priority = filtered_df[filtered_df["priority"] == "HIGH"].sort_values("ai_score", ascending=False)
st.dataframe(format_dates(high_priority), use_container_width=True)

st.subheader("🟢 All filtered tenders")
sorted_df = filtered_df.sort_values(["ai_score", "closing_date"], ascending=[False, True])
st.dataframe(
    format_dates(sorted_df),
    use_container_width=True,
)

csv_export = format_dates(sorted_df).to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "⬇️ Download filtered tenders",
    data=csv_export,
    file_name=f"uae_tenders_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv",
)

st.subheader("🧠 Semantic search")
semantic_col, k_col = st.columns([4, 1])
with semantic_col:
    semantic_query = st.text_input(
        "Describe the tenders you're looking for",
        placeholder="e.g. Cloud migration services for government agencies",
    )
with k_col:
    top_k = st.slider("Results", min_value=3, max_value=20, value=5)

if semantic_query:
    with st.spinner("Searching vector database..."):
        try:
            semantic_results = semantic_search(semantic_query, top_k=top_k)
        except Exception as exc:
            st.error(f"Semantic search failed: {exc}")
        else:
            render_semantic_results(semantic_results, raw_df)

data_timestamp = datetime.fromtimestamp(data_path.stat().st_mtime)
st.caption(f"Last updated: {data_timestamp.strftime('%Y-%m-%d %H:%M:%S')} | Data source: {data_path}")


st.divider()
st.header("📧 Email notifications")

settings = get_notification_settings()

with st.form("notification-settings"):
    st.markdown("### Daily digest settings")
    recipients_input = st.text_area(
        "Primary recipients",
        value="\n".join(settings.recipients),
        help="Use commas, semicolons, or new lines to separate addresses.",
    )
    cc_input = st.text_area(
        "CC recipients",
        value="\n".join(settings.cc),
        help="Optional CC list.",
    )
    subject_input = st.text_input("Email subject", value=settings.subject)
    keywords_input = st.text_input(
        "Filter keywords",
        value=settings.keywords,
        help="Used to filter the high-priority tenders included in the digest.",
    )
    time_col, tz_col, enabled_col = st.columns([1, 2, 1])
    with time_col:
        send_time_input = st.text_input("Send time (HH:MM)", value=settings.send_time)
    with tz_col:
        timezone_input = st.text_input("Timezone", value=settings.timezone)
    with enabled_col:
        enabled_input = st.checkbox("Enable daily digest", value=settings.enabled)

    submitted = st.form_submit_button("💾 Save settings")

    if submitted:
        primary_recipients = parse_recipient_input(recipients_input)
        cc_recipients = parse_recipient_input(cc_input)
        valid_primary, primary_errors = validate_addresses(primary_recipients)
        valid_cc, cc_errors = validate_addresses(cc_recipients)

        error_messages: List[str] = []
        if primary_errors or cc_errors:
            error_messages.extend(primary_errors + cc_errors)
        if enabled_input and not valid_primary:
            error_messages.append(
                "At least one valid primary recipient is required when notifications are enabled."
            )

        try:
            parse_time_string(send_time_input)
        except ValueError as exc:
            error_messages.append(f"Invalid send time: {exc}")

        try:
            ZoneInfo(timezone_input)
        except ZoneInfoNotFoundError:
            error_messages.append(f"Unknown timezone: {timezone_input}")

        if error_messages:
            st.error("\n".join(error_messages))
        else:
            new_settings = NotificationSettings(
                recipients=valid_primary,
                cc=valid_cc,
                subject=subject_input,
                send_time=send_time_input,
                timezone=timezone_input,
                enabled=enabled_input,
                keywords=keywords_input,
            )
            save_notification_settings(new_settings)
            update_notification_settings(new_settings)
            ok, message, next_run = ensure_scheduler_job(new_settings)
            if ok:
                text = "Settings saved."
                if message:
                    text += f" {message}"
                if next_run:
                    text += f" Next email scheduled for {next_run.strftime('%Y-%m-%d %H:%M %Z')}."
                st.success(text)
            else:
                st.error(message or "Failed to update scheduler.")


manual_col, status_col = st.columns([1, 1])
with manual_col:
    if st.button("✉️ Send high-priority digest now", use_container_width=True):
        current_settings = get_notification_settings()
        try:
            success, message = send_priority_digest(raw_df, current_settings)
        except Exception as exc:  # noqa: BLE001
            success = False
            message = f"Manual send failed: {exc}"
        append_notification_log(message, success)
        if success:
            st.success(message)
        else:
            st.error(message)

with status_col:
    st.markdown("#### Recent send status")
    log_lines = tail_notification_log()
    if log_lines:
        st.code("\n".join(log_lines), language="text")
    else:
        st.info("No notification activity logged yet.")

ensure_scheduler_job(settings)
