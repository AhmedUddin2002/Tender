from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

from ai_utils import get_ai_score_and_summary
from config import LATEST_CLEAN_PATH
from run_pipeline import run_pipeline
from vectorstore import VectorResult, semantic_search


FALLBACK_DATA_PATH = Path("tenders_clean.csv")


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
