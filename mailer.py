from __future__ import annotations

import io
import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from config import (
    FALLBACK_CLEAN_PATH,
    LATEST_CLEAN_PATH,
    SMTP_FROM_ADDRESS,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_USE_TLS,
    SMTP_USERNAME,
)
from notifications import NotificationSettings, append_notification_log


@dataclass
class SMTPSettings:
    host: str
    port: int
    username: str | None
    password: str | None
    use_tls: bool
    from_address: str


def get_smtp_settings() -> SMTPSettings:
    return SMTPSettings(
        host=SMTP_HOST,
        port=SMTP_PORT,
        username=SMTP_USERNAME,
        password=SMTP_PASSWORD,
        use_tls=SMTP_USE_TLS,
        from_address=SMTP_FROM_ADDRESS,
    )


def resolve_clean_dataset_path() -> Path:
    for candidate in (LATEST_CLEAN_PATH, FALLBACK_CLEAN_PATH):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError("No cleaned dataset found. Run the pipeline first.")


def load_clean_dataframe() -> pd.DataFrame:
    dataset_path = resolve_clean_dataset_path()
    return pd.read_csv(dataset_path)


def _ensure_priority_column(df: pd.DataFrame) -> pd.DataFrame:
    if "priority" in df.columns:
        return df
    working = df.copy()
    if "ai_score" in working.columns:
        working["ai_score"] = pd.to_numeric(working["ai_score"], errors="coerce").fillna(0).astype(int)
        working["priority"] = pd.cut(
            working["ai_score"],
            bins=[-1, 39, 69, 100],
            labels=["LOW", "MEDIUM", "HIGH"],
        ).astype(str)
    else:
        working["priority"] = "UNKNOWN"
    return working


def filter_priority_tenders(
    df: pd.DataFrame,
    keywords: str,
    limit: int | None = 20,
) -> pd.DataFrame:
    working = _ensure_priority_column(df)
    priority_df = working[working["priority"] == "HIGH"].copy()
    if priority_df.empty:
        return priority_df

    if keywords:
        tokens = [token.strip() for token in keywords.split(",") if token.strip()]
        if tokens:
            import re

            pattern = "|".join(re.escape(token) for token in tokens)
            search_columns: List[str] = [
                col
                for col in ("tender_title", "description", "ai_summary", "category")
                if col in priority_df.columns
            ]
            if search_columns:
                mask = priority_df[search_columns].fillna("").apply(
                    lambda col: col.str.contains(pattern, case=False, regex=True)
                ).any(axis=1)
                priority_df = priority_df[mask]

    priority_df = priority_df.sort_values(
        ["ai_score", "closing_date"] if "closing_date" in priority_df.columns else ["ai_score"],
        ascending=[False, True] if "closing_date" in priority_df.columns else [False],
    )

    if limit is not None and limit > 0:
        priority_df = priority_df.head(limit)
    return priority_df


def _format_link_cell(url: str | float | None) -> str:
    if not url or not isinstance(url, str):
        return ""
    return f'<a href="{url}" target="_blank">View Tender</a>'


def _render_html_table(df: pd.DataFrame) -> str:
    display_columns = [
        col
        for col in (
            "tender_title",
            "reference_no",
            "closing_date",
            "buyer_name",
            "category",
            "ai_score",
            "ai_summary",
            "link",
        )
        if col in df.columns
    ]
    if not display_columns:
        display_columns = list(df.columns)

    table_df = df[display_columns].copy()
    if "link" in table_df.columns:
        table_df["link"] = table_df["link"].apply(_format_link_cell)
    table_df = table_df.fillna("")
    return table_df.to_html(index=False, escape=False, justify="left")


def _build_email_body(df: pd.DataFrame, settings: NotificationSettings) -> str:
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    intro = (
        f"<p>Here are the current high-priority UAE tenders that match your keywords: <strong>{settings.keywords}</strong>.</p>"
        if settings.keywords
        else "<p>Here are the current high-priority UAE tenders.</p>"
    )
    table_html = _render_html_table(df)
    outro = (
        "<p>You are receiving this because daily notifications are enabled. "
        "Manage your preferences anytime from the Streamlit dashboard.</p>"
    )
    return f"{intro}{table_html}<p>Generated at {generated_at}.</p>{outro}"


def _build_csv_attachment(df: pd.DataFrame) -> Tuple[bytes, str]:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, encoding="utf-8-sig")
    csv_bytes = buffer.getvalue().encode("utf-8-sig")
    filename = f"uae_tenders_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    return csv_bytes, filename


def send_priority_digest(
    df: pd.DataFrame,
    settings: NotificationSettings,
    smtp_settings: SMTPSettings | None = None,
    limit: int | None = 20,
) -> Tuple[bool, str]:
    if smtp_settings is None:
        smtp_settings = get_smtp_settings()

    if not settings.recipients:
        return False, "No recipients configured for notifications."

    filtered_df = filter_priority_tenders(df, settings.keywords, limit=limit)
    if filtered_df.empty:
        return False, "No high-priority tenders matched the criteria."

    msg = MIMEMultipart()
    msg["Subject"] = settings.subject
    msg["From"] = smtp_settings.from_address
    msg["To"] = ", ".join(settings.recipients)
    if settings.cc:
        msg["Cc"] = ", ".join(settings.cc)

    body_html = _build_email_body(filtered_df, settings)
    msg.attach(MIMEText(body_html, "html", "utf-8"))

    attachment_bytes, filename = _build_csv_attachment(filtered_df)
    attachment = MIMEApplication(attachment_bytes, Name=filename)
    attachment["Content-Disposition"] = f"attachment; filename=\"{filename}\""
    msg.attach(attachment)

    recipients = settings.recipients + settings.cc

    try:
        with smtplib.SMTP(smtp_settings.host, smtp_settings.port, timeout=30) as server:
            if smtp_settings.use_tls:
                server.starttls()
            if smtp_settings.username and smtp_settings.password:
                server.login(smtp_settings.username, smtp_settings.password)
            server.sendmail(smtp_settings.from_address, recipients, msg.as_string())
    except Exception as exc:
        return False, f"Email send failed: {exc}"

    return True, f"Sent {len(filtered_df)} high-priority tenders to {len(settings.recipients)} recipient(s)."


def run_scheduled_digest(settings_dict: dict) -> None:
    settings = NotificationSettings.from_dict(settings_dict)
    smtp_settings = get_smtp_settings()
    try:
        df = load_clean_dataframe()
        success, message = send_priority_digest(df, settings, smtp_settings)
    except Exception as exc:  # noqa: BLE001
        success = False
        message = f"Scheduled send failed: {exc}"
    append_notification_log(message, success)
