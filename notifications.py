from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, time
from pathlib import Path
from typing import List, Tuple

from email_validator import EmailNotValidError, validate_email

from config import (
    DEFAULT_NOTIFICATION_KEYWORDS,
    DEFAULT_NOTIFICATION_RECIPIENTS,
    DEFAULT_NOTIFICATION_SUBJECT,
    DEFAULT_NOTIFICATION_TIME,
    DEFAULT_NOTIFICATION_TIMEZONE,
    NOTIFICATION_LOG_PATH,
    NOTIFICATIONS_CONFIG_PATH,
)


@dataclass
class NotificationSettings:
    recipients: List[str]
    cc: List[str]
    subject: str
    send_time: str
    timezone: str
    enabled: bool
    keywords: str

    @classmethod
    def default(cls) -> "NotificationSettings":
        return cls(
            recipients=list(DEFAULT_NOTIFICATION_RECIPIENTS or []),
            cc=[],
            subject=DEFAULT_NOTIFICATION_SUBJECT,
            send_time=DEFAULT_NOTIFICATION_TIME,
            timezone=DEFAULT_NOTIFICATION_TIMEZONE,
            enabled=bool(DEFAULT_NOTIFICATION_RECIPIENTS),
            keywords=DEFAULT_NOTIFICATION_KEYWORDS,
        )

    @classmethod
    def from_dict(cls, payload: dict | None) -> "NotificationSettings":
        if not payload:
            return cls.default()
        defaults = cls.default()
        return cls(
            recipients=payload.get("recipients", defaults.recipients),
            cc=payload.get("cc", defaults.cc),
            subject=payload.get("subject", defaults.subject),
            send_time=payload.get("send_time", defaults.send_time),
            timezone=payload.get("timezone", defaults.timezone),
            enabled=payload.get("enabled", defaults.enabled),
            keywords=payload.get("keywords", defaults.keywords),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def time_object(self) -> time:
        hour, minute = parse_time_string(self.send_time)
        return time(hour, minute)


def parse_time_string(value: str) -> Tuple[int, int]:
    parts = value.strip().split(":")
    if len(parts) != 2:
        raise ValueError("Time must be in HH:MM format")
    hour, minute = (int(part) for part in parts)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("Time values out of range")
    return hour, minute


def parse_recipient_input(raw: str) -> List[str]:
    tokens = re.split(r"[;,\n]+", raw)
    return [token.strip() for token in tokens if token.strip()]


def validate_addresses(addresses: List[str]) -> Tuple[List[str], List[str]]:
    valid: List[str] = []
    errors: List[str] = []
    for address in addresses:
        try:
            valid.append(validate_email(address, check_deliverability=False).email)
        except EmailNotValidError as exc:
            errors.append(f"{address}: {exc}")
    return valid, errors


def load_notification_settings(path: Path = NOTIFICATIONS_CONFIG_PATH) -> NotificationSettings:
    if path.exists():
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            payload = None
    else:
        payload = None
    return NotificationSettings.from_dict(payload)


def save_notification_settings(settings: NotificationSettings, path: Path = NOTIFICATIONS_CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings.to_dict(), indent=2))


def append_notification_log(message: str, success: bool, path: Path = NOTIFICATION_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    status = "SUCCESS" if success else "ERROR"
    path.open("a", encoding="utf-8").write(f"{timestamp} [{status}] {message}\n")


def tail_notification_log(limit: int = 5, path: Path = NOTIFICATION_LOG_PATH) -> List[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    return lines
