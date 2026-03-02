"""
Shared entry filters used by both live execution and backtesting.
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Dict, List

import pandas as pd

import config
from logger import get_logger

log = get_logger("entry_filters")


class EarningsCalendar:
    """Loads an optional earnings calendar and applies blackout checks."""

    def __init__(self) -> None:
        self._events: Dict[str, List[dt.date]] = {}
        self._enabled = bool(config.EARNINGS_FILTER_ENABLED)
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        if not self._enabled:
            return

        path = config.EARNINGS_CALENDAR_CSV
        if not path or not os.path.exists(path):
            log.info("Earnings filter enabled but calendar missing: %s", path)
            return

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            log.warning("Failed to read earnings calendar %s: %s", path, exc)
            return

        if df is None or df.empty:
            return

        col_map = {str(col).strip().lower(): col for col in df.columns}
        symbol_col = col_map.get("symbol")
        date_col = col_map.get("date")
        if symbol_col is None or date_col is None:
            log.warning("Earnings calendar must include 'symbol' and 'date' columns: %s", path)
            return

        df = df[[symbol_col, date_col]].copy()
        df[symbol_col] = df[symbol_col].astype(str).str.upper().str.strip()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
        df = df.dropna(subset=[symbol_col, date_col])

        for symbol, group in df.groupby(symbol_col):
            dates = sorted(set(group[date_col].tolist()))
            if dates:
                self._events[symbol] = dates

        log.info("Loaded earnings calendar: %d symbols from %s", len(self._events), path)

    def is_blocked(self, symbol: str, date: dt.date) -> bool:
        """Return True when date is inside blackout window around earnings."""
        self._load()
        if not self._enabled:
            return False

        events = self._events.get(symbol.upper())
        if not events:
            return False

        before = dt.timedelta(days=max(0, int(config.EARNINGS_BLACKOUT_DAYS_BEFORE)))
        after = dt.timedelta(days=max(0, int(config.EARNINGS_BLACKOUT_DAYS_AFTER)))

        for event_date in events:
            if event_date - before <= date <= event_date + after:
                return True
            if event_date > date + after:
                break

        return False
