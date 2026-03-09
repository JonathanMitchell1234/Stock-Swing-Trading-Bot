"""
Pattern Day Trader (PDT) Guard.

Accounts under $25 000 are limited to 3 day trades in any rolling
5-business-day window.  A *day trade* is opening AND closing (or
closing and opening) the same security on the **same calendar day**.

This module enforces the limit set by MAX_DAY_TRADES_ALLOWED in config.py.
When set to 0 it behaves as a hard lock (original behaviour).
When set to 3 it allows up to 3 day trades per PDT_LOOKBACK_DAYS window.

How it works
------------
1.  Maintains a local JSON ledger of every BUY fill with the fill date,
    plus a list of completed day-trade timestamps.
2.  Before any SELL, checks:
    - Was this position opened today?  If yes, count it as a day trade.
    - Have we already used MAX_DAY_TRADES_ALLOWED day trades this window?
      If yes → block the sell.
3.  Before any BUY, checks whether an open sell order for that symbol
    already exists today (reverse day-trade).
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Dict, List

import config
from logger import get_logger

log = get_logger("pdt_guard")

LEDGER_PATH = Path("logs/pdt_ledger.json")


class PDTGuard:
    """Enforces the PDT day-trade limit."""

    def __init__(self) -> None:
        self._ledger: Dict[str, str] = {}           # symbol → buy_date (ISO str)
        self._buy_times: Dict[str, str] = {}         # symbol → buy datetime ISO str (for min-hold check)
        self._day_trades: List[str] = []             # list of ISO date strings of day trades made
        self._sell_dates: Dict[str, str] = {}        # symbol → last sell_date (ISO str)
        self._load()

    # ── persistence ──────────────────────────────────────────
    def _load(self) -> None:
        if LEDGER_PATH.exists():
            with open(LEDGER_PATH) as f:
                data = json.load(f)
            if isinstance(data, dict) and "ledger" in data:
                self._ledger = data.get("ledger", {})
                self._day_trades = data.get("day_trades", [])
                self._sell_dates = data.get("sell_dates", {})
                self._buy_times = data.get("buy_times", {})
            else:
                self._ledger = data  # legacy format
                self._day_trades = []
                self._sell_dates = {}
                self._buy_times = {}
            log.debug("PDT ledger loaded: %d entries, %d day trades recorded",
                      len(self._ledger), len(self._day_trades))

    def _save(self) -> None:
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LEDGER_PATH, "w") as f:
            json.dump({
                "ledger": self._ledger,
                "day_trades": self._day_trades,
                "sell_dates": self._sell_dates,
                "buy_times": self._buy_times,
            }, f, indent=2)

    # ── helpers ───────────────────────────────────────────────
    def _rolling_day_trade_count(self) -> int:
        """Count total day trades within the rolling PDT_LOOKBACK_DAYS business-day window."""
        today = dt.date.today()
        # Walk back exactly PDT_LOOKBACK_DAYS business days to find the cutoff date
        cutoff = today
        bdays = 0
        while bdays < config.PDT_LOOKBACK_DAYS:
            cutoff -= dt.timedelta(days=1)
            if cutoff.weekday() < 5:  # Mon–Fri
                bdays += 1
        return sum(1 for d in self._day_trades if dt.date.fromisoformat(d) > cutoff)

    # ── public API ───────────────────────────────────────────
    def record_buy(self, symbol: str, fill_date: dt.date | None = None) -> None:
        """Record the date and time a position was opened."""
        fill_date = fill_date or dt.date.today()
        self._ledger[symbol] = fill_date.isoformat()
        self._buy_times[symbol] = dt.datetime.now().isoformat()
        self._save()
        log.info("PDT ledger: recorded BUY  %s on %s", symbol, fill_date)

    def record_sell(self, symbol: str) -> None:
        """Remove symbol from ledger after a successful exit. Records a day trade if applicable."""
        buy_date_str = self._ledger.get(symbol)
        if buy_date_str is not None:
            buy_date = dt.date.fromisoformat(buy_date_str)
            if buy_date >= dt.date.today():
                # This is a day trade — record it
                self._day_trades.append(dt.date.today().isoformat())
                log.info("PDT: day trade recorded for %s (total in window: %d/%d)",
                         symbol, self._rolling_day_trade_count(), config.MAX_DAY_TRADES_ALLOWED)
        self._ledger.pop(symbol, None)
        self._sell_dates[symbol] = dt.date.today().isoformat()
        self._save()
        log.info("PDT ledger: removed %s after SELL", symbol)

    def can_sell_today(self, symbol: str) -> bool:
        """
        Return True if selling symbol today is allowed under PDT rules.
        """
        buy_date_str = self._ledger.get(symbol)
        if buy_date_str is None:
            return True

        buy_date = dt.date.fromisoformat(buy_date_str)
        today = dt.date.today()
        is_day_trade = buy_date >= today

        if is_day_trade:
            # Enforce a minimum hold of 60 minutes to avoid thrashing
            MIN_HOLD_MINUTES = 60
            buy_time_str = self._buy_times.get(symbol)
            if buy_time_str:
                minutes_held = (dt.datetime.now() - dt.datetime.fromisoformat(buy_time_str)).total_seconds() / 60
                if minutes_held < MIN_HOLD_MINUTES:
                    log.warning(
                        "PDT BLOCK: cannot sell %s — only held %.0f min (min=%d min)",
                        symbol, minutes_held, MIN_HOLD_MINUTES,
                    )
                    return False

            if config.MAX_DAY_TRADES_ALLOWED == 0:
                log.warning("PDT BLOCK: cannot sell %s — bought today and day trades are disabled", symbol)
                return False

            used = self._rolling_day_trade_count()
            if used >= config.MAX_DAY_TRADES_ALLOWED:
                log.warning(
                    "PDT BLOCK: cannot sell %s — day trade limit reached (%d/%d in last %d days)",
                    symbol, used, config.MAX_DAY_TRADES_ALLOWED, config.PDT_LOOKBACK_DAYS,
                )
                return False

            log.info("PDT: day trade allowed for %s (%d/%d used)",
                     symbol, used + 1, config.MAX_DAY_TRADES_ALLOWED)
            return True

        # Not a day trade — check minimum hold period
        days_held = (today - buy_date).days
        if days_held < config.MIN_HOLD_CALENDAR_DAYS:
            log.warning(
                "PDT BLOCK: %s held only %d day(s), min=%d",
                symbol, days_held, config.MIN_HOLD_CALENDAR_DAYS,
            )
            return False

        return True

    def can_buy_today(self, symbol: str) -> bool:
        """Block re-buying a symbol that was already sold today (reverse day-trade
        / churn prevention) or within the RE_ENTRY_COOLDOWN_DAYS window."""
        sell_date_str = self._sell_dates.get(symbol)
        if sell_date_str is None:
            return True

        sell_date = dt.date.fromisoformat(sell_date_str)
        today = dt.date.today()
        days_since_sell = (today - sell_date).days

        if days_since_sell == 0:
            log.warning("PDT BLOCK: cannot re-buy %s — sold earlier today (avoids reverse day-trade)", symbol)
            return False

        if days_since_sell < config.RE_ENTRY_COOLDOWN_DAYS:
            log.info("Cooldown: skipping %s — sold %d day(s) ago (cooldown=%d)",
                     symbol, days_since_sell, config.RE_ENTRY_COOLDOWN_DAYS)
            return False

        return True

    def days_held(self, symbol: str) -> int | None:
        """Return how many calendar days a position has been held, or None."""
        buy_date_str = self._ledger.get(symbol)
        if buy_date_str is None:
            return None
        buy_date = dt.date.fromisoformat(buy_date_str)
        return (dt.date.today() - buy_date).days

    def open_symbols(self) -> list[str]:
        """Return all symbols currently tracked in the ledger."""
        return list(self._ledger.keys())

    def cleanup_stale(self, active_symbols: set[str]) -> None:
        """Remove ledger entries for symbols no longer held."""
        stale = set(self._ledger.keys()) - active_symbols
        for sym in stale:
            log.info("PDT ledger: cleaning stale entry %s", sym)
            self._ledger.pop(sym, None)
        if stale:
            self._save()
