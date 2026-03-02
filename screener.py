"""
Stock Screener – filters the watchlist down to actionable candidates.
"""

from __future__ import annotations

from typing import List

import pandas as pd

import config
from broker import AlpacaBroker
from indicators import compute_all, latest_row
from logger import get_logger

log = get_logger("screener")


class Screener:
    """Pre-filters the watchlist to find stocks worth analysing."""

    def __init__(self, broker: AlpacaBroker) -> None:
        self.broker = broker

    def screen(self, symbols: List[str] | None = None) -> List[dict]:
        """
        For each symbol in the watchlist, fetch bars, compute indicators,
        and return a list of dicts for symbols that pass the pre-filter:

            { symbol, df, latest, entry_price, atr, ... }
        """
        symbols = symbols or config.WATCHLIST
        candidates = []

        for symbol in symbols:
            try:
                df = self.broker.get_bars(
                    symbol,
                    limit=max(config.BARS_LOOKBACK, config.MIN_BARS_HISTORY_ENTRY),
                )
                if df is None or len(df) < config.EMA_TREND + 5:
                    continue

                if len(df) < config.MIN_BARS_HISTORY_ENTRY:
                    continue

                df = compute_all(df)
                row = latest_row(df)

                # ── Price filter ─────────────────────────────
                price = row["close"]
                if price < config.MIN_PRICE or price > config.MAX_PRICE:
                    continue

                # ── Volume filter ────────────────────────────
                avg_vol = row.get("vol_sma", 0)
                if avg_vol < config.MIN_AVG_VOLUME:
                    continue

                candidates.append(
                    {
                        "symbol": symbol,
                        "df": df,
                        "latest": row,
                        "price": price,
                        "atr": row["atr"],
                        "rsi": row["rsi"],
                        "macd_hist": row["macd_hist"],
                        "adx": row["adx"],
                        "vol_ratio": row["vol_ratio"],
                    }
                )
            except Exception as exc:
                log.warning("Screener error for %s: %s", symbol, exc)

        log.info("Screener: %d / %d symbols passed pre-filter", len(candidates), len(symbols))
        return candidates

    def compute_breadth(self, symbols: List[str] | None = None) -> float | None:
        """
        Compute market breadth as the fraction of symbols above EMA-50.
        Returns None if too few symbols had usable data.
        """
        symbols = symbols or config.WATCHLIST
        valid = 0
        above = 0

        for symbol in symbols:
            try:
                df = self.broker.get_bars(
                    symbol,
                    limit=max(config.BARS_LOOKBACK, config.EMA_TREND + 10),
                )
                if df is None or len(df) < config.EMA_TREND + 5:
                    continue

                df = compute_all(df)
                row = latest_row(df)
                close = row.get("close", None)
                ema_trend = row.get("ema_trend", None)
                if close is None or ema_trend is None or pd.isna(close) or pd.isna(ema_trend):
                    continue

                valid += 1
                if close > ema_trend:
                    above += 1
            except Exception:
                continue

        if valid < config.BREADTH_MIN_SYMBOLS:
            return None
        return above / valid
