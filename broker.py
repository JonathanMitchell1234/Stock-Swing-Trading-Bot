"""
Alpaca API client wrapper.
Centralises all API calls so the rest of the bot never touches the SDK directly.
"""

from __future__ import annotations

import datetime as dt
from typing import List, Optional

import alpaca_trade_api as tradeapi
import pandas as pd

import config
from logger import get_logger

log = get_logger("broker")


class AlpacaBroker:
    """Thin wrapper around the Alpaca REST API."""

    def __init__(self) -> None:
        self.api = tradeapi.REST(
            key_id=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
            base_url=config.BASE_URL,
            api_version="v2",
        )
        log.info(
            "Broker connected  mode=%s  url=%s",
            config.TRADING_MODE,
            config.BASE_URL,
        )

    # ── Account ──────────────────────────────────────────────
    def get_account(self):
        """Return the full Alpaca account object."""
        return self.api.get_account()

    def get_equity(self) -> float:
        return float(self.api.get_account().equity)

    def get_cash(self) -> float:
        return float(self.api.get_account().cash)

    def get_buying_power(self) -> float:
        return float(self.api.get_account().buying_power)

    # ── Positions ────────────────────────────────────────────
    def get_positions(self) -> list:
        """Return list of current open positions."""
        return self.api.list_positions()

    def get_position(self, symbol: str):
        """Return position for a single symbol, or None."""
        try:
            return self.api.get_position(symbol)
        except tradeapi.rest.APIError:
            return None

    def has_position(self, symbol: str) -> bool:
        return self.get_position(symbol) is not None

    # ── Orders ───────────────────────────────────────────────
    def submit_market_buy(
        self,
        symbol: str,
        qty: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ):
        """Submit a bracket or simple market buy order."""
        if qty <= 0:
            log.warning("Skipping buy – qty=%d for %s", qty, symbol)
            return None

        order_params = dict(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="market",
            time_in_force="day",
        )

        # Use bracket order if both SL and TP are provided
        if stop_loss and take_profit:
            order_params["order_class"] = "bracket"
            order_params["stop_loss"] = {"stop_price": round(stop_loss, 2)}
            order_params["take_profit"] = {"limit_price": round(take_profit, 2)}

        log.info(
            "BUY  %s  qty=%d  sl=%.2f  tp=%.2f",
            symbol,
            qty,
            stop_loss or 0,
            take_profit or 0,
        )
        return self.api.submit_order(**order_params)

    def submit_market_sell(self, symbol: str, qty: int):
        """Submit a market sell (exit position)."""
        if qty <= 0:
            return None
        log.info("SELL %s  qty=%d", symbol, qty)
        return self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side="sell",
            type="market",
            time_in_force="day",
        )

    def submit_trailing_stop(self, symbol: str, qty: int, trail_pct: float):
        """Submit a trailing-stop sell order."""
        if qty <= 0:
            return None
        log.info("TRAILING STOP  %s  qty=%d  trail=%.1f%%", symbol, qty, trail_pct * 100)
        return self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side="sell",
            type="trailing_stop",
            trail_percent=str(round(trail_pct * 100, 2)),
            time_in_force="gtc",
        )

    def cancel_all_orders(self):
        self.api.cancel_all_orders()
        log.info("All open orders cancelled")

    def get_open_orders(self, symbol: Optional[str] = None) -> list:
        orders = self.api.list_orders(status="open")
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    # ── Historical Data ──────────────────────────────────────
    def get_bars(
        self,
        symbol: str,
        timeframe: str = config.BAR_TIMEFRAME,
        limit: int = config.BARS_LOOKBACK,
    ) -> pd.DataFrame:
        """
        Fetch historical bars and return a clean DataFrame.
        Columns: open, high, low, close, volume
        """
        bars = self.api.get_bars(
            symbol,
            timeframe,
            limit=limit,
        )
        df = bars.df.copy()
        if df is None or df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        required = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required):
            return pd.DataFrame(columns=required)

        df.index = pd.to_datetime(df.index)
        df = df[required]
        return df

    def get_latest_price(self, symbol: str) -> float:
        """Get most recent trade price."""
        trade = self.api.get_latest_trade(symbol)
        return float(trade.price)

    # ── Market Clock ─────────────────────────────────────────
    def is_market_open(self) -> bool:
        clock = self.api.get_clock()
        return clock.is_open

    def get_clock(self):
        return self.api.get_clock()

    # ── Activity / Order History (for PDT tracking) ──────────
    def get_closed_orders(
        self, after: Optional[dt.datetime] = None, limit: int = 200
    ) -> list:
        """Return recently closed (filled) orders for PDT tracking."""
        params = {"status": "closed", "limit": limit, "direction": "desc"}
        if after:
            params["after"] = after.isoformat()
        return self.api.list_orders(**params)
