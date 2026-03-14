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
        qty: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ):
        """Submit a buy order.

        Alpaca does not allow bracket orders for fractional shares.
        When FRACTIONAL_SHARES is enabled we submit a simple market buy,
        then attach a separate stop-loss sell order so downside is still
        protected.  Take-profit is skipped (the exit scanner handles it).
        """
        if qty <= 0:
            log.warning("Skipping buy – qty=%.4f for %s", qty, symbol)
            return None

        log.info(
            "BUY  %s  qty=%.4f  sl=%.2f  tp=%.2f",
            symbol,
            qty,
            stop_loss or 0,
            take_profit or 0,
        )

        if config.FRACTIONAL_SHARES:
            # Simple market buy (fractional-safe)
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side="buy",
                type="market",
                time_in_force="day",
            )
            # Attach a stop-loss as a separate GTC stop order
            if stop_loss:
                try:
                    self.api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side="sell",
                        type="stop",
                        stop_price=round(stop_loss, 2),
                        time_in_force="day",
                    )
                except Exception as exc:
                    log.warning("Stop-loss order failed for %s: %s", symbol, exc)
            return order

        # Integer qty — use bracket order for full protection
        order_params = dict(
            symbol=symbol,
            qty=int(qty),
            side="buy",
            type="market",
            time_in_force="day",
        )
        if stop_loss and take_profit:
            order_params["order_class"] = "bracket"
            order_params["stop_loss"] = {"stop_price": round(stop_loss, 2)}
            order_params["take_profit"] = {"limit_price": round(take_profit, 2)}
        return self.api.submit_order(**order_params)

    def submit_market_sell(self, symbol: str, qty: float):
        """Submit a market sell (exit position)."""
        if qty <= 0:
            return None
        log.info("SELL %s  qty=%.4f", symbol, qty)
        return self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side="sell",
            type="market",
            time_in_force="day",
        )

    def submit_trailing_stop(self, symbol: str, qty: float, trail_pct: float):
        """Submit a trailing-stop sell order."""
        if qty <= 0:
            return None
        log.info("TRAILING STOP  %s  qty=%f  trail=%.1f%%", symbol, qty, trail_pct * 100)
        
        # Alpaca requires fractional orders to be DAY orders
        is_fractional = qty % 1 != 0
        tif = "day" if is_fractional else "gtc"
        
        return self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side="sell",
            type="trailing_stop",
            trail_percent=str(round(trail_pct * 100, 2)),
            time_in_force=tif,
        )

    def resubmit_stop_losses(self, pdt_guard) -> int:
        """
        Called once each morning after market open.

        For every open position that does NOT already have a live stop or
        trailing-stop order, resubmit a stop-loss at ATR-based distance from
        the original entry price (or a fixed 5 % fallback).

        Returns the number of stop-loss orders newly submitted.
        """
        from risk_manager import RiskManager

        positions = self.get_positions()
        if not positions:
            log.info("Stop-loss refresh: no open positions")
            return 0

        # Build a set of symbols that already have a stop / trailing-stop order
        open_orders = self.get_open_orders()
        protected = {
            o.symbol
            for o in open_orders
            if o.type in ("stop", "stop_limit", "trailing_stop") and o.side == "sell"
        }

        equity = self.get_equity()
        n_pos = len(positions)
        risk = RiskManager(equity, n_pos)
        submitted = 0

        for pos in positions:
            symbol = pos.symbol
            qty = float(pos.qty)
            entry_price = float(pos.avg_entry_price)

            if symbol in protected:
                log.debug("Stop-loss refresh: %s already protected", symbol)
                continue

            # Try to compute ATR-based stop; fall back to fixed 5 %
            stop_price: float | None = None
            try:
                df = self.get_bars(symbol)
                if df is not None and len(df) >= 15:
                    from indicators import compute_all
                    df = compute_all(df)
                    atr = float(df.iloc[-1].get("atr", 0) or 0)
                    if atr > 0:
                        stop_price = risk.compute_stop_loss(entry_price, atr)
            except Exception as exc:
                log.warning("Stop-loss refresh: cannot compute ATR for %s – %s", symbol, exc)

            if stop_price is None or stop_price <= 0:
                stop_price = round(entry_price * (1 - config.STOP_LOSS_ATR_MULT * 0.01), 2)

            # Never place a stop above current price (position already in loss)
            current_price = float(pos.current_price)
            if stop_price >= current_price:
                stop_price = round(current_price * 0.95, 2)

            log.info(
                "Stop-loss refresh: submitting stop for %s  qty=%.4f  stop=%.2f  entry=%.2f",
                symbol, qty, stop_price, entry_price,
            )
            try:
                self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="sell",
                    type="stop",
                    stop_price=round(stop_price, 2),
                    time_in_force="day",
                )
                submitted += 1
            except Exception as exc:
                log.error("Stop-loss refresh failed for %s: %s", symbol, exc)

        log.info("Stop-loss refresh: submitted %d new stop order(s)", submitted)
        return submitted

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
        # The live Alpaca API requires an explicit start date to return
        # historical bars — passing only limit= returns just 1 bar.
        # Use ~1.5x calendar days to account for weekends/holidays.
        start_date = (dt.date.today() - dt.timedelta(days=int(limit * 1.5))).isoformat()
        bars = self.api.get_bars(
            symbol,
            timeframe,
            start=start_date,
            limit=limit,
            feed=config.DATA_FEED,
        )
        df = bars.df.copy()

        # Newer alpaca-trade-api versions return a MultiIndex DataFrame
        # when fetching a single symbol: (symbol, field). Flatten it.
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(symbol, axis=1, level=0) if symbol in df.columns.get_level_values(0) else df.droplevel(0, axis=1)

        df.index = pd.to_datetime(df.index)
        df = df[["open", "high", "low", "close", "volume"]]
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
            # Alpaca requires RFC 3339 / ISO 8601 with 'Z' suffix (UTC)
            params["after"] = after.strftime("%Y-%m-%dT%H:%M:%SZ")
        return self.api.list_orders(**params)
