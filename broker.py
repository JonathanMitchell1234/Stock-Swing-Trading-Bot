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
    def submit_limit_buy(
        self,
        symbol: str,
        qty: float,
        limit_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ):
        """Submit a limit buy order at *limit_price*.

        For fractional quantities Alpaca does not support bracket orders, so a
        separate stop-loss order is attached after the limit buy is placed.
        For integer quantities a bracket order is used when SL/TP are provided.
        """
        if qty <= 0:
            log.warning("Skipping limit buy – qty=%.4f for %s", qty, symbol)
            return None

        limit_price = round(limit_price, 2)
        log.info(
            "LIMIT BUY  %s  qty=%.4f  limit=%.2f  sl=%.2f  tp=%.2f",
            symbol,
            qty,
            limit_price,
            stop_loss or 0,
            take_profit or 0,
        )

        if config.FRACTIONAL_SHARES:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side="buy",
                type="limit",
                limit_price=limit_price,
                time_in_force=config.LIMIT_ORDER_TIF,
            )
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

        # Integer qty – use bracket order when SL/TP provided
        order_params = dict(
            symbol=symbol,
            qty=int(qty),
            side="buy",
            type="limit",
            limit_price=limit_price,
            time_in_force=config.LIMIT_ORDER_TIF,
        )
        if stop_loss and take_profit:
            order_params["order_class"] = "bracket"
            order_params["stop_loss"] = {"stop_price": round(stop_loss, 2)}
            order_params["take_profit"] = {"limit_price": round(take_profit, 2)}
        return self.api.submit_order(**order_params)

    def submit_market_buy(
        self,
        symbol: str,
        qty: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ):
        """Submit a buy order.

        Routes to a limit order automatically when config.USE_LIMIT_ORDERS is
        True, fetching the latest price and applying LIMIT_ORDER_OFFSET_PCT.

        Alpaca does not allow bracket orders for fractional shares.
        When FRACTIONAL_SHARES is enabled we submit a simple market buy,
        then attach a separate stop-loss sell order so downside is still
        protected.  Take-profit is skipped (the exit scanner handles it).
        """
        if qty <= 0:
            log.warning("Skipping buy – qty=%.4f for %s", qty, symbol)
            return None

        # ── Route to limit order if configured ──────────────
        if config.USE_LIMIT_ORDERS:
            try:
                last_price = self.get_latest_price(symbol)
                limit_price = round(last_price * (1 + config.LIMIT_ORDER_OFFSET_PCT), 2)
                return self.submit_limit_buy(
                    symbol, qty, limit_price,
                    stop_loss=stop_loss, take_profit=take_profit,
                )
            except Exception as exc:
                log.warning(
                    "Could not fetch price for limit order on %s (%s) – falling back to market order",
                    symbol, exc,
                )

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

    def submit_short_sell(
        self,
        symbol: str,
        qty: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ):
        """
        Open a SHORT position by selling shares we don't own.
        Alpaca supports short selling on margin accounts.

        For shorts:
          - stop_loss is ABOVE entry (buy-to-cover if price rises)
          - take_profit is BELOW entry (buy-to-cover at profit target)
        """
        if qty <= 0:
            log.warning("Skipping short sell – qty=%.4f for %s", qty, symbol)
            return None

        log.info(
            "SHORT SELL  %s  qty=%.4f  sl=%.2f  tp=%.2f",
            symbol, qty, stop_loss or 0, take_profit or 0,
        )

        # Use limit order routing if configured
        if config.USE_LIMIT_ORDERS:
            try:
                last_price = self.get_latest_price(symbol)
                # For shorts, limit BELOW current price (willing to sell at slightly less)
                limit_price = round(last_price * (1 - config.LIMIT_ORDER_OFFSET_PCT), 2)
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="sell",
                    type="limit",
                    limit_price=limit_price,
                    time_in_force=config.LIMIT_ORDER_TIF,
                )
                # Attach stop-loss (buy-to-cover) as separate order
                if stop_loss:
                    try:
                        self.api.submit_order(
                            symbol=symbol,
                            qty=qty,
                            side="buy",
                            type="stop",
                            stop_price=round(stop_loss, 2),
                            time_in_force="day",
                        )
                    except Exception as exc:
                        log.warning("Short stop-loss order failed for %s: %s", symbol, exc)
                return order
            except Exception as exc:
                log.warning(
                    "Limit short sell failed for %s (%s) – falling back to market",
                    symbol, exc,
                )

        # Market short sell
        order = self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side="sell",
            type="market",
            time_in_force="day",
        )
        # Attach stop-loss (buy-to-cover) as separate order
        if stop_loss:
            try:
                self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="buy",
                    type="stop",
                    stop_price=round(stop_loss, 2),
                    time_in_force="day",
                )
            except Exception as exc:
                log.warning("Short stop-loss order failed for %s: %s", symbol, exc)
        return order

    def submit_market_cover(self, symbol: str, qty: float):
        """Buy-to-cover: close a short position by buying shares back."""
        if qty <= 0:
            return None
        log.info("COVER (buy-to-cover) %s  qty=%.4f", symbol, qty)
        return self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="market",
            time_in_force="day",
        )

    def submit_trailing_stop(self, symbol: str, qty: float, trail_pct: float):
        """Submit a trailing-stop sell order.
        Note: Alpaca does not support trailing stop orders for fractional shares.
        If qty is fractional, we will submit a standard stop order based on the 
        current price and the trail_pct. The executor will continuously update 
        this stop if the price moves up.
        """
        if qty <= 0:
            return None
        
        is_fractional = qty % 1 != 0
        
        if is_fractional:
            log.info("TRAILING STOP (Emulated) %s  qty=%f  trail=%.1f%%  (fractional fallback)", symbol, qty, trail_pct * 100)
            try:
                current_price = self.get_latest_price(symbol)
                stop_price = round(current_price * (1 - trail_pct), 2)
                return self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="sell",
                    type="stop",
                    stop_price=stop_price,
                    time_in_force="day",
                )
            except Exception as exc:
                log.warning("Emulated trailing stop failed for %s: %s", symbol, exc)
                return None
                
        # Pure integer share sizes can use native trailing stops
        log.info("TRAILING STOP  %s  qty=%f  trail=%.1f%%", symbol, qty, trail_pct * 100)
        
        return self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side="sell",
            type="trailing_stop",
            trail_percent=str(round(trail_pct * 100, 2)),
            time_in_force="gtc",
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
            if o.type in ("stop", "stop_limit", "trailing_stop")
               and o.side in ("sell", "buy")
        }

        equity = self.get_equity()
        n_pos = len(positions)
        risk = RiskManager(equity, n_pos)
        submitted = 0

        for pos in positions:
            symbol = pos.symbol
            qty = float(pos.qty)
            is_short = qty < 0
            abs_qty = abs(qty)
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
                        if is_short:
                            stop_price = risk.compute_short_stop_loss(entry_price, atr)
                        else:
                            stop_price = risk.compute_stop_loss(entry_price, atr)
            except Exception as exc:
                log.warning("Stop-loss refresh: cannot compute ATR for %s – %s", symbol, exc)

            current_price = float(pos.current_price)

            if is_short:
                # Short stop is ABOVE entry (buy-to-cover)
                if stop_price is None or stop_price <= 0:
                    stop_price = round(entry_price * 1.05, 2)  # fixed 5% above
                # Never place a stop below current price for shorts (already winning)
                if stop_price <= current_price:
                    stop_price = round(current_price * 1.05, 2)

                log.info(
                    "Stop-loss refresh (SHORT): submitting buy-stop for %s  qty=%.4f  stop=%.2f  entry=%.2f",
                    symbol, abs_qty, stop_price, entry_price,
                )
                try:
                    self.api.submit_order(
                        symbol=symbol,
                        qty=abs_qty,
                        side="buy",
                        type="stop",
                        stop_price=round(stop_price, 2),
                        time_in_force="day",
                    )
                    submitted += 1
                except Exception as exc:
                    log.error("Stop-loss refresh failed for short %s: %s", symbol, exc)
            else:
                # Long stop is BELOW entry (sell)
                if stop_price is None or stop_price <= 0:
                    stop_price = round(entry_price * 0.95, 2)  # fixed 5% below
                # Never place a stop above current price (position already in loss)
                if stop_price >= current_price:
                    stop_price = round(current_price * 0.95, 2)

                log.info(
                    "Stop-loss refresh: submitting stop for %s  qty=%.4f  stop=%.2f  entry=%.2f",
                    symbol, abs_qty, stop_price, entry_price,
                )
                try:
                    self.api.submit_order(
                        symbol=symbol,
                        qty=abs_qty,
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

    # ── News Data ────────────────────────────────────────────
    def get_news(self, symbol: str, limit: int = 10, end: str | None = None) -> list[str]:
        """Fetch latest news headlines for a symbol before an optional end date (RFC3339)."""
        try:
            kwargs = {"limit": limit}
            if end:
                kwargs["end"] = end
            news_items = self.api.get_news(symbol, **kwargs)
            return [item.headline for item in news_items if hasattr(item, 'headline')]
        except Exception as exc:
            log.warning("Failed to fetch news for %s: %s", symbol, exc)
            return []

    # ── Market Clock ─────────────────────────────────────────
    def _get_clock_with_retry(self, retries: int = 5, backoff: float = 2.0):
        """Call api.get_clock() with exponential-backoff retries on transient connection errors."""
        import time as _time
        from requests.exceptions import ConnectionError as _ConnError
        delay = 2.0
        for attempt in range(1, retries + 1):
            try:
                return self.api.get_clock()
            except (_ConnError, Exception) as exc:
                # Only retry on connection-level errors; re-raise others on last attempt
                is_conn_err = isinstance(exc, _ConnError) or "RemoteDisconnected" in str(exc) or "Connection aborted" in str(exc)
                if not is_conn_err or attempt == retries:
                    raise
                log.warning(
                    "Transient connection error calling get_clock (attempt %d/%d): %s — retrying in %.0fs",
                    attempt, retries, exc, delay,
                )
                _time.sleep(delay)
                delay = min(delay * backoff, 60.0)

    def is_market_open(self) -> bool:
        clock = self._get_clock_with_retry()
        return clock.is_open

    def get_clock(self):
        return self._get_clock_with_retry()

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
