"""
Alpaca API client wrapper.
Centralises all API calls so the rest of the bot never touches the SDK directly.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import List, Optional
from zoneinfo import ZoneInfo

import alpaca_trade_api as tradeapi
import pandas as pd

import config
from logger import get_logger

log = get_logger("broker")
EASTERN = ZoneInfo("America/New_York")
CORE_OPEN = dt.time(9, 30)
CORE_CLOSE = dt.time(16, 0)
EXTENDED_OPEN = dt.time(4, 0)
EXTENDED_CLOSE = dt.time(20, 0)
OVERNIGHT_OPEN = dt.time(20, 0)
OVERNIGHT_CLOSE = dt.time(4, 0)


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

    def _round_order_price(self, price: float) -> float:
        decimals = 2 if price >= 1 else 4
        return round(price, decimals)

    def _clock_timestamp_et(self, clock) -> dt.datetime:
        timestamp = getattr(clock, "timestamp", None)
        if isinstance(timestamp, str):
            timestamp = dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if timestamp is None:
            timestamp = dt.datetime.now(dt.timezone.utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
        return timestamp.astimezone(EASTERN)

    def get_trading_session(self) -> str:
        """Return the current Alpaca equities session: regular, premarket, afterhours, overnight, or closed."""
        clock = self._get_clock_with_retry()
        if clock.is_open:
            return "regular"

        now_et = self._clock_timestamp_et(clock)
        weekday = now_et.weekday()
        now_time = now_et.timetz().replace(tzinfo=None)

        # Alpaca overnight equities trading runs 8:00pm-4:00am ET,
        # reopening on Sunday night for the Monday session.
        if now_time >= OVERNIGHT_OPEN:
            return "overnight" if weekday in (6, 0, 1, 2, 3) else "closed"
        if now_time < OVERNIGHT_CLOSE:
            return "overnight" if weekday in (0, 1, 2, 3, 4) else "closed"

        if weekday >= 5:
            return "closed"

        if EXTENDED_OPEN <= now_time < CORE_OPEN:
            return "premarket"
        if CORE_CLOSE <= now_time < EXTENDED_CLOSE:
            return "afterhours"
        return "closed"

    def is_extended_hours_session(self) -> bool:
        return self.get_trading_session() in ("premarket", "afterhours", "overnight")

    def is_trading_session_open(self) -> bool:
        session = self.get_trading_session()
        if session == "regular":
            return True
        return bool(
            getattr(config, "EXTENDED_HOURS_TRADING", False)
            and session in ("premarket", "afterhours", "overnight")
        )

    def _should_use_extended_hours_orders(self) -> bool:
        return bool(
            getattr(config, "EXTENDED_HOURS_TRADING", False)
            and self.is_extended_hours_session()
        )

    def _extended_order_tif(self) -> str:
        tif = str(getattr(config, "LIMIT_ORDER_TIF", "day")).lower()
        if tif not in ("day", "gtc"):
            log.warning(
                "Extended-hours orders require day/gtc time_in_force; overriding %s to day",
                tif,
            )
            return "day"
        if getattr(config, "FRACTIONAL_SHARES", False) and tif != "day":
            log.warning(
                "Extended-hours fractional orders require day time_in_force; overriding %s to day",
                tif,
            )
            return "day"
        return tif

    def _normalise_order_qty(self, qty: float):
        if not getattr(config, "FRACTIONAL_SHARES", False) and float(qty).is_integer():
            return int(qty)
        return qty

    def _get_latest_bid_ask(self, symbol: str) -> tuple[float | None, float | None]:
        try:
            quote = self.api.get_latest_quote(symbol)
        except Exception as exc:
            log.debug("Quote lookup failed for %s: %s", symbol, exc)
            return None, None

        def _extract(*names: str) -> float | None:
            for name in names:
                value = getattr(quote, name, None)
                if value is None:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if numeric > 0:
                    return numeric
            return None

        bid = _extract("bidprice", "bid_price", "bp")
        ask = _extract("askprice", "ask_price", "ap")
        return bid, ask

    def _get_marketable_limit_price(self, symbol: str, side: str, offset_pct: float) -> float:
        bid, ask = self._get_latest_bid_ask(symbol)
        if side == "buy":
            reference = ask or self.get_latest_price(symbol)
            return self._round_order_price(reference * (1 + offset_pct))
        reference = bid or self.get_latest_price(symbol)
        return self._round_order_price(reference * (1 - offset_pct))

    def _submit_limit_order(
        self,
        *,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
        time_in_force: str,
        extended_hours: bool = False,
    ):
        order_params = dict(
            symbol=symbol,
            qty=self._normalise_order_qty(qty),
            side=side,
            type="limit",
            limit_price=self._round_order_price(limit_price),
            time_in_force=time_in_force,
        )
        if extended_hours:
            order_params["extended_hours"] = True
        return self.api.submit_order(**order_params)

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

        limit_price = self._round_order_price(limit_price)
        log.info(
            "LIMIT BUY  %s  qty=%.4f  limit=%.2f  sl=%.2f  tp=%.2f",
            symbol,
            qty,
            limit_price,
            stop_loss or 0,
            take_profit or 0,
        )

        use_extended_hours = self._should_use_extended_hours_orders()
        if use_extended_hours:
            if stop_loss or take_profit:
                log.info(
                    "Extended-hours entry for %s skips stop/take-profit attachments until the core session.",
                    symbol,
                )
            return self._submit_limit_order(
                symbol=symbol,
                qty=qty,
                side="buy",
                limit_price=limit_price,
                time_in_force=self._extended_order_tif(),
                extended_hours=True,
            )

        if config.FRACTIONAL_SHARES:
            order = self._submit_limit_order(
                symbol=symbol,
                qty=qty,
                side="buy",
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

    def submit_vwap_buy(
        self,
        symbol: str,
        qty: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        slices: int = 5,
        interval_seconds: int = 180,
    ) -> None:
        """
        Micro-VWAP execution: split *qty* into *slices* limit orders spaced
        *interval_seconds* apart, submitted in a background daemon thread.

        Each slice re-fetches the latest price and applies
        ``LIMIT_ORDER_OFFSET_PCT`` so the limit rises with the breakout.
        A stop-loss is attached to each slice so the position is protected
        even during the spread-out fill window.

        The method returns immediately; execution continues asynchronously.
        """
        import threading
        import time

        if qty <= 0:
            log.warning("VWAP: skipping – qty=%.4f for %s", qty, symbol)
            return

        if config.FRACTIONAL_SHARES:
            per_slice = round(qty / slices, 3)
        else:
            per_slice = int(qty // slices)

        if per_slice <= 0:
            log.warning(
                "VWAP: per-slice qty too small for %s "
                "(total=%.3f slices=%d) – falling back to single order",
                symbol, qty, slices,
            )
            self.submit_market_buy(symbol, qty, stop_loss=stop_loss, take_profit=take_profit)
            return

        def _execute() -> None:
            submitted_qty = 0.0
            for i in range(slices):
                # Final slice absorbs any rounding residual
                if i == slices - 1:
                    slice_qty = (
                        round(qty - submitted_qty, 3)
                        if config.FRACTIONAL_SHARES
                        else int(qty - submitted_qty)
                    )
                else:
                    slice_qty = per_slice

                if slice_qty <= 0:
                    break

                try:
                    last_price = self.get_latest_price(symbol)
                    limit_price = round(
                        last_price * (1 + config.LIMIT_ORDER_OFFSET_PCT), 2
                    )
                    log.info(
                        "VWAP slice %d/%d  %s  qty=%.3f  limit=%.2f",
                        i + 1, slices, symbol, slice_qty, limit_price,
                    )
                    self.submit_limit_buy(
                        symbol,
                        slice_qty,
                        limit_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                    )
                    submitted_qty += slice_qty
                except Exception as exc:
                    log.warning(
                        "VWAP slice %d/%d failed for %s: %s – continuing",
                        i + 1, slices, symbol, exc,
                    )

                if i < slices - 1:
                    time.sleep(interval_seconds)

            log.info(
                "VWAP EXECUTION complete: %s  submitted_qty=%.3f / %.3f",
                symbol, submitted_qty, qty,
            )

        thread = threading.Thread(
            target=_execute, name=f"vwap-{symbol}", daemon=True
        )
        thread.start()
        log.info(
            "VWAP EXECUTION started: %s  total_qty=%.3f  %d slices × %ds",
            symbol, qty, slices, interval_seconds,
        )

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

        use_extended_hours = self._should_use_extended_hours_orders()

        # ── Route to limit order if configured ──────────────
        if config.USE_LIMIT_ORDERS or use_extended_hours:
            try:
                offset_pct = (
                    config.EXTENDED_HOURS_LIMIT_OFFSET_PCT
                    if use_extended_hours
                    else config.LIMIT_ORDER_OFFSET_PCT
                )
                limit_price = self._get_marketable_limit_price(symbol, "buy", offset_pct)

                # Ensure limit buy notional does not exceed available buying power
                try:
                    bp = self.get_buying_power()
                    max_notional = bp * config.MAX_PORTFOLIO_EXPOSURE_PCT
                    if limit_price > 0 and (qty * limit_price) > max_notional:
                        capped_qty = max_notional / limit_price
                        qty = round(capped_qty, 3) if config.FRACTIONAL_SHARES else math.floor(capped_qty)
                        log.info(
                            "Adjusted limit buy qty to %.4f for %s to fit available buying power ($%.2f @ limit $%.2f)",
                            qty, symbol, max_notional, limit_price,
                        )
                except Exception as exc:
                    log.debug("Buying power check before limit buy failed: %s", exc)

                if qty <= 0:
                    log.warning("Skipping limit buy – qty <= 0 after buying power check for %s", symbol)
                    return None

                return self.submit_limit_buy(
                    symbol, qty, limit_price,
                    stop_loss=stop_loss, take_profit=take_profit,
                )
            except Exception as exc:
                if use_extended_hours:
                    log.error(
                        "Could not price extended-hours buy for %s (%s) — no market-order fallback outside core session",
                        symbol, exc,
                    )
                    raise
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
        """Submit an exit sell order, using an extended-hours limit order when required."""
        if qty <= 0:
            return None

        if self._should_use_extended_hours_orders():
            limit_price = self._get_marketable_limit_price(
                symbol,
                "sell",
                config.EXTENDED_HOURS_LIMIT_OFFSET_PCT,
            )
            log.info("EXTENDED HOURS SELL %s  qty=%.4f  limit=%.2f", symbol, qty, limit_price)
            return self._submit_limit_order(
                symbol=symbol,
                qty=qty,
                side="sell",
                limit_price=limit_price,
                time_in_force=self._extended_order_tif(),
                extended_hours=True,
            )

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

        use_extended_hours = self._should_use_extended_hours_orders()

        # Use limit order routing if configured
        if config.USE_LIMIT_ORDERS or use_extended_hours:
            try:
                offset_pct = (
                    config.EXTENDED_HOURS_LIMIT_OFFSET_PCT
                    if use_extended_hours
                    else config.LIMIT_ORDER_OFFSET_PCT
                )
                limit_price = self._get_marketable_limit_price(symbol, "sell", offset_pct)

                # Ensure short sell notional does not exceed available buying power
                try:
                    bp = self.get_buying_power()
                    max_notional = bp * config.MAX_PORTFOLIO_EXPOSURE_PCT
                    if limit_price > 0 and (qty * limit_price) > max_notional:
                        capped_qty = max_notional / limit_price
                        qty = round(capped_qty, 3) if config.FRACTIONAL_SHARES else math.floor(capped_qty)
                        log.info(
                            "Adjusted short limit sell qty to %.4f for %s to fit available buying power ($%.2f @ limit $%.2f)",
                            qty, symbol, max_notional, limit_price,
                        )
                except Exception as exc:
                    log.debug("Buying power check before short limit sell failed: %s", exc)

                if qty <= 0:
                    log.warning("Skipping short sell – qty <= 0 after buying power check for %s", symbol)
                    return None

                order = self._submit_limit_order(
                    symbol=symbol,
                    qty=qty,
                    side="sell",
                    limit_price=limit_price,
                    time_in_force=(
                        self._extended_order_tif()
                        if use_extended_hours
                        else config.LIMIT_ORDER_TIF
                    ),
                    extended_hours=use_extended_hours,
                )
                # Attach stop-loss (buy-to-cover) as separate order
                if stop_loss:
                    if use_extended_hours:
                        log.info(
                            "Extended-hours short entry for %s skips stop-loss attachment until the core session.",
                            symbol,
                        )
                    else:
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
                if use_extended_hours:
                    log.error(
                        "Extended-hours short sell failed for %s (%s) — no market-order fallback outside core session",
                        symbol, exc,
                    )
                    raise
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
        """Buy-to-cover, using an extended-hours limit order when required."""
        if qty <= 0:
            return None

        if self._should_use_extended_hours_orders():
            limit_price = self._get_marketable_limit_price(
                symbol,
                "buy",
                config.EXTENDED_HOURS_LIMIT_OFFSET_PCT,
            )
            log.info("EXTENDED HOURS COVER %s  qty=%.4f  limit=%.2f", symbol, qty, limit_price)
            return self._submit_limit_order(
                symbol=symbol,
                qty=qty,
                side="buy",
                limit_price=limit_price,
                time_in_force=self._extended_order_tif(),
                extended_hours=True,
            )

        log.info("COVER (buy-to-cover) %s  qty=%.4f", symbol, qty)
        return self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="market",
            time_in_force="day",
        )

    def submit_trailing_stop(self, symbol: str, qty: float, trail_pct: float,
                             stop_price: float | None = None,
                             trail_amount: float | None = None):
        """Submit a trailing-stop sell order.

        Parameters
        ----------
        trail_pct    : fallback percentage (used only when stop_price/trail_amount
                       are not provided — legacy static-% mode).
        stop_price   : absolute stop price (Chandelier Exit).  Used for emulated
                       (fractional-share) stops.
        trail_amount : dollar distance for native trailing stops (integer shares).
                       If not provided, falls back to trail_pct.
        """
        if qty <= 0:
            return None
        
        is_fractional = qty % 1 != 0
        
        if is_fractional:
            # Emulated stop for fractional shares — use explicit stop_price
            # when available, otherwise fall back to percentage.
            if stop_price is None:
                try:
                    current_price = self.get_latest_price(symbol)
                    stop_price = round(current_price * (1 - trail_pct), 2)
                except Exception as exc:
                    log.warning("Emulated trailing stop failed for %s: %s", symbol, exc)
                    return None

            log.info("TRAILING STOP (Emulated) %s  qty=%f  stop=%.2f  (fractional fallback)",
                     symbol, qty, stop_price)
            try:
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
        if trail_amount is not None and trail_amount > 0:
            log.info("TRAILING STOP  %s  qty=%f  trail=$%.2f (ATR-based)",
                     symbol, qty, trail_amount)
            return self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side="sell",
                type="trailing_stop",
                trail_price=str(round(trail_amount, 2)),
                time_in_force="gtc",
            )

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
                            stop_price = risk.compute_short_stop_loss(entry_price, atr, symbol)
                        else:
                            stop_price = risk.compute_stop_loss(entry_price, atr, symbol)
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

    def cancel_orders_for_symbol(self, symbol: str) -> int:
        """
        Cancel all open orders for a single symbol and return the count cancelled.

        Must be called before submitting a sell/cover when an existing open order
        may already have the shares locked (Alpaca tracks qty vs available_qty
        separately — locked shares show as qty > 0 but available = 0, causing
        'insufficient qty available' rejections).
        """
        orders = self.get_open_orders(symbol=symbol)
        cancelled = 0
        for o in orders:
            try:
                self.api.cancel_order(o.id)
                log.info("Cancelled order %s for %s (side=%s qty=%s)",
                         o.id, symbol, o.side, o.qty)
                cancelled += 1
            except Exception as exc:
                log.warning("Failed to cancel order %s for %s: %s", o.id, symbol, exc)
        return cancelled

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

        Returns the most recent `limit` bars.
        """
        # The live Alpaca API requires an explicit start date to return
        # historical bars — passing only limit= returns just 1 bar.
        # Use 2x calendar days to generously account for weekends/holidays.
        # We intentionally omit the API-level `limit` and instead fetch all
        # bars in the date range, then keep the last `limit` rows.  The API
        # truncates from the START when you pass both start+limit, which
        # silently drops the most recent bars when trading days > limit.
        start_date = (dt.date.today() - dt.timedelta(days=int(limit * 2))).isoformat()
        bars = self.api.get_bars(
            symbol,
            timeframe,
            start=start_date,
            feed=config.DATA_FEED,
        )
        df = bars.df.copy()

        # Newer alpaca-trade-api versions return a MultiIndex DataFrame
        # when fetching a single symbol: (symbol, field). Flatten it.
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(symbol, axis=1, level=0) if symbol in df.columns.get_level_values(0) else df.droplevel(0, axis=1)

        df.index = pd.to_datetime(df.index)
        df = df[["open", "high", "low", "close", "volume"]]
        # Keep only the most recent `limit` bars
        if len(df) > limit:
            df = df.tail(limit)
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
