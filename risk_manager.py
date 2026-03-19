"""
Risk Manager – position sizing and portfolio-level risk controls.
Supports fractional shares for small accounts ($200-$300+).
"""

from __future__ import annotations

import math
from typing import Optional

import config
from logger import get_logger

log = get_logger("risk_mgr")


class RiskManager:
    """Decides *how much* to buy and enforces portfolio-level guardrails."""

    def __init__(self, equity: float, open_positions: int) -> None:
        self.equity = equity
        self.open_positions = open_positions

    def refresh(self, equity: float, open_positions: int) -> None:
        self.equity = equity
        self.open_positions = open_positions

    # ── checks ───────────────────────────────────────────────
    def can_open_new_position(self) -> bool:
        max_pos = config.get_max_positions(self.equity)
        if self.open_positions >= max_pos:
            log.info(
                "Risk: max positions reached (%d/%d)",
                self.open_positions,
                max_pos,
            )
            return False
        return True

    # ── position sizing ──────────────────────────────────────
    def calculate_position_size(
        self,
        entry_price: float,
        stop_price: float,
        buying_power: float,
    ) -> float:
        """
        Size the position so that if the stop is hit, the loss equals
        get_risk_per_trade() of equity.  Also cap at get_position_pct()
        of equity and available buying power.

        Returns the number of shares to buy (fractional if enabled,
        otherwise whole shares).  Returns 0 if trade is too risky or
        there isn't enough capital.
        """
        if entry_price <= 0 or stop_price <= 0 or stop_price >= entry_price:
            log.warning(
                "Invalid prices for sizing: entry=%.2f stop=%.2f",
                entry_price,
                stop_price,
            )
            return 0

        risk_per_share = entry_price - stop_price
        if risk_per_share <= 0:
            return 0

        # Use dynamic parameters based on account size
        risk_pct = config.get_risk_per_trade(self.equity)
        pos_pct = config.get_position_pct(self.equity)

        # 1. Size by risk budget
        max_risk_dollars = self.equity * risk_pct
        shares_by_risk = max_risk_dollars / risk_per_share

        # 2. Size by max position value
        max_position_value = self.equity * pos_pct
        shares_by_value = max_position_value / entry_price

        # 3. Size by buying power
        usable_bp = buying_power * config.MAX_PORTFOLIO_EXPOSURE_PCT
        shares_by_bp = usable_bp / entry_price

        qty_raw = min(shares_by_risk, shares_by_value, shares_by_bp)

        # Apply fractional vs whole share rounding
        if config.FRACTIONAL_SHARES:
            qty = round(qty_raw, 3)   # Alpaca supports 0.001 precision
        else:
            qty = math.floor(qty_raw)

        qty = max(0, qty)

        log.info(
            "Sizing: risk=%.3f  value=%.3f  bp=%.3f -> qty=%.3f  (entry=%.2f  stop=%.2f)",
            shares_by_risk,
            shares_by_value,
            shares_by_bp,
            qty,
            entry_price,
            stop_price,
        )
        return qty

    # ── stop / target ────────────────────────────────────────
    def compute_stop_loss(self, entry: float, atr: float) -> float:
        """ATR-based stop loss (uses dynamic multiplier for account size)."""
        mult = config.get_atr_stop_mult(self.equity)
        return round(entry - atr * mult, 2)

    def compute_take_profit(self, entry: float, atr: float) -> float:
        """ATR-based profit target (uses dynamic multiplier for account size)."""
        mult = config.get_atr_profit_mult(self.equity)
        return round(entry + atr * mult, 2)

    # ── SHORT-specific stop / target ─────────────────────────
    def compute_short_stop_loss(self, entry: float, atr: float) -> float:
        """ATR-based stop loss for shorts — ABOVE entry price (buy-to-cover)."""
        mult = config.get_atr_stop_mult(self.equity)
        return round(entry + atr * mult, 2)

    def compute_short_take_profit(self, entry: float, atr: float) -> float:
        """ATR-based profit target for shorts — BELOW entry price."""
        mult = config.get_atr_profit_mult(self.equity)
        return round(entry - atr * mult, 2)

    def calculate_short_position_size(
        self,
        entry_price: float,
        stop_price: float,
        buying_power: float,
    ) -> float:
        """
        Size a SHORT position so that if the stop is hit, the loss equals
        get_risk_per_trade() of equity.

        For shorts, stop_price is ABOVE entry_price (buy-to-cover on adverse move).
        """
        if entry_price <= 0 or stop_price <= 0 or stop_price <= entry_price:
            log.warning(
                "Invalid prices for short sizing: entry=%.2f stop=%.2f",
                entry_price, stop_price,
            )
            return 0

        risk_per_share = stop_price - entry_price
        if risk_per_share <= 0:
            return 0

        risk_pct = config.get_risk_per_trade(self.equity)
        pos_pct = config.get_position_pct(self.equity)

        max_risk_dollars = self.equity * risk_pct
        shares_by_risk = max_risk_dollars / risk_per_share

        max_position_value = self.equity * pos_pct
        shares_by_value = max_position_value / entry_price

        usable_bp = buying_power * config.MAX_PORTFOLIO_EXPOSURE_PCT
        shares_by_bp = usable_bp / entry_price

        qty_raw = min(shares_by_risk, shares_by_value, shares_by_bp)

        if config.FRACTIONAL_SHARES:
            qty = round(qty_raw, 3)
        else:
            qty = math.floor(qty_raw)

        qty = max(0, qty)

        log.info(
            "Short sizing: risk=%.3f  value=%.3f  bp=%.3f -> qty=%.3f  (entry=%.2f  stop=%.2f)",
            shares_by_risk, shares_by_value, shares_by_bp,
            qty, entry_price, stop_price,
        )
        return qty

    def portfolio_at_risk(self, positions: list) -> float:
        """
        Rough estimate of total portfolio risk (sum of unrealised P&L
        as a fraction of equity).
        """
        total_risk = 0.0
        for pos in positions:
            unrealised = float(pos.unrealized_plpc)  # decimal fraction
            total_risk += abs(unrealised) * float(pos.market_value)
        return total_risk / self.equity if self.equity > 0 else 0.0
