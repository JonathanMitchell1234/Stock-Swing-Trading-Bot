"""
Trade Journal — persists entry/exit metadata to a local SQLite database.

Database: logs/trades.db
"""

from __future__ import annotations

import datetime as dt
import math
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "logs" / "trades.db"

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    side            TEXT    NOT NULL,        -- 'long', 'short', 'inverse'
    entry_time      TEXT,
    exit_time       TEXT,
    entry_price     REAL,
    exit_price      REAL,
    qty             REAL,
    stop_loss       REAL,
    take_profit     REAL,
    signal_score    INTEGER,
    ml_prob         REAL,
    signal_reason   TEXT,
    regime          TEXT,                    -- 'bull' or 'bear'
    vwap_used       INTEGER DEFAULT 0,       -- 0 or 1
    requested_price REAL,
    filled_price    REAL,
    slippage_bps    REAL,
    order_type      TEXT,                    -- 'market', 'limit', 'vwap'
    pnl             REAL,
    pnl_pct         REAL,
    hold_days       INTEGER,
    exit_reason     TEXT
)
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_CREATE_SQL)
    conn.commit()
    return conn


class TradeJournal:
    """Thread-safe SQLite-backed trade journal."""

    def __init__(self) -> None:
        conn = _connect()
        conn.close()

    # ─────────────────────────────────────────────────────────
    # Write
    # ─────────────────────────────────────────────────────────

    def record_entry(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        qty: float,
        stop_loss: float,
        take_profit: float,
        signal: dict,
        regime: str = "bull",
        vwap_used: bool = False,
        order_type: str = "market",
    ) -> int:
        """
        Persist entry metadata.  Returns the new row id.
        Exit fields (exit_price, pnl, …) are NULL until record_exit() is called.
        """
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        requested_price = signal.get("price", entry_price)
        ml_prob = signal.get("ml_prob")
        signal_reason = signal.get("reason", "")
        score = signal.get("score")

        conn = _connect()
        try:
            cur = conn.execute(
                """
                INSERT INTO trades
                    (symbol, side, entry_time, entry_price, qty,
                     stop_loss, take_profit, signal_score, ml_prob,
                     signal_reason, regime, vwap_used,
                     requested_price, order_type)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    symbol, side, now, entry_price, qty,
                    stop_loss, take_profit, score, ml_prob,
                    signal_reason, regime, int(vwap_used),
                    requested_price, order_type,
                ),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def record_exit(
        self,
        symbol: str,
        exit_price: float,
        pnl: float,
        hold_days: int,
        exit_reason: str,
        filled_price: float | None = None,
    ) -> None:
        """
        Update the most recent open trade row for *symbol* with exit info.
        If filled_price is provided, slippage_bps is computed against the
        original requested_price.
        """
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT id, requested_price, entry_price FROM trades "
                "WHERE symbol=? AND exit_time IS NULL "
                "ORDER BY id DESC LIMIT 1",
                (symbol,),
            ).fetchone()
            if row is None:
                return

            row_id = row["id"]
            req_price = row["requested_price"] or row["entry_price"]
            slippage_bps: float | None = None
            if filled_price is not None and req_price:
                slippage_bps = round(abs(filled_price - req_price) / req_price * 10_000, 2)

            entry_price = row["entry_price"] or exit_price
            pnl_pct = round(pnl / max(abs(entry_price), 0.0001) * 100, 4) if entry_price else None

            conn.execute(
                """
                UPDATE trades SET
                    exit_time    = ?,
                    exit_price   = ?,
                    filled_price = COALESCE(?, filled_price),
                    slippage_bps = COALESCE(?, slippage_bps),
                    pnl          = ?,
                    pnl_pct      = ?,
                    hold_days    = ?,
                    exit_reason  = ?
                WHERE id = ?
                """,
                (
                    now, exit_price,
                    filled_price, slippage_bps,
                    pnl, pnl_pct,
                    hold_days, exit_reason,
                    row_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def backfill_trade(
        self,
        symbol: str,
        side: str,
        entry_time: str,
        exit_time: str,
        entry_price: float,
        exit_price: float,
        qty: float,
        pnl: float,
        hold_days: int,
    ) -> bool:
        """
        Insert a historical (backfilled) closed trade.
        Skips silently if a trade with the same symbol + entry_time already exists.
        Returns True if a new row was inserted.
        """
        conn = _connect()
        try:
            existing = conn.execute(
                "SELECT id FROM trades WHERE symbol=? AND entry_time=?",
                (symbol, entry_time),
            ).fetchone()
            if existing:
                return False

            pnl_pct = round(pnl / max(abs(entry_price * qty), 1e-9) * 100, 4) if entry_price and qty else None
            conn.execute(
                """
                INSERT INTO trades
                    (symbol, side, entry_time, exit_time,
                     entry_price, exit_price, qty,
                     pnl, pnl_pct, hold_days, order_type)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    symbol, side, entry_time, exit_time,
                    entry_price, exit_price, qty,
                    round(pnl, 4), pnl_pct, int(hold_days), "market",
                ),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    # ─────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────

    def get_closed_trades(self, limit: int = 100) -> list[dict]:
        """Return the most recent *limit* closed trades, newest first."""
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT * FROM trades WHERE exit_time IS NOT NULL "
                "ORDER BY exit_time DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_performance_stats(self) -> dict:
        """
        Compute aggregate performance metrics over all closed trades.
        Returns: total_trades, win_rate, profit_factor, avg_win, avg_loss,
                 avg_hold_days, sharpe_ratio, max_drawdown_pct,
                 current_drawdown_pct, rolling_10_win_rate.
        """
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT pnl, hold_days FROM trades "
                "WHERE exit_time IS NOT NULL AND pnl IS NOT NULL "
                "ORDER BY exit_time ASC",
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return {
                "total_trades": 0, "win_rate": None, "profit_factor": None,
                "avg_win": None, "avg_loss": None, "avg_hold_days": None,
                "sharpe_ratio": None, "max_drawdown_pct": None,
                "current_drawdown_pct": None, "rolling_10_win_rate": None,
            }

        pnls = [r["pnl"] for r in rows]
        hold_days_list = [r["hold_days"] for r in rows if r["hold_days"] is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total = len(pnls)

        win_rate = round(len(wins) / total * 100, 1) if total else None
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = round(gross_profit / gross_loss, 3) if gross_loss else None
        avg_win = round(sum(wins) / len(wins), 2) if wins else None
        avg_loss = round(sum(losses) / len(losses), 2) if losses else None
        avg_hold_days = round(sum(hold_days_list) / len(hold_days_list), 1) if hold_days_list else None

        # Annualised Sharpe: mean / std * sqrt(252 / avg_hold_days)
        sharpe_ratio = None
        if len(pnls) >= 5:
            n = len(pnls)
            mean = sum(pnls) / n
            variance = sum((p - mean) ** 2 for p in pnls) / n
            std = math.sqrt(variance)
            if std > 0:
                avg_days = avg_hold_days or 1
                ann_factor = math.sqrt(252 / max(avg_days, 1))
                sharpe_ratio = round((mean / std) * ann_factor, 3)

        # Drawdown from running cumulative P&L peak
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            cumulative += p
            peak = max(peak, cumulative)
            dd = (peak - cumulative) / peak * 100 if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        current_dd = (peak - cumulative) / peak * 100 if peak > 0 else 0.0

        # Rolling last-10 win rate
        last10 = pnls[-10:]
        rolling_10_wr = round(sum(1 for p in last10 if p > 0) / len(last10) * 100, 1) if last10 else None

        return {
            "total_trades":         total,
            "win_rate":             win_rate,
            "profit_factor":        profit_factor,
            "avg_win":              avg_win,
            "avg_loss":             avg_loss,
            "avg_hold_days":        avg_hold_days,
            "sharpe_ratio":         sharpe_ratio,
            "max_drawdown_pct":     round(max_dd, 2),
            "current_drawdown_pct": round(current_dd, 2),
            "rolling_10_win_rate":  rolling_10_wr,
        }

    def get_fill_metrics(self) -> dict:
        """
        Return fill-quality metrics: per-symbol and aggregate.
        Only trades where slippage_bps was recorded are included.
        """
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT symbol, order_type, slippage_bps "
                "FROM trades "
                "WHERE exit_time IS NOT NULL AND slippage_bps IS NOT NULL",
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return {"aggregate": {}, "by_symbol": [], "by_order_type": []}

        all_slip = [r["slippage_bps"] for r in rows]
        avg_slip = round(sum(all_slip) / len(all_slip), 2)

        sym_data: dict[str, list[float]] = {}
        for r in rows:
            sym_data.setdefault(r["symbol"], []).append(r["slippage_bps"])
        by_symbol = [
            {
                "symbol": sym,
                "avg_slippage_bps": round(sum(v) / len(v), 2),
                "trades": len(v),
            }
            for sym, v in sorted(sym_data.items())
        ]

        type_data: dict[str, list[float]] = {}
        for r in rows:
            otype = r["order_type"] or "market"
            type_data.setdefault(otype, []).append(r["slippage_bps"])
        by_order_type = [
            {
                "order_type": otype,
                "avg_slippage_bps": round(sum(v) / len(v), 2),
                "trades": len(v),
            }
            for otype, v in sorted(type_data.items())
        ]

        return {
            "aggregate": {
                "avg_slippage_bps":     avg_slip,
                "total_fills_tracked":  len(rows),
            },
            "by_symbol":     by_symbol,
            "by_order_type": by_order_type,
        }
