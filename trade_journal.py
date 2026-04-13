"""
Trade Journal — persists entry/exit metadata to a local SQLite database.

Database: logs/trades.db
"""

from __future__ import annotations

import datetime as dt
import math
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

import config

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
        If filled_price is provided, slippage_bps is computed as the
        deviation of the actual fill from the intended exit price.
        """
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT id, entry_price, qty FROM trades "
                "WHERE symbol=? AND exit_time IS NULL "
                "ORDER BY id DESC LIMIT 1",
                (symbol,),
            ).fetchone()
            if row is None:
                return

            row_id = row["id"]
            # Slippage = deviation of actual fill from the intended exit price
            slippage_bps: float | None = None
            if filled_price is not None and exit_price and exit_price > 0:
                slippage_bps = round(abs(filled_price - exit_price) / exit_price * 10_000, 2)

            entry_price = row["entry_price"] or exit_price
            qty = row["qty"] or 0.0
            cost_basis = abs(entry_price * qty)
            pnl_pct = round(pnl / max(cost_basis, 0.0001) * 100, 4) if cost_basis else None

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
        slippage_bps: float | None = 0.0,
    ) -> bool:
        """
        Insert or complete a historical (backfilled) closed trade.

        Strategy:
        1. If an exact (symbol + entry_time) row already exists, skip.
        2. If an open (exit_time IS NULL) row exists for the same symbol on
           the same calendar day, complete it with exit data — this preserves
           all entry-level fields such as ml_prob and signal_score.
        3. Otherwise insert a new backfilled row.

        Returns True if a row was inserted or updated.
        """
        conn = _connect()
        try:
            # 1. Exact-match dedup (already backfilled from Alpaca)
            #    Also check same symbol + same calendar-day entry to catch
            #    duplicate backfills with slightly different timestamps.
            existing = conn.execute(
                "SELECT id FROM trades WHERE symbol=? AND entry_time=?",
                (symbol, entry_time),
            ).fetchone()
            if existing:
                return False

            entry_date_prefix = entry_time[:10] + "%"
            same_day_closed = conn.execute(
                "SELECT id FROM trades WHERE symbol=? AND entry_time LIKE ? "
                "AND exit_time IS NOT NULL ORDER BY id DESC LIMIT 1",
                (symbol, entry_date_prefix),
            ).fetchone()
            if same_day_closed:
                return False  # already have a closed trade for this symbol on this day

            pnl_pct = round(pnl / max(abs(entry_price * qty), 1e-9) * 100, 4) if entry_price and qty else None

            # 2. Try to complete an existing open entry recorded by the live bot
            #    on the same calendar day (entry_time[:10] == date prefix).
            entry_date_prefix = entry_time[:10] + "%"
            open_row = conn.execute(
                "SELECT id FROM trades WHERE symbol=? AND exit_time IS NULL "
                "AND entry_time LIKE ? ORDER BY id DESC LIMIT 1",
                (symbol, entry_date_prefix),
            ).fetchone()
            if open_row:
                conn.execute(
                    """
                    UPDATE trades SET
                        exit_time    = ?,
                        exit_price   = ?,
                        filled_price = COALESCE(filled_price, ?),
                        slippage_bps = COALESCE(slippage_bps, ?),
                        pnl          = ?,
                        pnl_pct      = ?,
                        hold_days    = ?
                    WHERE id = ?
                    """,
                    (
                        exit_time, exit_price,
                        exit_price,  # filled_price fallback
                        slippage_bps,
                        round(pnl, 4), pnl_pct, int(hold_days),
                        open_row["id"],
                    ),
                )
                conn.commit()
                return True

            # 3. No matching open entry — insert a brand-new backfilled row.
            conn.execute(
                """
                INSERT INTO trades
                    (symbol, side, entry_time, exit_time,
                     entry_price, exit_price, filled_price, qty,
                     pnl, pnl_pct, hold_days, order_type, slippage_bps)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    symbol, side, entry_time, exit_time,
                    entry_price, exit_price, exit_price, qty,
                    round(pnl, 4), pnl_pct, int(hold_days), "market",
                    slippage_bps,
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

    def get_open_trades(self) -> list[dict]:
        """Return all trades that have no exit recorded yet (exit_time IS NULL)."""
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT id, symbol, side, entry_time, entry_price, qty "
                "FROM trades WHERE exit_time IS NULL ORDER BY id ASC",
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

        # Drawdown from running equity high-water mark
        # Uses starting equity so drawdown % is relative to account size,
        # not cumulative PnL (which breaks when PnL peak is small).
        starting_eq = getattr(config, "STARTING_EQUITY", 1_000.0)
        equity = starting_eq
        hwm = equity  # high-water mark
        max_dd = 0.0
        for p in pnls:
            equity += p
            hwm = max(hwm, equity)
            dd = (hwm - equity) / hwm * 100 if hwm > 0 else 0.0
            max_dd = max(max_dd, dd)
        current_dd = (hwm - equity) / hwm * 100 if hwm > 0 else 0.0

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

    def get_trade_diagnostics(self) -> dict:
        """Return chart-ready diagnostics for analysing trade quality."""
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT symbol, side, entry_time, exit_time, signal_score, ml_prob, "
                "regime, pnl, pnl_pct, hold_days, exit_reason "
                "FROM trades WHERE exit_time IS NOT NULL AND pnl IS NOT NULL "
                "ORDER BY exit_time ASC"
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return {
                "summary": {
                    "total_trades": 0,
                    "expectancy_per_trade": None,
                    "median_pnl_pct": None,
                    "payoff_ratio": None,
                    "best_win_streak": 0,
                    "worst_loss_streak": 0,
                    "avg_win_hold_days": None,
                    "avg_loss_hold_days": None,
                    "profit_concentration_top5_pct": None,
                },
                "cumulative_pnl": [],
                "return_distribution": [],
                "exit_reasons": [],
                "score_buckets": [],
                "weekdays": [],
                "regimes": [],
                "insights": [],
            }

        trades = [dict(row) for row in rows]

        def _parse_iso(value: str | None) -> dt.datetime | None:
            if not value:
                return None
            try:
                return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                return None

        def _humanize(token: str | None) -> str:
            return (token or "unknown").replace("_", " ").strip().title()

        def _win_rate(items: list[dict]) -> float | None:
            if not items:
                return None
            return round(sum(1 for item in items if item["pnl"] > 0) / len(items) * 100, 1)

        def _avg(items: list[dict], key: str, digits: int = 2) -> float | None:
            values = [item[key] for item in items if item.get(key) is not None]
            if not values:
                return None
            return round(sum(values) / len(values), digits)

        def _score_bucket(score: int | float | None) -> str:
            if score is None:
                return "N/A"
            try:
                score_value = int(round(float(score)))
            except (TypeError, ValueError):
                return "N/A"
            if score_value <= 3:
                return "0-3"
            if score_value <= 5:
                return "4-5"
            if score_value <= 7:
                return "6-7"
            return "8+"

        for trade in trades:
            trade["pnl"] = float(trade.get("pnl") or 0.0)
            trade["pnl_pct"] = float(trade["pnl_pct"]) if trade.get("pnl_pct") is not None else None
            trade["hold_days"] = float(trade["hold_days"]) if trade.get("hold_days") is not None else None

        pnls = [trade["pnl"] for trade in trades]
        pnl_pcts = [trade["pnl_pct"] for trade in trades if trade["pnl_pct"] is not None]
        wins = [trade for trade in trades if trade["pnl"] > 0]
        losses = [trade for trade in trades if trade["pnl"] <= 0]

        avg_win = _avg(wins, "pnl")
        avg_loss = _avg(losses, "pnl")
        payoff_ratio = None
        if avg_win is not None and avg_loss not in (None, 0):
            payoff_ratio = round(avg_win / abs(avg_loss), 2)

        avg_win_hold = _avg(wins, "hold_days", digits=1)
        avg_loss_hold = _avg(losses, "hold_days", digits=1)

        best_win_streak = 0
        worst_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0
        running_pnl = 0.0
        cumulative_pnl = []

        for index, trade in enumerate(trades, start=1):
            if trade["pnl"] > 0:
                current_win_streak += 1
                current_loss_streak = 0
            else:
                current_loss_streak += 1
                current_win_streak = 0
            best_win_streak = max(best_win_streak, current_win_streak)
            worst_loss_streak = max(worst_loss_streak, current_loss_streak)

            running_pnl += trade["pnl"]
            exit_dt = _parse_iso(trade.get("exit_time")) or _parse_iso(trade.get("entry_time"))
            cumulative_pnl.append(
                {
                    "trade_number": index,
                    "symbol": trade.get("symbol"),
                    "date": exit_dt.strftime("%Y-%m-%d") if exit_dt else "",
                    "pnl": round(trade["pnl"], 2),
                    "cumulative_pnl": round(running_pnl, 2),
                }
            )

        gross_profit = sum(trade["pnl"] for trade in wins)
        top_winners = sorted((trade["pnl"] for trade in wins), reverse=True)[:5]
        profit_concentration = None
        if gross_profit > 0:
            profit_concentration = round(sum(top_winners) / gross_profit * 100, 1)

        distribution_specs = [
            ("<= -8%", None, -8.0),
            ("-8% to -4%", -8.0, -4.0),
            ("-4% to -1%", -4.0, -1.0),
            ("-1% to 1%", -1.0, 1.0),
            ("1% to 4%", 1.0, 4.0),
            ("4% to 8%", 4.0, 8.0),
            (">= 8%", 8.0, None),
        ]
        return_distribution = [
            {"label": label, "count": 0, "bucket_index": idx}
            for idx, (label, _lower, _upper) in enumerate(distribution_specs)
        ]
        for pct in pnl_pcts:
            for idx, (_label, lower, upper) in enumerate(distribution_specs):
                lower_ok = lower is None or pct >= lower
                upper_ok = upper is None or pct < upper
                if lower_ok and upper_ok:
                    return_distribution[idx]["count"] += 1
                    break

        exit_groups: dict[str, list[dict]] = defaultdict(list)
        score_groups: dict[str, list[dict]] = defaultdict(list)
        weekday_groups: dict[str, list[dict]] = defaultdict(list)
        regime_groups: dict[str, list[dict]] = defaultdict(list)

        for trade in trades:
            exit_groups[trade.get("exit_reason") or "unknown"].append(trade)
            score_groups[_score_bucket(trade.get("signal_score"))].append(trade)
            regime_groups[(trade.get("regime") or "unknown").lower()].append(trade)

            entry_dt = _parse_iso(trade.get("entry_time")) or _parse_iso(trade.get("exit_time"))
            weekday_label = entry_dt.strftime("%a") if entry_dt else "Unknown"
            weekday_groups[weekday_label].append(trade)

        exit_reasons = []
        for reason, items in exit_groups.items():
            exit_reasons.append(
                {
                    "reason": reason,
                    "label": _humanize(reason),
                    "trades": len(items),
                    "win_rate": _win_rate(items),
                    "avg_pnl": _avg(items, "pnl"),
                    "total_pnl": round(sum(item["pnl"] for item in items), 2),
                }
            )
        exit_reasons.sort(key=lambda row: (-row["trades"], row["total_pnl"]))

        score_bucket_order = {"0-3": 0, "4-5": 1, "6-7": 2, "8+": 3, "N/A": 4}
        score_buckets = []
        for bucket, items in score_groups.items():
            score_buckets.append(
                {
                    "bucket": bucket,
                    "trades": len(items),
                    "win_rate": _win_rate(items),
                    "avg_pnl": _avg(items, "pnl"),
                    "avg_pnl_pct": _avg(items, "pnl_pct"),
                }
            )
        score_buckets.sort(key=lambda row: score_bucket_order.get(row["bucket"], 99))

        weekday_order = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6, "Unknown": 7}
        weekdays = []
        for weekday, items in weekday_groups.items():
            weekdays.append(
                {
                    "weekday": weekday,
                    "trades": len(items),
                    "win_rate": _win_rate(items),
                    "avg_pnl": _avg(items, "pnl"),
                    "total_pnl": round(sum(item["pnl"] for item in items), 2),
                }
            )
        weekdays.sort(key=lambda row: weekday_order.get(row["weekday"], 99))

        regimes = []
        regime_order = {"bull": 0, "bear": 1, "inverse": 2, "unknown": 3}
        for regime, items in regime_groups.items():
            regimes.append(
                {
                    "regime": regime,
                    "label": _humanize(regime),
                    "trades": len(items),
                    "win_rate": _win_rate(items),
                    "avg_pnl": _avg(items, "pnl"),
                }
            )
        regimes.sort(key=lambda row: regime_order.get(row["regime"], 99))

        insights: list[str] = []
        if avg_win is not None and avg_loss is not None and abs(avg_loss) > avg_win:
            insights.append(
                f"Average loser (${abs(avg_loss):.2f}) is larger than average winner (${avg_win:.2f})."
            )
        if avg_loss_hold is not None and avg_win_hold is not None and avg_loss_hold > avg_win_hold:
            insights.append(
                f"Losing trades are held longer than winners ({avg_loss_hold:.1f}d vs {avg_win_hold:.1f}d)."
            )
        worst_exit = min(exit_reasons, key=lambda row: row["total_pnl"], default=None)
        if worst_exit and worst_exit["total_pnl"] < 0:
            insights.append(
                f"Most expensive exit path: {worst_exit['label']} ({worst_exit['total_pnl']:+.2f})."
            )
        scored_buckets = [row for row in score_buckets if row["bucket"] != "N/A" and row["trades"] >= 2 and row["avg_pnl_pct"] is not None]
        if scored_buckets:
            best_bucket = max(scored_buckets, key=lambda row: row["avg_pnl_pct"])
            insights.append(
                f"Best score bucket so far: {best_bucket['bucket']} (win rate {best_bucket['win_rate']:.1f}%, avg {best_bucket['avg_pnl_pct']:+.2f}%)."
            )
        weak_days = [row for row in weekdays if row["trades"] >= 2 and row["avg_pnl"] is not None]
        if weak_days:
            weakest_day = min(weak_days, key=lambda row: row["avg_pnl"])
            if weakest_day["avg_pnl"] < 0:
                insights.append(
                    f"Weakest entry day: {weakest_day['weekday']} ({weakest_day['avg_pnl']:+.2f} avg P&L)."
                )
        if profit_concentration is not None and profit_concentration >= 60:
            insights.append(
                f"Top 5 winners contribute {profit_concentration:.1f}% of gross profits."
            )

        return {
            "summary": {
                "total_trades": len(trades),
                "expectancy_per_trade": round(sum(pnls) / len(pnls), 2),
                "median_pnl_pct": round(statistics.median(pnl_pcts), 2) if pnl_pcts else None,
                "payoff_ratio": payoff_ratio,
                "best_win_streak": best_win_streak,
                "worst_loss_streak": worst_loss_streak,
                "avg_win_hold_days": avg_win_hold,
                "avg_loss_hold_days": avg_loss_hold,
                "profit_concentration_top5_pct": profit_concentration,
            },
            "cumulative_pnl": cumulative_pnl,
            "return_distribution": return_distribution,
            "exit_reasons": exit_reasons,
            "score_buckets": score_buckets,
            "weekdays": weekdays,
            "regimes": regimes,
            "insights": insights[:4],
        }
