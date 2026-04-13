from pathlib import Path

from trade_journal import TradeJournal
import trade_journal


def test_record_exit_uses_full_position_cost_for_pnl_pct(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(trade_journal, "DB_PATH", tmp_path / "trades.db")

    journal = TradeJournal()
    journal.record_entry(
        symbol="AAPL",
        side="long",
        entry_price=10.0,
        qty=5.0,
        stop_loss=9.0,
        take_profit=12.0,
        signal={"price": 10.0, "score": 5, "reason": "test"},
    )

    journal.record_exit(
        symbol="AAPL",
        exit_price=11.0,
        pnl=5.0,
        hold_days=1,
        exit_reason="target",
        filled_price=11.0,
    )

    closed = journal.get_closed_trades(limit=1)
    assert len(closed) == 1
    assert closed[0]["pnl_pct"] == 10.0


def test_trade_diagnostics_summarises_patterns(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(trade_journal, "DB_PATH", tmp_path / "trades.db")

    journal = TradeJournal()

    journal.record_entry(
        symbol="AAPL",
        side="long",
        entry_price=10.0,
        qty=1.0,
        stop_loss=9.0,
        take_profit=12.0,
        signal={"price": 10.0, "score": 5, "reason": "trend pullback"},
        regime="bull",
    )
    journal.record_exit(
        symbol="AAPL",
        exit_price=12.0,
        pnl=2.0,
        hold_days=2,
        exit_reason="take_profit",
        filled_price=12.0,
    )

    journal.record_entry(
        symbol="MSFT",
        side="long",
        entry_price=20.0,
        qty=1.0,
        stop_loss=18.0,
        take_profit=24.0,
        signal={"price": 20.0, "score": 3, "reason": "weak setup"},
        regime="bull",
    )
    journal.record_exit(
        symbol="MSFT",
        exit_price=18.0,
        pnl=-2.0,
        hold_days=4,
        exit_reason="stop_loss",
        filled_price=18.0,
    )

    journal.record_entry(
        symbol="TSLA",
        side="short",
        entry_price=30.0,
        qty=2.0,
        stop_loss=33.0,
        take_profit=27.0,
        signal={"price": 30.0, "score": 7, "reason": "bear continuation"},
        regime="bear",
    )
    journal.record_exit(
        symbol="TSLA",
        exit_price=27.0,
        pnl=6.0,
        hold_days=1,
        exit_reason="take_profit",
        filled_price=27.0,
    )

    diagnostics = journal.get_trade_diagnostics()

    assert diagnostics["summary"]["total_trades"] == 3
    assert diagnostics["summary"]["expectancy_per_trade"] == 2.0
    assert diagnostics["summary"]["payoff_ratio"] == 2.0
    assert diagnostics["summary"]["best_win_streak"] == 1
    assert diagnostics["summary"]["worst_loss_streak"] == 1

    exit_totals = {row["reason"]: row["total_pnl"] for row in diagnostics["exit_reasons"]}
    assert exit_totals["take_profit"] == 8.0
    assert exit_totals["stop_loss"] == -2.0

    score_rows = {row["bucket"]: row["trades"] for row in diagnostics["score_buckets"]}
    assert score_rows["0-3"] == 1
    assert score_rows["4-5"] == 1
    assert score_rows["6-7"] == 1

    weekday_trades = sum(row["trades"] for row in diagnostics["weekdays"])
    assert weekday_trades == 3
    assert sum(bucket["count"] for bucket in diagnostics["return_distribution"]) == 3
