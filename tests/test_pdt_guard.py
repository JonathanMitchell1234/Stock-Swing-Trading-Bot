import datetime as dt
from types import SimpleNamespace
from pathlib import Path

import pdt_guard
from pdt_guard import PDTGuard


class FakeBroker:
    def __init__(self, positions, orders):
        self._positions = positions
        self._orders = orders

    def get_closed_orders(self, after=None, limit=500):
        return self._orders

    def get_positions(self):
        return self._positions


def test_can_sell_today_handles_timezone_aware_buy_time(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(pdt_guard, "LEDGER_PATH", tmp_path / "pdt_ledger.json")
    guard = PDTGuard()
    today = dt.date.today().isoformat()
    guard._ledger["AAPL"] = today
    aware_buy_time = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=61)
    guard._buy_times["AAPL"] = aware_buy_time.isoformat()

    assert guard.can_sell_today("AAPL") is True


def test_reconcile_recovers_held_short_position(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(pdt_guard, "LEDGER_PATH", tmp_path / "pdt_ledger.json")

    filled_at = dt.datetime(2026, 4, 1, 14, 30, tzinfo=dt.timezone.utc)
    orders = [
        SimpleNamespace(
            status="filled",
            symbol="TSLA",
            side="sell",
            filled_at=filled_at,
        )
    ]
    positions = [SimpleNamespace(symbol="TSLA", qty="-3")]

    guard = PDTGuard(broker=FakeBroker(positions=positions, orders=orders))

    assert guard._ledger["TSLA"] == "2026-04-01"
    assert guard._buy_times["TSLA"] == filled_at.isoformat()
