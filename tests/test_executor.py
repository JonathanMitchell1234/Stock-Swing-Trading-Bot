import datetime as dt
from types import SimpleNamespace

import config
from executor import TradeExecutor


class DummyPDT:
    def __init__(self):
        self.calls = []
        self.sell_calls = []
        self._buy_times = {}

    def open_symbols(self):
        return []

    def record_buy(self, symbol, fill_date=None):
        self.calls.append((symbol, fill_date))

    def record_sell(self, symbol):
        self.sell_calls.append(symbol)

    def _save(self):
        return None


class DummyJournal:
    def __init__(self):
        self.entries = []
        self.exits = []

    def get_open_trades(self):
        return []

    def record_entry(self, *args, **kwargs):
        self.entries.append((args, kwargs))

    def record_exit(self, *args, **kwargs):
        self.exits.append((args, kwargs))


class DummyBroker:
    def __init__(self, positions=None, orders=None):
        self._positions = positions or []
        self._orders = orders or []
        self.api = SimpleNamespace(list_orders=self._list_orders)

    def _list_orders(self, **kwargs):
        return self._orders

    def get_positions(self):
        return self._positions

    def get_position(self, symbol):
        for pos in self._positions:
            if pos.symbol == symbol:
                return pos
        return None


def test_finalize_entry_defers_accounting_until_fill_exists():
    executor = TradeExecutor.__new__(TradeExecutor)
    executor.broker = DummyBroker()
    executor.pdt = DummyPDT()
    executor.journal = DummyJournal()
    executor.risk = SimpleNamespace(open_positions=0)
    executor._sector_counts = {}
    executor._poll_fill_price = lambda order: None
    executor._get_recent_fill = lambda symbol, side: (None, None)

    opened = executor._finalize_entry(
        symbol="AAPL",
        sector="Tech",
        side="long",
        requested_entry_price=100.0,
        qty=1.0,
        stop_loss=95.0,
        take_profit=110.0,
        signal={"price": 100.0, "score": 5, "reason": "test"},
        regime="bull",
        order_type="market",
        vwap_used=False,
        order=SimpleNamespace(id="1"),
    )

    assert opened == 0
    assert executor.pdt.calls == []
    assert executor.journal.entries == []


def test_reconcile_open_positions_backfills_missing_entry(monkeypatch):
    monkeypatch.setattr(config, "INVERSE_WATCHLIST", [])

    fill_dt = dt.datetime(2026, 4, 2, 14, 30, tzinfo=dt.timezone.utc)
    order = SimpleNamespace(side="buy", filled_at=fill_dt, filled_avg_price="101.5")
    position = SimpleNamespace(symbol="AAPL", qty="2", avg_entry_price="101.5")

    executor = TradeExecutor.__new__(TradeExecutor)
    executor.broker = DummyBroker(positions=[position], orders=[order])
    executor.pdt = DummyPDT()
    executor.journal = DummyJournal()

    executor._reconcile_open_positions([position])

    assert executor.pdt.calls == [("AAPL", dt.date(2026, 4, 2))]
    assert executor.journal.entries
    args, kwargs = executor.journal.entries[0]
    assert args[0] == "AAPL"
    assert args[1] == "long"
    assert args[2] == 101.5
    assert kwargs["order_type"] == "reconciled"


def test_get_recent_fill_parses_nanosecond_timestamp():
    order = SimpleNamespace(
        side="buy",
        filled_at="2026-04-13T14:45:19.251336672+00:00",
        filled_avg_price="101.5",
    )

    executor = TradeExecutor.__new__(TradeExecutor)
    executor.broker = DummyBroker(orders=[order])

    fill_dt, fill_price = executor._get_recent_fill("AAPL", "buy")

    assert fill_dt == dt.datetime(2026, 4, 13, 14, 45, 19, 251336, tzinfo=dt.timezone.utc)
    assert fill_price == 101.5


def test_finalize_exit_defers_accounting_until_position_is_closed():
    position = SimpleNamespace(symbol="AAPL", qty="1", avg_entry_price="100")

    executor = TradeExecutor.__new__(TradeExecutor)
    executor.broker = DummyBroker(positions=[position])
    executor.pdt = DummyPDT()
    executor.journal = DummyJournal()
    executor._poll_fill_price = lambda order: None
    executor._get_recent_fill = lambda symbol, side: (None, None)

    closed = executor._finalize_exit(
        symbol="AAPL",
        side="long",
        qty=1.0,
        entry_price=100.0,
        requested_exit_price=98.0,
        hold_days=3,
        exit_reason="signal",
        order=SimpleNamespace(id="1"),
    )

    assert closed == 0
    assert executor.pdt.sell_calls == []
    assert executor.journal.exits == []


def test_run_cycle_executes_during_after_hours_when_enabled(monkeypatch):
    monkeypatch.setattr(config, "EXTENDED_HOURS_TRADING", True)

    executor = TradeExecutor.__new__(TradeExecutor)
    executor.broker = SimpleNamespace(
        get_trading_session=lambda: "afterhours",
        get_equity=lambda: 1000.0,
        get_positions=lambda: [],
    )

    calls = []
    executor.scan_exits = lambda: calls.append("exits")
    executor.scan_entries = lambda: calls.append("entries")

    executor.run_cycle()

    assert calls == ["exits", "entries"]
