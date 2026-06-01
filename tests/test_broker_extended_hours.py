import datetime as dt
from types import SimpleNamespace

import config
from broker import AlpacaBroker


class DummyApi:
    def __init__(self):
        self.orders = []

    def submit_order(self, **kwargs):
        self.orders.append(kwargs)
        return SimpleNamespace(id=str(len(self.orders)), **kwargs)

    def get_latest_quote(self, symbol):
        return SimpleNamespace(bidprice="99.80", askprice="100.00")

    def get_latest_trade(self, symbol):
        return SimpleNamespace(price="99.90")


def test_submit_market_buy_uses_extended_hours_limit_order(monkeypatch):
    monkeypatch.setattr(config, "EXTENDED_HOURS_TRADING", True)
    monkeypatch.setattr(config, "USE_LIMIT_ORDERS", False)
    monkeypatch.setattr(config, "FRACTIONAL_SHARES", True)
    monkeypatch.setattr(config, "LIMIT_ORDER_TIF", "day")
    monkeypatch.setattr(config, "EXTENDED_HOURS_LIMIT_OFFSET_PCT", 0.003)

    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker.api = DummyApi()
    broker._should_use_extended_hours_orders = lambda: True

    broker.submit_market_buy("AAPL", 1.25, stop_loss=95.0, take_profit=110.0)

    assert len(broker.api.orders) == 1
    order = broker.api.orders[0]
    assert order["side"] == "buy"
    assert order["type"] == "limit"
    assert order["time_in_force"] == "day"
    assert order["extended_hours"] is True
    assert order["limit_price"] == 100.3


def test_submit_market_sell_uses_extended_hours_limit_order(monkeypatch):
    monkeypatch.setattr(config, "EXTENDED_HOURS_TRADING", True)
    monkeypatch.setattr(config, "LIMIT_ORDER_TIF", "day")
    monkeypatch.setattr(config, "EXTENDED_HOURS_LIMIT_OFFSET_PCT", 0.003)

    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker.api = DummyApi()
    broker._should_use_extended_hours_orders = lambda: True

    broker.submit_market_sell("AAPL", 2)

    assert len(broker.api.orders) == 1
    order = broker.api.orders[0]
    assert order["side"] == "sell"
    assert order["type"] == "limit"
    assert order["time_in_force"] == "day"
    assert order["extended_hours"] is True
    assert order["limit_price"] == 99.5


def test_get_trading_session_recognizes_sunday_overnight_session(monkeypatch):
    monkeypatch.setattr(config, "EXTENDED_HOURS_TRADING", True)

    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._get_clock_with_retry = lambda: SimpleNamespace(
        is_open=False,
        timestamp=dt.datetime(2026, 5, 31, 20, 13, tzinfo=dt.timezone(dt.timedelta(hours=-4))),
    )

    assert broker.get_trading_session() == "overnight"
    assert broker.is_trading_session_open() is True