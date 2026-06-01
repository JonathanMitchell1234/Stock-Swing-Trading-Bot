import signal
from types import SimpleNamespace

import pytest

import main


def test_graceful_shutdown_stops_news_monitor_and_saves_pdt(monkeypatch):
    calls = []

    main._news_monitor = SimpleNamespace(stop=lambda: calls.append("stop"))
    main._executor = SimpleNamespace(pdt=SimpleNamespace(_save=lambda: calls.append("save")))

    with pytest.raises(SystemExit) as exc:
        main._graceful_shutdown(signal.SIGTERM, None)

    assert exc.value.code == 0
    assert calls == ["stop", "save"]


def test_should_trigger_session_open_cycle_only_on_closed_to_open_transition():
    assert main._should_trigger_session_open_cycle(False, True) is True
    assert main._should_trigger_session_open_cycle(True, True) is False
    assert main._should_trigger_session_open_cycle(False, False) is False
    assert main._should_trigger_session_open_cycle(None, True) is False