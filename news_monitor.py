"""
Live News Monitor — Catalyst Ejection Shield

Connects to Alpaca's real-time news WebSocket in a background daemon thread.
For every incoming headline, if any mentioned symbol matches an open position,
FinBERT scores the headline.  If the score drops below NLP_EJECTION_THRESHOLD
(default -0.80 — indicating FDA rejection / missed earnings / C-suite scandal),
the bot immediately submits a market sell order, exiting before the MACD or
EMA crossovers have time to register the crash.

Configuration keys (config.py / overrides JSON):
  NLP_NEWS_EJECTION_ENABLED  : bool  — master on/off switch (default False)
  NLP_EJECTION_THRESHOLD     : float — FinBERT score floor  (default -0.80)
  NLP_EJECTION_COOLDOWN_SECS : int   — seconds between re-ejections per symbol (default 300)
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import TYPE_CHECKING

import config
from logger import get_logger

if TYPE_CHECKING:
    from broker import AlpacaBroker

log = get_logger("news_monitor")

# Alpaca news stream endpoint (same URL for paper + live — data tier is separate)
_WS_URL = "wss://stream.data.alpaca.markets/v1beta1/news"

_RECONNECT_BASE_DELAY = 5.0    # seconds before first reconnect attempt
_RECONNECT_MAX_DELAY  = 120.0  # exponential-backoff ceiling


class NewsMonitor:
    """
    Background Alpaca news WebSocket listener that acts as a defensive
    catalyst-ejection shield.

    Usage::

        monitor = NewsMonitor(broker)
        monitor.start()   # non-blocking — spawns a daemon thread
        ...
        monitor.stop()
    """

    def __init__(self, broker: "AlpacaBroker") -> None:
        self._broker = broker
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # Per-symbol cooldown — maps symbol -> timestamp of last ejection
        self._ejected_at: dict[str, float] = {}
        # Lock protecting _ejected_at from concurrent writes
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public interface                                                      #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the monitor.  No-op if disabled in config or already running."""
        if not getattr(config, "NLP_NEWS_EJECTION_ENABLED", False):
            log.info("News ejection shield disabled (NLP_NEWS_EJECTION_ENABLED=False)")
            return
        if not getattr(config, "NLP_SENTIMENT_ENABLED", False):
            log.info("News ejection shield requires NLP_SENTIMENT_ENABLED=True — skipping")
            return
        if self._thread is not None and self._thread.is_alive():
            log.debug("News monitor already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_event_loop,
            name="news-monitor",
            daemon=True,
        )
        self._thread.start()
        log.info(
            "News ejection shield started  threshold=%.2f  cooldown=%ds",
            float(getattr(config, "NLP_EJECTION_THRESHOLD", -0.80)),
            int(getattr(config, "NLP_EJECTION_COOLDOWN_SECS", 300)),
        )

    def stop(self) -> None:
        """Signal the monitor to stop and wait for its thread to finish."""
        self._stop_event.set()
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------ #
    # Thread entry point — owns a private asyncio event loop              #
    # ------------------------------------------------------------------ #

    def _run_event_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._stream_with_reconnect())
        except Exception as exc:
            log.error("News monitor event loop crashed: %s", exc, exc_info=True)
        finally:
            try:
                self._loop.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # WebSocket lifecycle                                                  #
    # ------------------------------------------------------------------ #

    async def _stream_with_reconnect(self) -> None:
        """Outer reconnect loop — re-establishes the stream after any failure."""
        delay = _RECONNECT_BASE_DELAY
        while not self._stop_event.is_set():
            try:
                await self._connect_and_stream()
                delay = _RECONNECT_BASE_DELAY  # reset on clean disconnect
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                log.warning(
                    "News stream disconnected (%s) — reconnecting in %.0fs",
                    exc, delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, _RECONNECT_MAX_DELAY)

    async def _connect_and_stream(self) -> None:
        """One WebSocket session: auth → subscribe → message loop."""
        try:
            import websockets  # type: ignore[import]
        except ImportError:
            log.error(
                "'websockets' package missing — install it: pip install websockets"
            )
            self._stop_event.set()
            return

        api_key = config.ALPACA_API_KEY
        secret  = config.ALPACA_SECRET_KEY

        async with websockets.connect(
            _WS_URL,
            ping_interval=20,
            ping_timeout=30,
            close_timeout=10,
        ) as ws:
            # 1. Authenticate
            await ws.send(json.dumps({"action": "auth", "key": api_key, "secret": secret}))
            auth_resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=15.0))
            if not self._is_authenticated(auth_resp):
                log.error("News stream auth failed: %s", auth_resp)
                return

            # 2. Subscribe to all news (we filter down to open positions in the handler)
            await ws.send(json.dumps({"action": "subscribe", "news": ["*"]}))
            sub_resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=15.0))
            log.info("News stream live  sub_ack=%s", sub_resp)

            # 3. Message loop
            while not self._stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
                except asyncio.TimeoutError:
                    # Heartbeat gap — loop back and poll stop_event
                    continue

                messages = json.loads(raw)
                if not isinstance(messages, list):
                    messages = [messages]

                for msg in messages:
                    if msg.get("T") == "n":  # "n" = news article event
                        # Run FinBERT in a thread-pool worker so it never
                        # blocks the asyncio recv loop
                        self._loop.run_in_executor(None, self._handle_news, msg)

    # ------------------------------------------------------------------ #
    # News handler — runs in a thread-pool worker                         #
    # ------------------------------------------------------------------ #

    def _handle_news(self, msg: dict) -> None:
        """Score a headline and eject any affected positions if strongly negative."""
        headline = msg.get("headline", "").strip()
        symbols  = msg.get("symbols", [])

        if not headline or not symbols:
            return

        threshold = float(getattr(config, "NLP_EJECTION_THRESHOLD", -0.80))
        cooldown  = int(getattr(config, "NLP_EJECTION_COOLDOWN_SECS", 300))

        # Fast path: fetch open positions and intersect with headline symbols
        try:
            open_positions = {p.symbol: p for p in self._broker.get_positions()}
        except Exception as exc:
            log.debug("get_positions failed in news handler: %s", exc)
            return

        affected = [s for s in symbols if s in open_positions]
        if not affected:
            return  # headline doesn't touch any holdings — quick exit

        log.info(
            "NEWS HIT  positions=%s  headline=%.120s",
            affected, headline,
        )

        # Score with FinBERT (reuses singleton already loaded by main thread)
        try:
            from sentiment import get_sentiment
            score = get_sentiment([headline])
        except Exception as exc:
            log.warning("FinBERT scoring failed in news handler: %s", exc)
            return

        log.info(
            "NEWS SENTIMENT  positions=%s  score=%.3f  threshold=%.2f",
            affected, score, threshold,
        )

        if score >= threshold:
            return  # not catastrophic enough — stand down

        # ── Trigger ejection for each affected symbol ──────────────────
        now = time.time()
        for symbol in affected:
            with self._lock:
                last_ejection = self._ejected_at.get(symbol, 0.0)
                if now - last_ejection < cooldown:
                    log.info(
                        "EJECTION suppressed for %s (cooldown active — %.0fs since last ejection)",
                        symbol, now - last_ejection,
                    )
                    continue
                # Stamp before the API call to prevent a parallel duplicate
                self._ejected_at[symbol] = now

            log.warning(
                "CATALYST EJECTION TRIGGERED  %s  score=%.3f  headline=%.120s",
                symbol, score, headline,
            )
            self._eject(symbol, score)

    def _eject(self, symbol: str, score: float) -> None:
        """Issue an immediate market exit for the position."""
        try:
            pos = self._broker.get_position(symbol)
            if pos is None:
                log.info("Ejection: %s no longer has an open position — skipping", symbol)
                return

            qty = float(pos.qty)
            abs_qty = abs(qty)

            if qty > 0:
                # Long position — market sell
                log.warning(
                    "EJECTING LONG  %s  qty=%.4f  score=%.3f",
                    symbol, abs_qty, score,
                )
                self._broker.submit_market_sell(symbol, abs_qty)
            else:
                # Short position — buy-to-cover
                log.warning(
                    "EJECTING SHORT (buy-to-cover)  %s  qty=%.4f  score=%.3f",
                    symbol, abs_qty, score,
                )
                self._broker.submit_market_cover(symbol, abs_qty)

            log.warning("EJECTION ORDER SUBMITTED  %s", symbol)

        except Exception as exc:
            log.error("Ejection order failed for %s: %s", symbol, exc, exc_info=True)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_authenticated(resp) -> bool:
        if not isinstance(resp, list):
            resp = [resp]
        for m in resp:
            if m.get("msg") in ("authenticated", "already authenticated"):
                return True
            if m.get("T") == "success":
                return True
        return False
