"""
Swing Trading Bot – Main Entry Point

Usage:
    python main.py              # run continuously during market hours
    python main.py --once       # run a single cycle and exit
    python main.py --status     # print account & position status
    python main.py --backtest   # run backtest (pass-through to backtest.py)
"""

from __future__ import annotations

import argparse
import datetime as dt
import signal
import sys
import time

import schedule

import config
from broker import AlpacaBroker
from executor import TradeExecutor
from logger import get_logger

log = get_logger("main")

# Global reference for graceful shutdown
_executor: TradeExecutor | None = None


def _graceful_shutdown(signum, frame):
    """Save PDT ledger and exit cleanly on SIGTERM/SIGINT."""
    sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    log.info("Received %s – saving PDT ledger and shutting down...", sig_name)
    if _executor is not None:
        try:
            _executor.pdt._save()
            log.info("PDT ledger saved successfully")
        except Exception as exc:
            log.error("Failed to save PDT ledger: %s", exc)
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGTERM, _graceful_shutdown)
signal.signal(signal.SIGINT, _graceful_shutdown)


def print_status() -> None:
    """Print a quick account & positions summary."""
    broker = AlpacaBroker()
    acct = broker.get_account()
    positions = broker.get_positions()
    clock = broker.get_clock()

    print("\n" + "=" * 60)
    print("  SWING TRADING BOT – STATUS")
    print("=" * 60)
    print(f"  Mode           : {config.TRADING_MODE.upper()}")
    print(f"  Market open    : {clock.is_open}")
    print(f"  Next open      : {clock.next_open}")
    print(f"  Next close     : {clock.next_close}")
    print(f"  Equity         : ${float(acct.equity):>12,.2f}")
    print(f"  Cash           : ${float(acct.cash):>12,.2f}")
    print(f"  Buying power   : ${float(acct.buying_power):>12,.2f}")
    print(f"  Day-trade count: {acct.daytrade_count}")
    print(f"  Open positions : {len(positions)}")
    print("-" * 60)

    if positions:
        print(f"  {'Symbol':<8} {'Qty':>6} {'Entry':>10} {'Current':>10} {'P&L':>10} {'P&L%':>8}")
        print("  " + "-" * 54)
        for p in positions:
            sym = p.symbol
            qty = int(p.qty)
            entry = float(p.avg_entry_price)
            cur = float(p.current_price)
            pnl = float(p.unrealized_pl)
            pnl_pct = float(p.unrealized_plpc) * 100
            print(f"  {sym:<8} {qty:>6} {entry:>10.2f} {cur:>10.2f} {pnl:>+10.2f} {pnl_pct:>+7.2f}%")
    else:
        print("  (no open positions)")

    print("=" * 60 + "\n")


def run_once() -> None:
    """Run a single scan cycle."""
    global _executor
    log.info("Running single cycle...")
    _executor = TradeExecutor()
    _executor.run_cycle()
    log.info("Single cycle complete.")


def run_loop() -> None:
    """
    Run the bot in a continuous loop using `schedule`.
    - Morning tasks (stop-loss refresh) fire once per day shortly after open
    - Exits are checked more frequently (every 15 min)
    - Full entry scans happen every 30 min
    """
    log.info("=" * 60)
    log.info("  SWING TRADING BOT STARTED")
    log.info("  Mode: %s", config.TRADING_MODE.upper())
    log.info("  Watchlist: %d symbols", len(config.WATCHLIST))
    log.info("  Entry scan every %d min", config.SCAN_INTERVAL_MINUTES)
    log.info("  Exit check every %d min", config.CHECK_EXITS_MINUTES)
    log.info("  RE_ENTRY_COOLDOWN_DAYS: %d", config.RE_ENTRY_COOLDOWN_DAYS)
    log.info("  MAX_DAY_TRADES_ALLOWED: %d", config.MAX_DAY_TRADES_ALLOWED)
    log.info("  PDT_LOOKBACK_DAYS: %d", config.PDT_LOOKBACK_DAYS)
    log.info("=" * 60)

    executor = TradeExecutor()
    global _executor
    _executor = executor
    _morning_done_date: list[dt.date] = [None]  # mutable container for closure
    _last_full_cycle_time: list[float] = [0.0]   # timestamp of last full_cycle run

    def morning_job():
        try:
            today = dt.date.today()
            if _morning_done_date[0] == today:
                return  # already ran today
            if executor.broker.is_market_open():
                log.info("Running morning tasks for %s", today)
                executor.morning_tasks()
                _morning_done_date[0] = today
        except Exception as exc:
            log.error("morning_job failed (will retry next cycle): %s", exc, exc_info=True)

    def exit_check():
        try:
            # Skip if a full_cycle ran very recently (within 60s) to avoid
            # double exit scans that could submit duplicate sell orders.
            if time.time() - _last_full_cycle_time[0] < 60:
                return
            if executor.broker.is_market_open():
                executor.refresh()
                executor.scan_exits()
        except Exception as exc:
            log.error("exit_check failed (will retry next cycle): %s", exc, exc_info=True)

    def full_cycle():
        try:
            executor.run_cycle()
            _last_full_cycle_time[0] = time.time()
        except Exception as exc:
            log.error("full_cycle failed (will retry next cycle): %s", exc, exc_info=True)

    # Schedule jobs
    schedule.every(5).minutes.do(morning_job)          # poll until market opens & runs once
    schedule.every(config.CHECK_EXITS_MINUTES).minutes.do(exit_check)
    schedule.every(config.SCAN_INTERVAL_MINUTES).minutes.do(full_cycle)

    # Run the first cycle immediately
    morning_job()
    full_cycle()

    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(30)  # sleep 30s between scheduler ticks


def main() -> None:
    parser = argparse.ArgumentParser(description="Swing Trading Bot (Alpaca)")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--status", action="store_true", help="Print account status")
    parser.add_argument("--backtest", action="store_true", help="Run backtester")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols for backtest")
    parser.add_argument("--months", type=int, default=12, help="Backtest lookback months")
    args = parser.parse_args()

    if not config.ALPACA_API_KEY or config.ALPACA_API_KEY == "your_api_key_here":
        print("\n  ERROR: Set your Alpaca API keys in .env (see .env.example)\n")
        sys.exit(1)

    if args.backtest:
        from backtest import Backtester
        import datetime as _dt
        symbols = args.symbols or ["SPY"]
        end = _dt.date.today()
        start = end - _dt.timedelta(days=args.months * 30)
        bt = Backtester(symbols=symbols, start_date=start, end_date=end)
        bt.run()
        bt.save_chart()
    elif args.status:
        print_status()
    elif args.once:
        run_once()
    else:
        run_loop()


if __name__ == "__main__":
    main()
