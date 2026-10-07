import argparse
import asyncio
import logging
import signal

import pandas as pd

from src.backtest import run_backtest
from src.core.config import settings
from src.core.db import SessionLocal, engine
from src.core.recovery import load_active_positions
from src.core.risk import RiskManager
from src.data.candles import backfill, ingest_live, load_candles
from src.data.gateway import ExchangeGateway
from src.strategies import STRATEGIES

log = logging.getLogger("trading-bot")
# ~4 years: spans the 2022 bear, 2023 chop and 2024-25 bull, so a strategy is judged across regimes.
HISTORY_DAYS = 1460


async def startup() -> None:
    async with SessionLocal() as session:
        positions = await load_active_positions(session)
    log.info("recovered %d active position(s)", len(positions))
    for (strategy, symbol, timeframe), p in positions.items():
        log.info("  %s %s %s %s @ %s", strategy, symbol, timeframe, p.side, p.entry_price)


async def on_candle_close(symbol: str, timeframe: str, closed: pd.DataFrame) -> None:
    # TODO(Phase 5): evaluate -> risk -> save position -> dispatch. Timing gotcha: evaluate()
    # shifts indicators, so row t holds the signal computed from data through t-1. For the
    # candle that just closed (t) call `strategy.triggers(strategy.indicators(candles))`
    # (unshifted) and read its last row, or every alert fires one bar late. Size with
    # `RiskManager.plan(entry=gw.prices[symbol], atr=risk.atr(candles).iloc[-1], equity=...)`,
    # entry being the next bar's open ~ live price, same as the backtest.
    log.info(
        "%s %s closed %d candle(s), last close %s",
        symbol,
        timeframe,
        len(closed),
        closed["close"].iloc[-1],
    )


async def run_backfill(symbol: str, timeframe: str, days: int) -> None:
    gw = ExchangeGateway()
    try:
        async with SessionLocal() as session:
            await backfill(gw, session, symbol, timeframe, days)
    finally:
        await gw.close()


async def run_backtest_cmd(args: argparse.Namespace) -> None:
    await run_backfill(args.symbol, args.timeframe, args.days)  # make sure history is present
    async with SessionLocal() as session:
        candles = await load_candles(session, args.symbol, args.timeframe, args.days)
    strategy = STRATEGIES[args.strategy]()
    risk = None if args.no_risk else RiskManager(risk_pct=settings.risk_per_trade)
    metrics = run_backtest(strategy, candles, args.timeframe, fees=args.fees, risk=risk)
    mode = "no stops, all-in" if risk is None else f"ATR stops, {risk.risk_pct:.1%} risk/trade"
    print(f"\n{strategy.name} | {args.symbol} {args.timeframe} | {len(candles)} candles | {mode}")
    print(f"{candles.index[0]} -> {candles.index[-1]}")
    for key, value in metrics.items():
        print(f"  {key:<18} {value:>10.2f}")


async def run_monitor(symbol: str, timeframe: str, days: int) -> None:
    await startup()
    gw = ExchangeGateway()
    try:
        async with SessionLocal() as session:
            await backfill(gw, session, symbol, timeframe, days)  # fill any gap since last run
        async with asyncio.TaskGroup() as tg:
            tg.create_task(gw.stream_prices(symbol))
            tg.create_task(ingest_live(gw, SessionLocal, symbol, timeframe, on_candle_close))
    finally:
        await gw.close()


async def _run(args: argparse.Namespace) -> None:
    task = asyncio.current_task()
    assert task is not None
    # SIGTERM (docker stop, kill) cancels like Ctrl-C so the finally blocks close sockets/DB.
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        if args.command == "recover":
            await startup()
        elif args.command == "backfill":
            for tf in args.timeframe:
                await run_backfill(args.symbol, tf, args.days)
        elif args.command == "backtest":
            await run_backtest_cmd(args)
        elif args.command == "monitor":
            await run_monitor(args.symbol, args.timeframe, args.days)
    finally:
        await engine.dispose()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(prog="trading-bot")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("recover", help="Load active positions from the DB (crash recovery)")
    bf = sub.add_parser("backfill", help="Seed historical candles via REST")
    bf.add_argument("--symbol", default="BTC/USDT")
    bf.add_argument("--timeframe", nargs="+", default=["15m", "1h"])
    bf.add_argument("--days", type=int, default=HISTORY_DAYS)
    bt = sub.add_parser("backtest", help="Backtest a strategy on stored candles")
    bt.add_argument("--strategy", choices=STRATEGIES, default="DoubleEma")
    bt.add_argument("--symbol", default="BTC/USDT")
    bt.add_argument("--timeframe", default="1h")
    bt.add_argument("--days", type=int, default=HISTORY_DAYS)
    bt.add_argument("--no-risk", action="store_true", help="Disable SL/TP and risk sizing")
    bt.add_argument("--fees", type=float, default=0.001, help="Fee per fill (0.001 = 0.1%%)")
    mon = sub.add_parser("monitor", help="Stream live candles into the DB")
    mon.add_argument("--symbol", default="BTC/USDT")
    mon.add_argument("--timeframe", default="15m")
    mon.add_argument("--days", type=int, default=60, help="Backfill depth if DB is empty")
    try:
        asyncio.run(_run(parser.parse_args()))
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("shut down")


if __name__ == "__main__":
    main()
