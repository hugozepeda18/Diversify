import argparse
import asyncio
import logging

import pandas as pd

from src.core.db import SessionLocal, engine
from src.core.recovery import load_active_positions
from src.data.candles import backfill, ingest_live
from src.data.gateway import ExchangeGateway

log = logging.getLogger("trading-bot")


async def startup() -> None:
    async with SessionLocal() as session:
        positions = await load_active_positions(session)
    log.info("recovered %d active position(s)", len(positions))
    for (strategy, symbol, timeframe), p in positions.items():
        log.info("  %s %s %s %s @ %s", strategy, symbol, timeframe, p.side, p.entry_price)


async def on_candle_close(symbol: str, timeframe: str, closed: pd.DataFrame) -> None:
    # ponytail: Phase 3 strategy engine plugs in here.
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
    try:
        if args.command == "recover":
            await startup()
        elif args.command == "backfill":
            for tf in args.timeframe:
                await run_backfill(args.symbol, tf, args.days)
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
    bf.add_argument("--days", type=int, default=60)
    mon = sub.add_parser("monitor", help="Stream live candles into the DB")
    mon.add_argument("--symbol", default="BTC/USDT")
    mon.add_argument("--timeframe", default="15m")
    mon.add_argument("--days", type=int, default=60, help="Backfill depth if DB is empty")
    try:
        asyncio.run(_run(parser.parse_args()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
