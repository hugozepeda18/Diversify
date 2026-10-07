import argparse
import asyncio

from src.core.db import SessionLocal, engine
from src.core.recovery import load_active_positions


async def startup() -> None:
    async with SessionLocal() as session:
        positions = await load_active_positions(session)
    await engine.dispose()
    print(f"Recovered {len(positions)} active position(s)")
    for (strategy, symbol, timeframe), p in positions.items():
        print(f"  {strategy} {symbol} {timeframe} {p.side} @ {p.entry_price}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="trading-bot")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("recover", help="Load active positions from the DB (crash recovery)")
    args = parser.parse_args()
    if args.command == "recover":
        asyncio.run(startup())


if __name__ == "__main__":
    main()
