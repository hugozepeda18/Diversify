import argparse
import asyncio
import json
import logging
import signal

import pandas as pd

from src.backtest import optimize, run_backtest
from src.core.config import settings
from src.core.db import SessionLocal, engine
from src.core.recovery import load_active_positions
from src.core.risk import RiskManager
from src.data.candles import backfill, ingest_live, load_candles
from src.data.gateway import ExchangeGateway
from src.signals import process_close
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


async def run_optimize_cmd(args: argparse.Namespace) -> None:
    cols = ["sharpe_ratio", "total_return_pct", "max_drawdown_pct", "trades"]
    for tf in args.timeframe:
        await run_backfill(args.symbol, tf, args.days)
        async with SessionLocal() as session:
            candles = await load_candles(session, args.symbol, tf, args.days)
        res = optimize(STRATEGIES[args.strategy], candles, tf, args.split, args.fees)
        bh = {w: res[f"{w}_buy_hold_return_pct"].iloc[0] for w in ("train", "test")}
        print(
            f"\n{args.strategy} | {args.symbol} {tf} | train < {args.split} <= test | "
            f"buy&hold train {bh['train']:.1f}% test {bh['test']:.1f}%"
        )
        res = res[res["train_trades"] >= args.min_trades]
        if res.empty:
            print(f"  no combo made >= {args.min_trades} train trades (lower --min-trades)")
            continue
        show = [c for c in res.columns if not c.endswith(("buy_hold_return_pct", "win_rate_pct"))]
        show = [c for c in show if not c.startswith(("train_", "test_"))] + [
            f"{w}_{c}" for w in ("train", "test") for c in cols
        ]
        print(res[show].head(args.top).round(2).to_string(index=False))


async def run_monitor(args: argparse.Namespace) -> None:
    symbol, timeframe = args.symbol, args.timeframe
    strategy = STRATEGIES[args.strategy]()
    risk = RiskManager(risk_pct=settings.risk_per_trade)
    await startup()
    gw = ExchangeGateway()

    async def on_candle_close(symbol: str, timeframe: str, closed: pd.DataFrame) -> None:
        log.info("%s %s closed @ %s", symbol, timeframe, closed["close"].iloc[-1])
        try:
            async with SessionLocal() as session:
                candles = await load_candles(session, symbol, timeframe, args.days)
                price = gw.prices.get(symbol, float(candles["close"].iloc[-1]))
                payload = await process_close(
                    session,
                    strategy,
                    risk,
                    symbol,
                    timeframe,
                    candles,
                    price,
                    settings.account_equity_usd,
                )
        except Exception:  # a bad candle must not kill ingestion; the next close retries
            log.exception("signal evaluation failed for %s %s", symbol, timeframe)
            return
        if payload:
            # ponytail: alert dispatch (Telegram/webhook) skipped by request; signal is in
            # trade_signals and the log. Plug a dispatcher in here when wanted.
            log.info("SIGNAL %s", json.dumps(payload))

    log.info("monitoring %s %s with %s %s", symbol, timeframe, strategy.name, strategy.params())
    try:
        async with SessionLocal() as session:
            await backfill(gw, session, symbol, timeframe, args.days)  # fill gap since last run
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
        elif args.command == "optimize":
            await run_optimize_cmd(args)
        elif args.command == "monitor":
            await run_monitor(args)
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
    op = sub.add_parser("optimize", help="Grid-search params on train, report on test")
    op.add_argument("--strategy", choices=STRATEGIES, default="DonchianBreakout")
    op.add_argument("--symbol", default="BTC/USDT")
    op.add_argument("--timeframe", nargs="+", default=["15m", "1h", "4h", "1d"])
    op.add_argument("--days", type=int, default=HISTORY_DAYS)
    op.add_argument("--split", default="2025-01-01", help="Train before, test from this date")
    op.add_argument("--fees", type=float, default=0.001)
    op.add_argument(
        "--min-trades", type=int, default=20, help="Drop combos with fewer train trades"
    )
    op.add_argument("--top", type=int, default=10)
    mon = sub.add_parser("monitor", help="Live signal engine: candles -> strategy -> risk -> DB")
    mon.add_argument("--strategy", choices=STRATEGIES, default="DoubleEmaCross")
    mon.add_argument("--symbol", default="BTC/USDT")
    mon.add_argument("--timeframe", default="4h")
    mon.add_argument("--days", type=int, default=365, help="History loaded for indicator warm-up")
    try:
        asyncio.run(_run(parser.parse_args()))
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("shut down")


if __name__ == "__main__":
    main()
