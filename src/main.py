import argparse
import asyncio
import logging
import signal

import pandas as pd

from src.backtest import optimize, rank_across, run_backtest
from src.core.config import settings
from src.core.db import SessionLocal, engine
from src.core.recovery import load_active_positions
from src.core.risk import RiskManager
from src.data.candles import backfill, ingest_live, load_candles
from src.data.gateway import ExchangeGateway
from src.execution import BinanceDemoBroker
from src.signals import SignalEngine
from src.strategies import STRATEGIES

log = logging.getLogger("trading-bot")
# ~4 years: spans the 2022 bear, 2023 chop and 2024-25 bull, so a strategy is judged across regimes.
HISTORY_DAYS = 1460
# Top 10 non-stablecoins by market cap on Binance spot with 4y of history (CoinGecko, 2026-10-07).
# Skipped: HYPE (listed on Binance 2026-09-24, no history), XMR (not on Binance spot).
# Research (2022-10..2026-10, 10 coins): 15m loses to fees, 1h is noisy, 1d trades too rarely.
TIMEFRAMES = ["2h", "4h"]
TOP_COINS = [
    f"{c}/USDT" for c in ("BTC", "ETH", "BNB", "XRP", "SOL", "TRX", "ZEC", "DOGE", "LINK", "ADA")
]


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
    classes = list(dict.fromkeys(STRATEGIES[name] for name in args.strategy))  # dedupe aliases
    for tf in args.timeframe:
        per_coin, bh = [], {}
        for symbol in args.symbol:
            await run_backfill(symbol, tf, args.days)
            async with SessionLocal() as session:
                candles = await load_candles(session, symbol, tf, args.days)
            for cls in classes:
                res = optimize(cls, candles, tf, args.split, args.fees).assign(symbol=symbol)
                per_coin.append(res)
            bh[symbol] = res["test_buy_hold_return_pct"].iloc[0]
        ranked = rank_across(pd.concat(per_coin), args.min_trades)
        print(
            f"\n=== {tf} | {len(args.symbol)} coin(s) | train < {args.split} <= test | "
            f"median buy&hold test {pd.Series(bh).median():.1f}% ==="
        )
        if ranked.empty:
            print(f"  no config made >= {args.min_trades} median train trades")
            continue
        print(ranked.head(args.top).round(2).to_string(index=False))
        best = ranked.groupby("strategy").head(1)  # best train config per strategy family
        print(f"\n--- best per strategy ({tf}) ---")
        print(best.round(2).to_string(index=False))


def make_broker() -> BinanceDemoBroker:
    key, secret = settings.binance_testnet_api_key, settings.binance_testnet_secret
    if key is None or secret is None:
        raise SystemExit("set BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_SECRET in .env")
    return BinanceDemoBroker(key.get_secret_value(), secret.get_secret_value())


async def run_broker_check() -> None:
    """Read-only: demo balance plus the exchange status of every exchange-held position."""
    broker = make_broker()
    try:
        print(f"Binance Demo free USDT: {await broker.free_usdt():.2f}")
        async with SessionLocal() as session:
            positions = (await load_active_positions(session)).values()
        for p in positions:
            if p.exchange_ref is None:
                print(f"  {p.symbol} {p.timeframe} {p.strategy_name}: virtual (no exchange order)")
                continue
            status = await broker.check(p.symbol, p.exchange_ref)
            print(
                f"  {p.symbol} {p.timeframe} {p.strategy_name}: qty {p.quantity} "
                f"SL {p.stop_loss} TP {p.take_profit} OCO {p.exchange_ref} -> "
                f"{'open' if status is None else status}"
            )
    finally:
        await broker.close()


async def run_monitor(args: argparse.Namespace) -> None:
    strategy = STRATEGIES[args.strategy]()
    risk = RiskManager(risk_pct=settings.risk_per_trade)
    broker = make_broker() if args.execute else None
    peers: list[SignalEngine] = []
    engines = {
        tf: SignalEngine(
            SessionLocal,
            strategy,
            risk,
            tf,
            settings.account_equity_usd,
            args.days,
            broker=broker,
            peers=peers,
        )
        for tf in args.timeframe
    }
    await startup()
    for eng in engines.values():
        await eng.load()
    gw = ExchangeGateway()

    async def on_candle_close(symbol: str, timeframe: str, closed: pd.DataFrame) -> None:
        try:
            await engines[timeframe].on_close(symbol, timeframe, closed, gw.prices.get(symbol))
        except Exception:  # a bad candle must not kill ingestion; the next close retries
            log.exception("signal evaluation failed for %s %s", symbol, timeframe)

    async def on_tick(symbol: str, price: float) -> None:
        for eng in engines.values():
            await eng.on_tick(symbol, price)

    log.info(
        "monitoring %d coin(s) on %s with %s %s | execution: %s",
        len(args.symbol),
        "+".join(engines),
        strategy.name,
        strategy.params(),
        "Binance Demo (real orders, fake funds)" if broker else "signals only",
    )
    try:
        for symbol in args.symbol:
            for tf in engines:
                async with SessionLocal() as session:
                    await backfill(gw, session, symbol, tf, args.days)  # fill gaps
        async with asyncio.TaskGroup() as tg:
            if broker is not None:  # also reconciles fills from downtime
                for eng in engines.values():
                    tg.create_task(eng.run_exchange_sync())
            for symbol in args.symbol:
                tg.create_task(gw.stream_prices(symbol, on_tick))
                for tf in engines:
                    tg.create_task(ingest_live(gw, SessionLocal, symbol, tf, on_candle_close))
    finally:
        await gw.close()
        if broker is not None:
            await broker.close()


async def _run(args: argparse.Namespace) -> None:
    task = asyncio.current_task()
    assert task is not None
    # SIGTERM (docker stop, kill) cancels like Ctrl-C so the finally blocks close sockets/DB.
    # Idempotent: a repeated SIGTERM (e.g. forwarded by `uv run`) must not cancel the cleanup.
    asyncio.get_running_loop().add_signal_handler(
        signal.SIGTERM, lambda: None if task.cancelling() else task.cancel()
    )
    try:
        if args.command == "recover":
            await startup()
        elif args.command == "backfill":
            for symbol in args.symbol:
                for tf in args.timeframe:
                    await run_backfill(symbol, tf, args.days)
        elif args.command == "backtest":
            await run_backtest_cmd(args)
        elif args.command == "optimize":
            await run_optimize_cmd(args)
        elif args.command == "monitor":
            await run_monitor(args)
        elif args.command == "broker-check":
            await run_broker_check()
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
    bf.add_argument("--symbol", nargs="+", default=TOP_COINS)
    bf.add_argument("--timeframe", nargs="+", choices=TIMEFRAMES, default=TIMEFRAMES)
    bf.add_argument("--days", type=int, default=HISTORY_DAYS)
    bt = sub.add_parser("backtest", help="Backtest a strategy on stored candles")
    bt.add_argument("--strategy", choices=STRATEGIES, default="DoubleEma")
    bt.add_argument("--symbol", default="BTC/USDT")
    bt.add_argument("--timeframe", choices=TIMEFRAMES, default="4h")
    bt.add_argument("--days", type=int, default=HISTORY_DAYS)
    bt.add_argument("--no-risk", action="store_true", help="Disable SL/TP and risk sizing")
    bt.add_argument("--fees", type=float, default=0.001, help="Fee per fill (0.001 = 0.1%%)")
    op = sub.add_parser("optimize", help="Grid-search params on train, report on test")
    op.add_argument("--strategy", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    op.add_argument("--symbol", nargs="+", default=TOP_COINS)
    op.add_argument("--timeframe", nargs="+", choices=TIMEFRAMES, default=TIMEFRAMES)
    op.add_argument("--days", type=int, default=HISTORY_DAYS)
    op.add_argument("--split", default="2025-01-01", help="Train before, test from this date")
    op.add_argument("--fees", type=float, default=0.001)
    op.add_argument("--min-trades", type=int, default=10, help="Min median train trades per coin")
    op.add_argument("--top", type=int, default=10)
    mon = sub.add_parser("monitor", help="Live signal engine: candles -> strategy -> risk -> DB")
    mon.add_argument("--strategy", choices=STRATEGIES, default="DoubleEmaCross")
    mon.add_argument("--symbol", nargs="+", default=TOP_COINS)
    mon.add_argument("--timeframe", nargs="+", choices=TIMEFRAMES, default=TIMEFRAMES)
    mon.add_argument("--days", type=int, default=365, help="History loaded for indicator warm-up")
    mon.add_argument(
        "--execute",
        action="store_true",
        help="Place real orders on Binance Demo Trading (keys from .env)",
    )
    sub.add_parser("broker-check", help="Read-only: demo balance and exchange-held positions")
    try:
        asyncio.run(_run(parser.parse_args()))
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("shut down")


if __name__ == "__main__":
    main()
