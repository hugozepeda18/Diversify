import itertools

import ccxt
import pandas as pd
import vectorbt as vbt

from src.core.risk import RiskManager
from src.strategies import BaseStrategy

# (atr_mult, reward_ratio) pairs searched by `optimize`; wide R:R ~ let winners run.
# None = all-in, no stops: the plain-exposure benchmark (e.g. SmaRegime 200d all-in).
RISK_GRID: list[tuple[float, float] | None] = [
    (2.0, 2.0),
    (3.0, 3.0),
    (3.0, 10.0),
    (5.0, 10.0),
    None,
]


def run_backtest(
    strategy: BaseStrategy,
    candles: pd.DataFrame,
    timeframe: str,
    fees: float = 0.001,  # Binance spot taker fee
    init_cash: float = 10_000.0,
    risk: RiskManager | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, float]:
    """Fill at next bar's open; with `risk`, every entry gets ATR SL/TP and risk-based size.

    Indicators warm up on all of `candles`; only [start, end) is traded (both causal, no leak).
    """
    entries, exits = strategy.evaluate(candles)
    stops: dict[str, object] = {}
    if risk is not None:
        entry = candles["open"]
        stop = entry - risk.atr_mult * risk.atr(candles).shift(1)  # ATR known at t-1's close
        stop_frac = (entry - stop) / entry
        stops = {
            "sl_stop": stop_frac,
            "tp_stop": stop_frac * risk.reward_ratio,
            "size": risk.size_fraction(entry, stop),
            "size_type": "percent",  # of cash; equals equity when flat (long-only, no pyramiding)
            "open": candles["open"],
            "high": candles["high"],  # stops trigger intrabar on high/low
            "low": candles["low"],
        }
    keep = pd.Series(True, index=candles.index)
    if start:
        keep &= candles.index >= pd.Timestamp(start, tz="UTC")
    if end:
        keep &= candles.index < pd.Timestamp(end, tz="UTC")
    stops = {k: v[keep] if isinstance(v, pd.Series) else v for k, v in stops.items()}
    candles, entries, exits = candles[keep], entries[keep], exits[keep]
    pf = vbt.Portfolio.from_signals(
        close=candles["close"],
        entries=entries,
        exits=exits,
        price=candles["open"],  # signals use t-1 data, so fill at bar t's open
        fees=fees,
        init_cash=init_cash,
        freq=pd.Timedelta(seconds=ccxt.Exchange.parse_timeframe(timeframe)),
        **stops,
    )
    return {
        "total_return_pct": float(pf.total_return()) * 100,
        "sharpe_ratio": float(pf.sharpe_ratio()),
        "max_drawdown_pct": float(pf.max_drawdown()) * 100,
        "win_rate_pct": float(pf.trades.win_rate()) * 100,
        "trades": float(pf.trades.count()),
        "buy_hold_return_pct": float(pf.total_benchmark_return()) * 100,
    }


def optimize(
    cls: type[BaseStrategy], candles: pd.DataFrame, timeframe: str, split: str, fees: float
) -> pd.DataFrame:
    """Grid-search `cls.GRID` x RISK_GRID on data before `split`, report each on data after.

    Loops over parameter combos only; each backtest is vectorized.
    """
    rows = []
    for values in itertools.product(*cls.GRID.values()):
        params = dict(zip(cls.GRID, values, strict=True))
        try:
            strategy = cls(**params)
        except ValueError:  # invalid combo, e.g. fast >= slow
            continue
        for stops in RISK_GRID:
            risk = None if stops is None else RiskManager(atr_mult=stops[0], reward_ratio=stops[1])
            train = run_backtest(strategy, candles, timeframe, fees, risk=risk, end=split)
            test = run_backtest(strategy, candles, timeframe, fees, risk=risk, start=split)
            rows.append(
                {
                    "strategy": cls.name,
                    "params": " ".join(f"{k}={v:g}" for k, v in params.items()),
                    "risk": "all-in" if stops is None else f"{stops[0]:g}xATR/{stops[1]:g}R",
                    **{f"train_{k}": v for k, v in train.items()},
                    **{f"test_{k}": v for k, v in test.items()},
                }
            )
    return pd.DataFrame(rows).sort_values("train_sharpe_ratio", ascending=False)


def rank_across(results: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    """Collapse per-symbol `optimize` rows into one row per config, judged across all coins.

    A real edge should hold on most coins with the same parameters, so rank by the *median*
    train Sharpe and show how many coins stayed profitable / beat buy & hold out-of-sample.
    """
    r = results.assign(
        test_profitable=results["test_total_return_pct"] > 0,
        test_beats_bh=results["test_total_return_pct"] > results["test_buy_hold_return_pct"],
    )
    out = r.groupby(["strategy", "params", "risk"]).agg(
        coins=("train_trades", "size"),
        train_trades=("train_trades", "median"),
        train_sharpe=("train_sharpe_ratio", "median"),
        train_return=("train_total_return_pct", "median"),
        test_sharpe=("test_sharpe_ratio", "median"),
        test_return=("test_total_return_pct", "median"),
        test_max_dd=("test_max_drawdown_pct", "median"),
        test_profitable_pct=("test_profitable", "mean"),
        test_beats_bh_pct=("test_beats_bh", "mean"),
    )
    out[["test_profitable_pct", "test_beats_bh_pct"]] *= 100
    out = out[out["train_trades"] >= min_trades]
    return out.sort_values("train_sharpe", ascending=False).reset_index()
