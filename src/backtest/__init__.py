import ccxt
import pandas as pd
import vectorbt as vbt

from src.core.risk import RiskManager
from src.strategies import BaseStrategy


def run_backtest(
    strategy: BaseStrategy,
    candles: pd.DataFrame,
    timeframe: str,
    fees: float = 0.001,  # Binance spot taker fee
    init_cash: float = 10_000.0,
    risk: RiskManager | None = None,
) -> dict[str, float]:
    """Fill at next bar's open; with `risk`, every entry gets ATR SL/TP and risk-based size."""
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
