import ccxt
import pandas as pd
import vectorbt as vbt

from src.strategies import BaseStrategy


def run_backtest(
    strategy: BaseStrategy,
    candles: pd.DataFrame,
    timeframe: str,
    fees: float = 0.001,  # Binance spot taker fee
    init_cash: float = 10_000.0,
) -> dict[str, float]:
    entries, exits = strategy.evaluate(candles)
    pf = vbt.Portfolio.from_signals(
        close=candles["close"],
        entries=entries,
        exits=exits,
        price=candles["open"],  # signals use t-1 data, so fill at bar t's open
        fees=fees,
        init_cash=init_cash,
        freq=pd.Timedelta(seconds=ccxt.Exchange.parse_timeframe(timeframe)),
    )
    return {
        "total_return_pct": float(pf.total_return()) * 100,
        "sharpe_ratio": float(pf.sharpe_ratio()),
        "max_drawdown_pct": float(pf.max_drawdown()) * 100,
        "win_rate_pct": float(pf.trades.win_rate()) * 100,
        "trades": float(pf.trades.count()),
    }
