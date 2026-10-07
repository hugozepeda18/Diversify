from abc import ABC, abstractmethod
from typing import ClassVar, NamedTuple

import pandas as pd
import vectorbt as vbt


class Signals(NamedTuple):
    entries: pd.Series
    exits: pd.Series


def crossed_above(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


class BaseStrategy(ABC):
    """Vectorized strategy contract.

    `evaluate` shifts every indicator by one bar before triggers are computed, so the signal on
    row t only uses candles that closed at or before t-1. Act on it at row t's open.
    """

    name: str
    GRID: ClassVar[dict[str, list[float]]] = {}  # parameter search space for `optimize`
    trend: int = 0  # EMA window; entries only while close is above it (0 = off)

    def params(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in self.GRID}

    @abstractmethod
    def indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Raw (unshifted) indicator columns, indexed like `candles`."""

    @abstractmethod
    def triggers(self, ind: pd.DataFrame) -> Signals:
        """Boolean entry/exit Series computed from already-shifted indicators."""

    def frame(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Unshifted indicators plus the trend-filter columns; also the live snapshot source."""
        ind = self.indicators(candles)
        if self.trend:
            close = candles["close"]
            ind = ind.assign(close=close, ema_trend=vbt.MA.run(close, self.trend, ewm=True).ma)
        return ind

    def evaluate(self, candles: pd.DataFrame, lag: int = 1) -> Signals:
        """lag=1 (default) for backtests. lag=0 is for live use only: its last row is the
        signal for the bar *after* the newest closed candle, i.e. evaluate()'s next row."""
        ind = self.frame(candles).shift(lag)
        entries, exits = self.triggers(ind)
        if self.trend:
            entries = entries & (ind["close"] > ind["ema_trend"])
        # NaN warm-up rows compare as False, but make it explicit and typed.
        return Signals(entries.fillna(False).astype(bool), exits.fillna(False).astype(bool))


class DoubleEmaCross(BaseStrategy):
    name = "DoubleEmaCross"
    GRID: ClassVar[dict[str, list[float]]] = {
        "fast": [10, 20, 50],
        "slow": [26, 50, 100, 200],
        "trend": [0, 200],
    }

    # Defaults from `optimize` (4h, train 2022-10..2024, test 2025-): robust plateau around 20/200.
    def __init__(self, fast: int = 20, slow: int = 200, trend: int = 0) -> None:
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be < slow ({slow})")
        self.fast, self.slow, self.trend = fast, slow, trend

    def indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        close = candles["close"]
        return pd.DataFrame(
            {
                "ema_fast": vbt.MA.run(close, self.fast, ewm=True).ma,
                "ema_slow": vbt.MA.run(close, self.slow, ewm=True).ma,
            }
        )

    def triggers(self, ind: pd.DataFrame) -> Signals:
        return Signals(
            crossed_above(ind["ema_fast"], ind["ema_slow"]),
            crossed_above(ind["ema_slow"], ind["ema_fast"]),
        )


class RsiThreshold(BaseStrategy):
    """Enter when RSI dips below `lower` (oversold), exit when it rises above `upper`."""

    name = "RsiThreshold"
    GRID: ClassVar[dict[str, list[float]]] = {
        "lower": [25, 30, 35, 40],
        "upper": [60, 70, 80],
        "trend": [0, 200],
    }

    def __init__(
        self, window: int = 14, lower: float = 30, upper: float = 70, trend: int = 0
    ) -> None:
        self.window, self.lower, self.upper, self.trend = window, lower, upper, trend

    def indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"rsi": vbt.RSI.run(candles["close"], self.window).rsi})

    def triggers(self, ind: pd.DataFrame) -> Signals:
        return Signals(ind["rsi"] < self.lower, ind["rsi"] > self.upper)


class DonchianBreakout(BaseStrategy):
    """Enter on a close above the prior `entry`-bar high; exit on a close below the prior
    `exit`-bar low (Turtle-style trend following)."""

    name = "DonchianBreakout"
    GRID: ClassVar[dict[str, list[float]]] = {
        "entry": [20, 50, 100],
        "exit": [10, 20, 50],
        "trend": [0, 200],
    }

    def __init__(self, entry: int = 20, exit: int = 20, trend: int = 200) -> None:
        self.entry, self.exit, self.trend = entry, exit, trend

    def indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "close": candles["close"],
                # Channels exclude the current bar, else close > high-incl-close never fires.
                "upper": candles["high"].rolling(self.entry).max().shift(1),
                "lower": candles["low"].rolling(self.exit).min().shift(1),
            }
        )

    def triggers(self, ind: pd.DataFrame) -> Signals:
        return Signals(ind["close"] > ind["upper"], ind["close"] < ind["lower"])


STRATEGIES: dict[str, type[BaseStrategy]] = {
    "DoubleEma": DoubleEmaCross,  # short alias used in CLAUDE.md
    "DoubleEmaCross": DoubleEmaCross,
    "RsiThreshold": RsiThreshold,
    "DonchianBreakout": DonchianBreakout,
}
