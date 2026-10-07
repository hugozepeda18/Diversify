from abc import ABC, abstractmethod
from typing import NamedTuple

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

    @abstractmethod
    def indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Raw (unshifted) indicator columns, indexed like `candles`."""

    @abstractmethod
    def triggers(self, ind: pd.DataFrame) -> Signals:
        """Boolean entry/exit Series computed from already-shifted indicators."""

    def evaluate(self, candles: pd.DataFrame) -> Signals:
        entries, exits = self.triggers(self.indicators(candles).shift(1))
        # NaN warm-up rows compare as False, but make it explicit and typed.
        return Signals(entries.fillna(False).astype(bool), exits.fillna(False).astype(bool))


class DoubleEmaCross(BaseStrategy):
    name = "DoubleEmaCross"

    def __init__(self, fast: int = 12, slow: int = 26) -> None:
        self.fast, self.slow = fast, slow

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

    def __init__(self, window: int = 14, lower: float = 30, upper: float = 70) -> None:
        self.window, self.lower, self.upper = window, lower, upper

    def indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"rsi": vbt.RSI.run(candles["close"], self.window).rsi})

    def triggers(self, ind: pd.DataFrame) -> Signals:
        return Signals(ind["rsi"] < self.lower, ind["rsi"] > self.upper)


STRATEGIES: dict[str, type[BaseStrategy]] = {
    "DoubleEma": DoubleEmaCross,  # short alias used in CLAUDE.md
    "DoubleEmaCross": DoubleEmaCross,
    "RsiThreshold": RsiThreshold,
}
