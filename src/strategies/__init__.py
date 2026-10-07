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


class SmaRegime(BaseStrategy):
    """Long while close is above its SMA ("200-day rule"), with a +/- `band` to cut whipsaws.

    On 1d, window=200 is the classic rule; on 4h the same 200 days is window=1200.
    """

    name = "SmaRegime"
    GRID: ClassVar[dict[str, list[float]]] = {
        "window": [50, 100, 200, 600, 1200],
        "band": [0.0, 0.02, 0.05],
    }

    def __init__(self, window: int = 200, band: float = 0.0) -> None:
        self.window, self.band = window, band

    def indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        close = candles["close"]
        return pd.DataFrame({"close": close, "sma": vbt.MA.run(close, self.window).ma})

    def triggers(self, ind: pd.DataFrame) -> Signals:
        return Signals(
            ind["close"] > ind["sma"] * (1 + self.band),
            ind["close"] < ind["sma"] * (1 - self.band),
        )


class TripleMa(BaseStrategy):
    """Long while EMAs stack fast > mid > slow (trend aligned on 3 horizons); exit on fast < mid."""

    name = "TripleMa"
    GRID: ClassVar[dict[str, list[float]]] = {
        "fast": [5, 10, 20],
        "mid": [20, 50, 100],
        "slow": [100, 200, 400],
    }

    def __init__(self, fast: int = 10, mid: int = 50, slow: int = 200) -> None:
        if not fast < mid < slow:
            raise ValueError(f"need fast < mid < slow, got {fast}/{mid}/{slow}")
        self.fast, self.mid, self.slow = fast, mid, slow

    def indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        close = candles["close"]
        return pd.DataFrame(
            {f"ema_{k}": vbt.MA.run(close, getattr(self, k), ewm=True).ma for k in self.GRID}
        )

    def triggers(self, ind: pd.DataFrame) -> Signals:
        f, m, s = ind["ema_fast"], ind["ema_mid"], ind["ema_slow"]
        return Signals((f > m) & (m > s), f < m)


class MaSlope(BaseStrategy):
    """Long while close is above a rising EMA (EMA higher than `lookback` bars ago)."""

    name = "MaSlope"
    GRID: ClassVar[dict[str, list[float]]] = {
        "window": [50, 100, 200],
        "lookback": [5, 10, 20],
    }

    def __init__(self, window: int = 100, lookback: int = 10) -> None:
        self.window, self.lookback = window, lookback

    def indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        close = candles["close"]
        ema = vbt.MA.run(close, self.window, ewm=True).ma
        return pd.DataFrame(
            {"close": close, "ema": ema, "ema_slope": ema - ema.shift(self.lookback)}
        )

    def triggers(self, ind: pd.DataFrame) -> Signals:
        # Explicit comparisons, not ~above: NaN warm-up rows must stay False on both sides.
        entries = (ind["close"] > ind["ema"]) & (ind["ema_slope"] > 0)
        return Signals(entries, (ind["close"] < ind["ema"]) | (ind["ema_slope"] < 0))


class RibbonScore(BaseStrategy):
    """Vote across an EMA ribbon: score = share of EMAs (10..200) that close is above.

    Enter when score >= `enter`, exit when score <= `exit`. Averaging many horizons makes it far
    less sensitive to any single window choice than a two-line crossover.
    """

    name = "RibbonScore"
    WINDOWS = (10, 20, 50, 100, 150, 200)
    GRID: ClassVar[dict[str, list[float]]] = {
        "enter": [0.66, 0.83, 1.0],
        "exit": [0.16, 0.33, 0.5],
    }

    def __init__(self, enter: float = 0.83, exit: float = 0.33) -> None:
        if exit >= enter:
            raise ValueError(f"exit ({exit}) must be < enter ({enter})")
        self.enter, self.exit = enter, exit

    def indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        close = candles["close"]
        emas = vbt.MA.run(close, list(self.WINDOWS), ewm=True).ma  # one column per window
        above = emas.lt(close, axis=0).to_numpy()
        score = pd.Series(above.mean(axis=1), index=close.index)
        return pd.DataFrame({"score": score.where(emas.notna().all(axis=1).to_numpy())})

    def triggers(self, ind: pd.DataFrame) -> Signals:
        return Signals(ind["score"] >= self.enter, ind["score"] <= self.exit)


class KeltnerBreakout(BaseStrategy):
    """Enter on a close above EMA + mult*ATR (volatility-adjusted breakout); exit below the EMA."""

    name = "KeltnerBreakout"
    GRID: ClassVar[dict[str, list[float]]] = {
        "window": [20, 50, 100],
        "mult": [1.0, 2.0, 3.0],
    }

    def __init__(self, window: int = 50, mult: float = 2.0) -> None:
        self.window, self.mult = window, mult

    def indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        close = candles["close"]
        ema = vbt.MA.run(close, self.window, ewm=True).ma
        atr = vbt.ATR.run(candles["high"], candles["low"], close, self.window).atr
        return pd.DataFrame({"close": close, "ema": ema, "upper": ema + self.mult * atr})

    def triggers(self, ind: pd.DataFrame) -> Signals:
        return Signals(ind["close"] > ind["upper"], ind["close"] < ind["ema"])


STRATEGIES: dict[str, type[BaseStrategy]] = {
    "DoubleEma": DoubleEmaCross,  # short alias used in CLAUDE.md
    "DoubleEmaCross": DoubleEmaCross,
    "RsiThreshold": RsiThreshold,
    "DonchianBreakout": DonchianBreakout,
    "SmaRegime": SmaRegime,
    "TripleMa": TripleMa,
    "MaSlope": MaSlope,
    "RibbonScore": RibbonScore,
    "KeltnerBreakout": KeltnerBreakout,
}
