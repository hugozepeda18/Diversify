from dataclasses import dataclass
from typing import NamedTuple, TypeVar, cast

import numpy as np
import pandas as pd
import vectorbt as vbt

T = TypeVar("T", float, pd.Series)


class RiskPlan(NamedTuple):
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    position_size_usd: float


@dataclass(frozen=True)
class RiskManager:
    """Long-only ATR stops; size so a stop-out loses `risk_pct` of equity.

    ponytail: long-only and spot (size capped at 100% of equity), matching the strategies.
    """

    risk_pct: float = 0.01
    atr_window: int = 14
    atr_mult: float = 2.0  # stop distance = atr_mult * ATR
    reward_ratio: float = 2.0  # TP distance = reward_ratio * stop distance

    def atr(self, candles: pd.DataFrame) -> pd.Series:
        """ATR through each bar's close (unshifted; the caller decides when it is known)."""
        atr: pd.Series = vbt.ATR.run(
            candles["high"], candles["low"], candles["close"], self.atr_window
        ).atr
        return atr

    def size_fraction(self, entry: T, stop_loss: T) -> T:
        """Fraction of equity to deploy; works on scalars or whole Series."""
        return cast(T, np.minimum(self.risk_pct * entry / (entry - stop_loss), 1.0))

    def plan(self, entry: float, atr: float, equity: float) -> RiskPlan:
        if not (entry > 0 and atr > 0 and equity > 0):
            raise ValueError(f"invalid risk inputs: entry={entry} atr={atr} equity={equity}")
        stop = entry - self.atr_mult * atr
        if stop <= 0:
            raise ValueError(f"stop {stop} <= 0: ATR too large for entry {entry}")
        return RiskPlan(
            stop_loss=stop,
            take_profit=entry + self.reward_ratio * (entry - stop),
            risk_reward_ratio=self.reward_ratio,
            position_size_usd=equity * float(self.size_fraction(entry, stop)),
        )
