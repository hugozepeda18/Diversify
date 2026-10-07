import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest
from src.strategies import STRATEGIES, BaseStrategy, DoubleEmaCross, RsiThreshold


def candles(close: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(close), freq="1h", tz="UTC")
    c = pd.Series(close, index=idx, dtype=float)
    return pd.DataFrame(
        {"open": c.shift(1).fillna(c), "high": c, "low": c, "close": c, "volume": 1.0}
    )


V = candles(np.r_[np.linspace(200, 100, 60), np.linspace(100, 200, 60)])  # down then up
A = candles(np.r_[np.linspace(100, 200, 60), np.linspace(200, 100, 60)])  # up then down
NOISE = candles(100 + np.cumsum(np.random.default_rng(42).normal(0, 1, 500)))


def test_ema_cross_v_shape_single_entry_after_bottom() -> None:
    entries, exits = DoubleEmaCross(fast=5, slow=20).evaluate(V)
    assert entries.sum() == 1 and exits.sum() == 0
    assert V.index.get_loc(entries.idxmax()) > 60  # only after the bottom at row 59


def test_ema_cross_a_shape_single_exit_after_top() -> None:
    entries, exits = DoubleEmaCross(fast=5, slow=20).evaluate(A)
    assert exits.sum() == 1 and entries.sum() == 0
    assert A.index.get_loc(exits.idxmax()) > 60


def test_ema_entry_is_one_bar_after_raw_cross() -> None:
    s = DoubleEmaCross(fast=5, slow=20)
    raw = s.indicators(V)
    above = raw["ema_fast"] > raw["ema_slow"]
    raw_cross = int(
        np.flatnonzero(above.to_numpy() & ~above.shift(1, fill_value=True).to_numpy())[0]
    )
    assert V.index.get_loc(s.evaluate(V).entries.idxmax()) == raw_cross + 1


def test_rsi_threshold_matches_shifted_levels() -> None:
    s = RsiThreshold(window=14, lower=30, upper=70)
    rsi = s.indicators(NOISE)["rsi"].shift(1)
    entries, exits = s.evaluate(NOISE)
    pd.testing.assert_series_equal(entries, (rsi < 30).rename(entries.name), check_names=False)
    pd.testing.assert_series_equal(exits, (rsi > 70).rename(exits.name), check_names=False)
    assert entries.any() and exits.any()


def test_rsi_v_shape_enters_on_dip_exits_on_rally() -> None:
    entries, exits = RsiThreshold().evaluate(V)
    assert entries.iloc[15:60].all()  # steady decline is deeply oversold
    assert exits.iloc[75:].all()  # steady rally is deeply overbought
    assert not (entries & exits).any()


@pytest.mark.parametrize("strategy", [DoubleEmaCross(), RsiThreshold()])
def test_signals_are_clean_boolean_arrays(strategy: BaseStrategy) -> None:
    for sig in strategy.evaluate(NOISE):
        assert sig.dtype == bool
        assert sig.index.equals(NOISE.index)
        assert not sig.iloc[0]  # no data before row 0 -> no signal


@pytest.mark.parametrize("strategy", [DoubleEmaCross(), RsiThreshold()])
def test_no_look_ahead(strategy: BaseStrategy) -> None:
    """Changing candle t must not change any signal at rows <= t."""
    base = strategy.evaluate(NOISE)
    for t in (100, 250, 400):
        shocked = NOISE.copy()
        shocked.iloc[t:, shocked.columns.get_loc("close")] *= 3
        after = strategy.evaluate(shocked)
        for b, a in zip(base, after, strict=True):
            assert b.iloc[: t + 1].equals(a.iloc[: t + 1])


def test_backtest_metrics() -> None:
    m = run_backtest(DoubleEmaCross(fast=5, slow=20), V, "1h")
    assert set(m) == {
        "total_return_pct",
        "sharpe_ratio",
        "max_drawdown_pct",
        "win_rate_pct",
        "trades",
    }
    assert m["trades"] == 1 and m["total_return_pct"] > 0


def test_cli_alias() -> None:
    assert STRATEGIES["DoubleEma"] is DoubleEmaCross
