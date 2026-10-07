import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest
from src.strategies import (
    STRATEGIES,
    BaseStrategy,
    DonchianBreakout,
    DoubleEmaCross,
    RsiThreshold,
)


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


ALL = [
    DoubleEmaCross(),
    DoubleEmaCross(trend=50),
    RsiThreshold(),
    RsiThreshold(trend=50),
    DonchianBreakout(),
    DonchianBreakout(trend=0),
]


@pytest.mark.parametrize("strategy", ALL)
def test_signals_are_clean_boolean_arrays(strategy: BaseStrategy) -> None:
    for sig in strategy.evaluate(NOISE):
        assert sig.dtype == bool
        assert sig.index.equals(NOISE.index)
        assert not sig.iloc[0]  # no data before row 0 -> no signal


@pytest.mark.parametrize("strategy", ALL)
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
        "buy_hold_return_pct",
    }
    assert m["trades"] == 1 and m["total_return_pct"] > 0


def test_cli_alias() -> None:
    assert STRATEGIES["DoubleEma"] is DoubleEmaCross


def test_backtest_with_risk_stops_out_trades() -> None:
    from src.core.risk import RiskManager

    plain = run_backtest(RsiThreshold(), NOISE, "1h")
    risked = run_backtest(RsiThreshold(), NOISE, "1h", risk=RiskManager())
    assert risked["trades"] >= plain["trades"]  # stops close trades early, freeing re-entries
    assert risked["max_drawdown_pct"] >= plain["max_drawdown_pct"]  # smaller size, shallower DD


def test_donchian_v_shape_breaks_out_after_bottom() -> None:
    entries, exits = DonchianBreakout(entry=20, exit=10, trend=0).evaluate(V)
    first = V.index.get_loc(entries.idxmax())
    assert 60 < first and entries.iloc[first:].all()  # every bar of the rally makes a new high
    assert exits.iloc[11:60].all() and not exits.iloc[61:].any()  # new lows only in the decline


def test_trend_filter_only_gates_entries() -> None:
    plain, gated = RsiThreshold().evaluate(NOISE), RsiThreshold(trend=50).evaluate(NOISE)
    assert gated.entries.sum() < plain.entries.sum()
    assert not (gated.entries & ~plain.entries).any()  # never adds entries
    assert gated.exits.equals(plain.exits)


def test_backtest_window_does_not_leak_future() -> None:
    from src.core.risk import RiskManager

    split = str(NOISE.index[300])
    shocked = NOISE.copy()
    shocked.iloc[300:] *= 3  # rewrite everything from the split on
    for risk in (None, RiskManager()):
        a = run_backtest(RsiThreshold(), NOISE, "1h", risk=risk, end=split)
        b = run_backtest(RsiThreshold(), shocked, "1h", risk=risk, end=split)
        assert a == b


def test_optimize_skips_invalid_and_reports_train_and_test() -> None:
    from src.backtest import RISK_GRID, optimize

    res = optimize(DoubleEmaCross, NOISE, "1h", str(NOISE.index[300]), 0.001)
    valid = sum(f < s for f in DoubleEmaCross.GRID["fast"] for s in DoubleEmaCross.GRID["slow"])
    assert len(res) == valid * len(DoubleEmaCross.GRID["trend"]) * len(RISK_GRID)
    assert (res["fast"] < res["slow"]).all()
    assert {"train_sharpe_ratio", "test_sharpe_ratio"} <= set(res.columns)


@pytest.mark.parametrize("strategy", ALL)
def test_live_lag0_matches_backtest_next_row(strategy: BaseStrategy) -> None:
    """Live signal for the just-closed candle t == backtest signal on row t+1 (no 1-bar lag)."""
    for t in (100, 250, 400):
        live = strategy.evaluate(NOISE.iloc[: t + 1], lag=0)
        backtest = strategy.evaluate(NOISE.iloc[: t + 2])
        assert live.entries.iloc[-1] == backtest.entries.iloc[t + 1]
        assert live.exits.iloc[-1] == backtest.exits.iloc[t + 1]
