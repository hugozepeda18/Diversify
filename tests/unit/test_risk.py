import pytest

from src.core.risk import RiskManager

RM = RiskManager(risk_pct=0.01, atr_mult=2.0, reward_ratio=2.0)


def test_plan_has_absolute_sl_tp_and_risks_one_percent() -> None:
    p = RM.plan(entry=100.0, atr=1.0, equity=10_000.0)
    assert p.stop_loss == pytest.approx(98.0)
    assert p.take_profit == pytest.approx(104.0)
    assert p.risk_reward_ratio == 2.0
    assert p.position_size_usd == pytest.approx(5_000.0)
    loss_at_stop = p.position_size_usd / 100.0 * (100.0 - p.stop_loss)
    assert loss_at_stop == pytest.approx(100.0)  # 1% of equity


def test_wider_stop_shrinks_position() -> None:
    sizes = [RM.plan(100.0, atr, 10_000.0).position_size_usd for atr in (1.0, 2.0, 4.0)]
    assert sizes == pytest.approx([5_000.0, 2_500.0, 1_250.0])  # doubling the stop halves size


def test_tight_stop_never_exceeds_equity() -> None:
    assert RM.plan(100.0, atr=0.01, equity=10_000.0).position_size_usd == 10_000.0


@pytest.mark.parametrize(
    ("entry", "atr", "equity"), [(0, 1, 1), (100, 0, 1), (100, 1, 0), (100, 60, 1)]
)
def test_rejects_invalid_inputs(entry: float, atr: float, equity: float) -> None:
    with pytest.raises(ValueError):
        RM.plan(entry, atr, equity)
