import pandas as pd

from src.data.candles import split_closed, to_frame

H = 3_600_000


def frame(*rows: tuple[int, float]) -> pd.DataFrame:
    return to_frame([[ts * H, 1, 1, 1, close, 1] for ts, close in rows], "X/Y", "1h")


def ts(i: int) -> pd.Timestamp:
    return pd.Timestamp(i * H, unit="ms", tz="UTC")


def test_forming_candle_updates_stay_pending() -> None:
    closed, pending = split_closed(frame((1, 10)), frame((1, 11)), None)
    assert closed.empty
    assert pending["close"].tolist() == [11]


def test_newer_candle_closes_previous_with_latest_values() -> None:
    _, pending = split_closed(frame(), frame((1, 10)), None)
    _, pending = split_closed(pending, frame((1, 12)), None)
    closed, pending = split_closed(pending, frame((2, 20)), None)
    assert closed["timestamp"].tolist() == [ts(1)]
    assert closed["close"].tolist() == [12]
    assert pending["timestamp"].tolist() == [ts(2)]


def test_update_carrying_final_and_new_candle() -> None:
    closed, pending = split_closed(frame((1, 10)), frame((1, 13), (2, 20)), None)
    assert closed["close"].tolist() == [13]
    assert pending["timestamp"].tolist() == [ts(2)]


def test_already_stored_candles_are_not_reemitted() -> None:
    closed, _ = split_closed(frame((1, 10)), frame((2, 20)), last_closed=ts(1))
    assert closed.empty
