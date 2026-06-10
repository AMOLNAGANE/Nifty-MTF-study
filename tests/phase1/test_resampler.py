import pytest
import pandas as pd
import numpy as np
from src.phase1_data.resampler import resample_to_15m, resample_to_1h, resample_to_1d

TZ = "Asia/Kolkata"


def make_one_day_5m() -> pd.DataFrame:
    """75 synthetic 5m bars for 2020-11-09 (09:15–15:25, closes at 15:30)."""
    start = pd.Timestamp("2020-11-09 09:15:00", tz=TZ)
    timestamps = [start + pd.Timedelta(minutes=5 * i) for i in range(75)]
    return pd.DataFrame({
        "timestamp": timestamps,
        "bar_close": [t + pd.Timedelta(minutes=5) for t in timestamps],
        "open":  np.arange(1.0, 76.0),
        "high":  np.arange(1.0, 76.0) + 0.5,
        "low":   np.arange(1.0, 76.0) - 0.5,
        "close": np.arange(1.0, 76.0) + 0.1,
        "session_date": [t.date() for t in timestamps],
        "bar_in_session": np.arange(1, 76),
    })


def make_two_day_5m() -> pd.DataFrame:
    day1 = make_one_day_5m()
    start2 = pd.Timestamp("2020-11-10 09:15:00", tz=TZ)
    timestamps2 = [start2 + pd.Timedelta(minutes=5 * i) for i in range(75)]
    day2 = pd.DataFrame({
        "timestamp": timestamps2,
        "bar_close": [t + pd.Timedelta(minutes=5) for t in timestamps2],
        "open":  np.arange(100.0, 175.0),
        "high":  np.arange(100.0, 175.0) + 0.5,
        "low":   np.arange(100.0, 175.0) - 0.5,
        "close": np.arange(100.0, 175.0) + 0.1,
        "session_date": [t.date() for t in timestamps2],
        "bar_in_session": np.arange(1, 76),
    })
    return pd.concat([day1, day2], ignore_index=True)


# ---- 15m tests ----

def test_15m_bar_count_per_day():
    assert len(resample_to_15m(make_one_day_5m())) == 25

def test_15m_first_bar_timestamp():
    result = resample_to_15m(make_one_day_5m())
    assert result["timestamp"].iloc[0] == pd.Timestamp("2020-11-09 09:15:00", tz=TZ)

def test_15m_first_bar_close():
    result = resample_to_15m(make_one_day_5m())
    assert result["bar_close"].iloc[0] == pd.Timestamp("2020-11-09 09:30:00", tz=TZ)

def test_15m_high_is_max_of_3_bars():
    result = resample_to_15m(make_one_day_5m())
    # Bars 0–2: highs 1.5, 2.5, 3.5 → max 3.5
    assert result["high"].iloc[0] == pytest.approx(3.5)

def test_15m_open_is_first_bar_open():
    result = resample_to_15m(make_one_day_5m())
    assert result["open"].iloc[0] == pytest.approx(1.0)

def test_15m_close_is_last_bar_close():
    result = resample_to_15m(make_one_day_5m())
    # First 15m covers bars 0–2; bar 2 close = 3.1
    assert result["close"].iloc[0] == pytest.approx(3.1)

def test_15m_no_bleed_across_days():
    assert len(resample_to_15m(make_two_day_5m())) == 50

def test_15m_bar_in_session_range():
    result = resample_to_15m(make_one_day_5m())
    assert result["bar_in_session"].iloc[0] == 1
    assert result["bar_in_session"].iloc[-1] == 25


# ---- 1h tests ----

def test_1h_bar_count_per_day_stub_dropped():
    assert len(resample_to_1h(make_one_day_5m())) == 6

def test_1h_first_bar_timestamp():
    result = resample_to_1h(make_one_day_5m())
    assert result["timestamp"].iloc[0] == pd.Timestamp("2020-11-09 09:15:00", tz=TZ)

def test_1h_first_bar_close():
    result = resample_to_1h(make_one_day_5m())
    assert result["bar_close"].iloc[0] == pd.Timestamp("2020-11-09 10:15:00", tz=TZ)

def test_1h_last_bar_close_is_1515():
    result = resample_to_1h(make_one_day_5m())
    assert result["bar_close"].iloc[-1] == pd.Timestamp("2020-11-09 15:15:00", tz=TZ)

def test_1h_no_bleed_across_days():
    assert len(resample_to_1h(make_two_day_5m())) == 12


# ---- 1d tests ----

def test_1d_bar_count():
    assert len(resample_to_1d(make_one_day_5m())) == 1

def test_1d_bar_close_is_1530():
    result = resample_to_1d(make_one_day_5m())
    assert result["bar_close"].iloc[0] == pd.Timestamp("2020-11-09 15:30:00", tz=TZ)

def test_1d_open_is_first_bar():
    result = resample_to_1d(make_one_day_5m())
    assert result["open"].iloc[0] == pytest.approx(1.0)

def test_1d_close_is_last_bar():
    result = resample_to_1d(make_one_day_5m())
    assert result["close"].iloc[0] == pytest.approx(75.1)

def test_1d_two_days():
    assert len(resample_to_1d(make_two_day_5m())) == 2

def test_output_columns_present():
    result = resample_to_15m(make_one_day_5m())
    for col in ["timestamp", "bar_close", "open", "high", "low", "close",
                "session_date", "bar_in_session"]:
        assert col in result.columns
