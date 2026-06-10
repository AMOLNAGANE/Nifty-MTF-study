import pytest
import pandas as pd
from pathlib import Path
from src.phase1_data.loader import load_5m_csv

RAW_CSV = Path("Data/nifty_intraday_5min.csv")

SAMPLE = (
    "open,high,low,close,volume,timestamp\n"
    "12389.0,12430.7999,12383.8999,12410.1,0.0,2020-11-09 09:15:00+05:30\n"
    "12409.7,12445.2,12407.7999,12443.6,0.0,2020-11-09 09:20:00+05:30\n"
    "12443.2999,12447.7,12433.1,12447.2,0.0,2020-11-09 09:25:00+05:30\n"
)

def _csv(tmp_path, content):
    p = tmp_path / "test.csv"
    p.write_text(content)
    return p

def test_returns_dataframe(tmp_path):
    df = load_5m_csv(_csv(tmp_path, SAMPLE))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3

def test_volume_dropped(tmp_path):
    df = load_5m_csv(_csv(tmp_path, SAMPLE))
    assert "volume" not in df.columns

def test_timestamp_is_tz_aware(tmp_path):
    df = load_5m_csv(_csv(tmp_path, SAMPLE))
    assert df["timestamp"].dt.tz is not None

def test_bar_close_is_timestamp_plus_5min(tmp_path):
    df = load_5m_csv(_csv(tmp_path, SAMPLE))
    expected = pd.Timestamp("2020-11-09 09:20:00", tz="Asia/Kolkata")
    assert df["bar_close"].iloc[0] == expected

def test_session_date_present(tmp_path):
    df = load_5m_csv(_csv(tmp_path, SAMPLE))
    assert "session_date" in df.columns

def test_bar_in_session_starts_at_1(tmp_path):
    df = load_5m_csv(_csv(tmp_path, SAMPLE))
    assert df["bar_in_session"].iloc[0] == 1

def test_bar_in_session_increments(tmp_path):
    df = load_5m_csv(_csv(tmp_path, SAMPLE))
    assert df["bar_in_session"].tolist() == [1, 2, 3]

def test_deduplicate_keeps_first(tmp_path):
    dup = (
        "open,high,low,close,volume,timestamp\n"
        "12389.0,12430.7999,12383.8999,12410.1,0.0,2020-11-09 09:15:00+05:30\n"
        "99999.0,99999.0,99999.0,99999.0,0.0,2020-11-09 09:15:00+05:30\n"
    )
    df = load_5m_csv(_csv(tmp_path, dup))
    assert len(df) == 1
    assert df["open"].iloc[0] == pytest.approx(12389.0)

def test_output_columns_exact(tmp_path):
    df = load_5m_csv(_csv(tmp_path, SAMPLE))
    expected = {"timestamp", "bar_close", "open", "high", "low", "close",
                "session_date", "bar_in_session"}
    assert set(df.columns) == expected

def test_nonzero_volume_logs_warning(tmp_path, capsys):
    content = (
        "open,high,low,close,volume,timestamp\n"
        "12389.0,12430.0,12383.0,12410.0,500.0,2020-11-09 09:15:00+05:30\n"
    )
    load_5m_csv(_csv(tmp_path, content))
    out = capsys.readouterr().out
    assert "WARNING" in out or "non-zero" in out

def test_real_data_shape():
    if not RAW_CSV.exists():
        pytest.skip("Raw CSV not available")
    df = load_5m_csv(RAW_CSV)
    assert len(df) > 90_000
    assert "volume" not in df.columns
