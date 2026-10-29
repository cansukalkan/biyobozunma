"""Zaman serisi ve girdi doğrulama testleri."""

import pytest

from bozunma.kinetics.timeseries import build_series, validate_row


def test_validate_row_valid() -> None:
    row = {
        "sample_id": "SAMPLE_01",
        "timestamp": "2026-10-29T10:00:00",
        "crack_area_mm2": 0.045,
        "pore_area_mm2": 0.012,
    }
    # Geçerli satırda hata fırlatılmamalı
    validate_row(row)


def test_validate_row_empty_sample_id() -> None:
    row = {
        "sample_id": "",
        "timestamp": "2026-10-29T10:00:00",
        "crack_area_mm2": 0.045,
    }
    with pytest.raises(ValueError, match="Numune kimliği boş olamaz"):
        validate_row(row)


def test_validate_row_negative_area() -> None:
    row = {
        "sample_id": "SAMPLE_01",
        "timestamp": "2026-10-29T10:00:00",
        "crack_area_mm2": -0.01,
    }
    with pytest.raises(ValueError, match="Alan değeri negatif olamaz"):
        validate_row(row)


def test_validate_row_invalid_timestamp() -> None:
    row = {
        "sample_id": "SAMPLE_01",
        "timestamp": "invalid-date",
        "crack_area_mm2": 0.05,
    }
    with pytest.raises(ValueError, match="Tarih / zaman biçimi geçersiz"):
        validate_row(row)


def test_build_series_sorted() -> None:
    rows = [
        {
            "sample_id": "ID2",
            "timestamp": "2026-10-29T12:00:00",
            "crack_area_mm2": 0.20,
        },
        {
            "sample_id": "ID1",
            "timestamp": "2026-10-29T14:00:00",
            "crack_area_mm2": 0.15,
        },
        {
            "sample_id": "ID1",
            "timestamp": "2026-10-29T10:00:00",
            "crack_area_mm2": 0.10,
        },
    ]

    df = build_series(rows)
    assert len(df) == 3
    # ID1 önce gelmeli ve kendi içinde tarihe göre sıralı olmalı
    assert df.iloc[0]["sample_id"] == "ID1"
    assert df.iloc[0]["timestamp"].hour == 10
    assert df.iloc[1]["sample_id"] == "ID1"
    assert df.iloc[1]["timestamp"].hour == 14
    assert df.iloc[2]["sample_id"] == "ID2"
