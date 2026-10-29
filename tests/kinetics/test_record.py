"""Değişmez (immutable/tamper-evident) analiz kaydı testleri."""

import json
from pathlib import Path

import pandas as pd

from bozunma.kinetics.record import verify_record, write_record


def test_write_and_verify_record_valid(tmp_path: Path) -> None:
    inputs = pd.DataFrame([
        {"sample_id": "S1", "timestamp": "2026-10-29T10:00:00", "area_mm2": 1.25},
        {"sample_id": "S1", "timestamp": "2026-10-29T11:00:00", "area_mm2": 1.10},
    ])
    results = {
        "c0": 1.25,
        "k": 0.05,
        "k_err": 0.002,
    }

    record_path = write_record(inputs, results, tmp_path)
    assert record_path.exists()

    # Değiştirilmemiş kayıt doğrulanmalı
    assert verify_record(record_path) is True


def test_verify_record_tampered_fails(tmp_path: Path) -> None:
    inputs = pd.DataFrame([
        {"sample_id": "S1", "timestamp": "2026-10-29T10:00:00", "area_mm2": 1.25},
    ])
    results = {"k": 0.05}

    record_path = write_record(inputs, results, tmp_path)

    # Dosya içeriğini açıp tek bir değeri değiştir
    with open(record_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Sonucu tahrif et (0.05 -> 0.09)
    data["results"]["k"] = 0.09
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    # Değiştirilmiş kayıt doğrulamadan geçmemeli
    assert verify_record(record_path) is False
