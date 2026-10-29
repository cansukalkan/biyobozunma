"""Mikroskop kalibrasyon ve pikselden mm²'ye dönüşüm testleri."""

import pytest

from bozunma.measure.calibration import Calibration, px_to_mm2


def test_px_to_mm2_measured() -> None:
    # 1 px = 2.0 µm = 0.002 mm
    # 1 px² = 4.0 µm² = 0.000004 mm² = 4e-6 mm²
    cal = Calibration(um_per_px=2.0, status="MEASURED")
    val_mm2, is_provisional = px_to_mm2(1000.0, cal)

    # 1000 px² = 1000 * 4e-6 = 0.004 mm²
    assert val_mm2 == pytest.approx(0.004)
    assert not is_provisional


def test_px_to_mm2_pending() -> None:
    cal = Calibration(um_per_px=1.0, status="PENDING")
    val_mm2, is_provisional = px_to_mm2(100.0, cal)

    # PENDING durumunda provisional bayrağı True olmalı
    assert is_provisional
    assert val_mm2 == pytest.approx(100.0 * (1e-3) ** 2)
