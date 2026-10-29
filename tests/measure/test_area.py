"""Görüntü başına alan ölçüm testleri."""

import numpy as np
import pytest

from bozunma.measure.area import measure_image
from bozunma.measure.calibration import Calibration


def test_measure_image_ratio_and_areas() -> None:
    # 10x10 = 100 px toplam alan
    # 20 piksel çatlak (sınıf 2) -> oran %20 (0.20)
    # 10 piksel gözenek (sınıf 3) -> oran %10 (0.10)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0:2, :] = 2  # 20 px çatlak
    mask[2, 0:10] = 3  # 10 px gözenek

    cal = Calibration(um_per_px=1.0, status="MEASURED")
    result = measure_image(mask, cal)

    assert result["total_pixels"] == 100
    assert result["crack_pixels"] == 20
    assert result["pore_pixels"] == 10
    assert result["crack_ratio"] == pytest.approx(0.20)
    assert result["pore_ratio"] == pytest.approx(0.10)
    assert not result["is_provisional"]
    assert "crack_area_mm2" in result
    assert "pore_area_mm2" in result


def test_measure_image_provisional() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    cal = Calibration(um_per_px=1.0, status="PENDING")
    result = measure_image(mask, cal)

    assert result["is_provisional"]
