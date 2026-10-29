"""Morfoloji testleri."""

import cv2
import numpy as np
import pytest

from bozunma.measure.morphology import crack_morphology, pore_morphology


def test_crack_straight_line() -> None:
    m = np.zeros((50, 100), dtype=np.uint8)
    m[20:23, 10:90] = 2  # 80 px uzun, 3 px geniş
    r = crack_morphology(m, um_per_px=2.0)
    assert r["count"] == 1
    assert 75 <= r["length_px"] <= 82
    assert r["mean_width_px"] == pytest.approx(3.0, abs=0.3)
    assert r["length_um"] == pytest.approx(r["length_px"] * 2.0)


def test_crack_empty() -> None:
    assert crack_morphology(np.zeros((5, 5), np.uint8))["count"] == 0


def test_pore_circle() -> None:
    m = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(m, (30, 30), 10, 3, -1)
    cv2.circle(m, (70, 70), 5, 3, -1)
    r = pore_morphology(m)
    assert r["count"] == 2
    assert r["max_diameter_px"] == pytest.approx(20, abs=1.5)
    assert r["mean_circularity"] >= 0.78
