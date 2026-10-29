"""Segmentasyon metrikleri (IoU ve Dice) testleri."""

import numpy as np
import pytest

from bozunma.evaluate.metrics import dice, iou


def test_iou_identical_masks() -> None:
    pred = np.zeros((50, 50), dtype=np.uint8)
    true = np.zeros((50, 50), dtype=np.uint8)
    pred[10:30, 10:30] = 2
    true[10:30, 10:30] = 2

    assert iou(pred, true, cls=2) == pytest.approx(1.0)
    assert dice(pred, true, cls=2) == pytest.approx(1.0)


def test_iou_disjoint_masks() -> None:
    pred = np.zeros((50, 50), dtype=np.uint8)
    true = np.zeros((50, 50), dtype=np.uint8)
    pred[0:10, 0:10] = 2
    true[20:30, 20:30] = 2

    assert iou(pred, true, cls=2) == pytest.approx(0.0)
    assert dice(pred, true, cls=2) == pytest.approx(0.0)


def test_iou_both_empty_masks() -> None:
    pred = np.zeros((50, 50), dtype=np.uint8)
    true = np.zeros((50, 50), dtype=np.uint8)

    # İki maskede de sınıf yoksa başarı 1.0 (mükemmel negatif uyum)
    assert iou(pred, true, cls=2) == pytest.approx(1.0)
    assert dice(pred, true, cls=2) == pytest.approx(1.0)


def test_iou_partial_overlap() -> None:
    pred = np.zeros((50, 50), dtype=np.uint8)
    true = np.zeros((50, 50), dtype=np.uint8)
    # pred: 20x20 = 400 px
    pred[10:30, 10:30] = 2
    # true: 20x20 = 400 px, 10x20 = 200 px overlap
    true[10:30, 20:40] = 2

    # intersection = 200, union = 600 -> IoU = 200 / 600 = 1/3
    # dice = 2 * 200 / (400 + 400) = 400 / 800 = 0.5
    assert iou(pred, true, cls=2) == pytest.approx(1.0 / 3.0)
    assert dice(pred, true, cls=2) == pytest.approx(0.5)
