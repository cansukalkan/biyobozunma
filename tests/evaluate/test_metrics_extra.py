"""Ek metrik testleri: precision, recall, piksel doğruluğu, karışıklık matrisi."""

import numpy as np
import pytest

from bozunma.evaluate.metrics import (
    confusion_matrix,
    pixel_accuracy,
    precision,
    recall,
    segmentation_report,
)


def _pair() -> tuple[np.ndarray, np.ndarray]:
    true = np.zeros((10, 10), dtype=np.uint8)
    pred = np.zeros((10, 10), dtype=np.uint8)
    true[0:4, 0:5] = 2  # 20 px gerçek çatlak
    pred[0:4, 0:3] = 2  # 12 px doğru
    pred[9, 0:4] = 2  # 4 px yanlış pozitif
    return pred, true


def test_precision_recall() -> None:
    pred, true = _pair()
    assert precision(pred, true, 2) == pytest.approx(12 / 16)
    assert recall(pred, true, 2) == pytest.approx(12 / 20)


def test_precision_recall_empty() -> None:
    z = np.zeros((4, 4), dtype=np.uint8)
    assert precision(z, z, 3) == 1.0
    assert recall(z, z, 3) == 1.0


def test_pixel_accuracy() -> None:
    pred, true = _pair()
    # 100 pikselden 8 FN + 4 FP yanlış
    assert pixel_accuracy(pred, true) == pytest.approx(88 / 100)


def test_confusion_matrix() -> None:
    pred, true = _pair()
    cm = confusion_matrix(pred, true, [0, 2])
    assert cm == [[76, 4], [8, 12]]


def test_report_structure() -> None:
    pred, true = _pair()
    rep = segmentation_report(pred, true, {2: "crack"}, [0, 2])
    assert set(rep["per_class"]["crack"]) == {"iou", "dice", "precision", "recall", "support_px"}
    assert rep["per_class"]["crack"]["support_px"] == 20
