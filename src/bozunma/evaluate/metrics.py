"""Segmentasyon değerlendirme metrikleri modülü.

Kural: Hem tahminde hem gerçekte ilgili sınıf yoksa IoU, Dice, precision ve recall 1.0 döner.
"""

import numpy as np


def _counts(pred: np.ndarray, true: np.ndarray, cls: int) -> tuple[int, int, int]:
    p = pred == cls
    t = true == cls
    tp = int(np.logical_and(p, t).sum())
    fp = int(np.logical_and(p, ~t).sum())
    fn = int(np.logical_and(~p, t).sum())
    return tp, fp, fn


def iou(pred: np.ndarray, true: np.ndarray, cls: int) -> float:
    """Intersection over Union (Jaccard)."""
    tp, fp, fn = _counts(pred, true, cls)
    denom = tp + fp + fn
    return 1.0 if denom == 0 else tp / denom


def dice(pred: np.ndarray, true: np.ndarray, cls: int) -> float:
    """Dice benzerlik katsayısı (F1)."""
    tp, fp, fn = _counts(pred, true, cls)
    denom = 2 * tp + fp + fn
    return 1.0 if denom == 0 else 2 * tp / denom


def precision(pred: np.ndarray, true: np.ndarray, cls: int) -> float:
    """Kesinlik: tahmin edilen sınıf piksellerinin doğru olan oranı."""
    tp, fp, fn = _counts(pred, true, cls)
    if tp + fp == 0:
        return 1.0 if fn == 0 else 0.0
    return tp / (tp + fp)


def recall(pred: np.ndarray, true: np.ndarray, cls: int) -> float:
    """Duyarlılık: gerçek sınıf piksellerinin yakalanan oranı."""
    tp, fp, fn = _counts(pred, true, cls)
    if tp + fn == 0:
        return 1.0 if fp == 0 else 0.0
    return tp / (tp + fn)


def pixel_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    """Tüm sınıflar üzerinden doğru sınıflanan piksel oranı."""
    if true.size == 0:
        return 1.0
    return float((pred == true).mean())


def confusion_matrix(pred: np.ndarray, true: np.ndarray, classes: list[int]) -> list[list[int]]:
    """Satır = gerçek sınıf, sütun = tahmin sınıfı."""
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    cm = np.zeros((n, n), dtype=np.int64)
    t = true.ravel()
    p = pred.ravel()
    valid = np.isin(t, classes) & np.isin(p, classes)
    ti = np.vectorize(idx.get)(t[valid]) if valid.any() else np.array([], dtype=int)
    pi = np.vectorize(idx.get)(p[valid]) if valid.any() else np.array([], dtype=int)
    np.add.at(cm, (ti, pi), 1)
    return cm.tolist()


def segmentation_report(
    pred: np.ndarray, true: np.ndarray, classes: dict[int, str], all_classes: list[int]
) -> dict:
    """Sınıf bazlı metrikler, piksel doğruluğu ve karışıklık matrisi."""
    per_class = {
        name: {
            "iou": iou(pred, true, c),
            "dice": dice(pred, true, c),
            "precision": precision(pred, true, c),
            "recall": recall(pred, true, c),
            "support_px": int((true == c).sum()),
        }
        for c, name in classes.items()
    }
    return {
        "per_class": per_class,
        "pixel_accuracy": pixel_accuracy(pred, true),
        "confusion_matrix": confusion_matrix(pred, true, all_classes),
        "labels": all_classes,
    }
