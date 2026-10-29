"""Segmentasyon değerlendirme metrikleri (IoU ve Dice) modülü."""

import numpy as np


def iou(pred: np.ndarray, true: np.ndarray, cls: int) -> float:
    """Belirli bir sınıf için Intersection over Union (IoU / Jaccard İndeksi) hesaplar.

    Kural: İki maskede de ilgili sınıf bulunmuyorsa (boşsa), sonuç 1.0 (mükemmel uyum) döner.

    Args:
        pred: Model tahmin maskesi (H x W).
        true: Gerçek zemin doğru (ground truth) maske (H x W).
        cls: Hesaplanacak hedef sınıf değeri.

    Returns:
        IoU değeri [0.0, 1.0].
    """
    pred_bin = pred == cls
    true_bin = true == cls

    intersection = np.logical_and(pred_bin, true_bin).sum()
    union = np.logical_or(pred_bin, true_bin).sum()

    if union == 0:
        return 1.0

    return float(intersection / union)


def dice(pred: np.ndarray, true: np.ndarray, cls: int) -> float:
    """Belirli bir sınıf için Dice Benzerlik Katsayısı (F1 Skoru) hesaplar.

    Kural: İki maskede de ilgili sınıf bulunmuyorsa (boşsa), sonuç 1.0 döner.

    Args:
        pred: Model tahmin maskesi (H x W).
        true: Gerçek zemin doğru (ground truth) maske (H x W).
        cls: Hesaplanacak hedef sınıf değeri.

    Returns:
        Dice katsayısı [0.0, 1.0].
    """
    pred_bin = pred == cls
    true_bin = true == cls

    intersection = np.logical_and(pred_bin, true_bin).sum()
    total = pred_bin.sum() + true_bin.sum()

    if total == 0:
        return 1.0

    return float(2.0 * intersection / total)
