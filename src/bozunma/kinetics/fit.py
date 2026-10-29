"""Bozunma kinetiği ve birinci derece üstel uydurma modülü."""

import numpy as np


def exponential_decay(t: np.ndarray, c0: float, k: float) -> np.ndarray:
    """Birinci derece üstel bozunma fonksiyonu: C(t) = c0 * exp(-k * t)."""
    return c0 * np.exp(-k * t)


def fit_first_order(t: np.ndarray, c: np.ndarray) -> dict:
    """Zaman ve konsantrasyon/alan verisine birinci derece üstel bozunma modeli uydurur.

    Model: C(t) = c0 * exp(-k * t)

    Args:
        t: Zaman dizisi (gün, saat vb.).
        c: Ölçülen büyüklük dizisi (ör. sağlam matris alanı veya kalan kütle).

    Returns:
        Sözlük:
          - c0: Başlangıç değeri kestirimi
          - k: Bozunma hız sabiti (1/zaman)
          - k_err: Hız sabiti standart hatası
    """
    t_arr = np.asarray(t, dtype=float)
    c_arr = np.asarray(c, dtype=float)

    if len(t_arr) < 2:
        raise ValueError("Uydurma için en az 2 veri noktası gereklidir.")

    # Pozitif veri noktaları üzerinde log-lineer regresyon: ln(C) = ln(c0) - k * t
    valid = c_arr > 0
    if valid.sum() < 2:
        p = np.polyfit(t_arr, c_arr, 1)
        return {"c0": float(c_arr[0]), "k": max(0.0, float(-p[0])), "k_err": 0.0}

    t_v = t_arr[valid]
    log_c = np.log(c_arr[valid])

    if len(t_v) > 2:
        poly, cov = np.polyfit(t_v, log_c, 1, cov=True)
        k_err = float(np.sqrt(cov[0, 0])) if cov is not None else 0.0
    else:
        poly = np.polyfit(t_v, log_c, 1, cov=False)
        cov = None
        k_err = 0.0

    slope = poly[0]
    intercept = poly[1]

    k = float(-slope)
    c0 = float(np.exp(intercept))

    # Tahmin edilen C(t) ve R^2 / t_1/2 hesaplama
    c_pred = c0 * np.exp(-k * t_arr)
    ss_res = float(np.sum((c_arr - c_pred) ** 2))
    ss_tot = float(np.sum((c_arr - np.mean(c_arr)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    half_life = float(np.log(2.0) / k) if k > 1e-9 else None

    return {
        "c0": c0,
        "k": k,
        "k_err": k_err,
        "r2": max(0.0, min(1.0, r2)),
        "half_life_days": half_life,
    }

