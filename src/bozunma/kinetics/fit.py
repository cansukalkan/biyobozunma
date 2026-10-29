"""Bozunma kinetiği ve birinci derece üstel uydurma modülü."""

import numpy as np
from scipy.optimize import curve_fit


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

    # Başlangıç parametre tahmini
    c0_init = float(c_arr[0]) if len(c_arr) > 0 and c_arr[0] > 0 else 1.0
    p0 = [c0_init, 0.01]

    popt, pcov = curve_fit(exponential_decay, t_arr, c_arr, p0=p0, maxfev=10000)

    perr = np.sqrt(np.diag(pcov))
    k_err = float(perr[1]) if len(perr) > 1 and not np.isinf(perr[1]) else 0.0

    return {
        "c0": float(popt[0]),
        "k": float(popt[1]),
        "k_err": k_err,
    }
