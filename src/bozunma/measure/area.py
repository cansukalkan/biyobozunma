"""Görüntü başına çatlak ve gözenek alan ölçüm modülü."""

import numpy as np

from bozunma.measure.calibration import Calibration, px_to_mm2


def measure_image(
    mask: np.ndarray,
    cal: Calibration,
    crack_val: int = 2,
    pore_val: int = 3,
) -> dict:
    """Tekil bir segmentasyon maskesi üzerinde piksel ve fiziksel alan ölçümü yapar.

    Args:
        mask: Segmentasyon maskesi dizisi (H x W).
        cal: Kalibrasyon nesnesi.
        crack_val: Çatlak sınıf değeri (varsayılan: 2).
        pore_val: Gözenek sınıf değeri (varsayılan: 3).

    Returns:
        Sözlük:
          - total_pixels: Toplam piksel sayısı
          - crack_pixels: Çatlak piksel sayısı
          - pore_pixels: Gözenek piksel sayısı
          - crack_ratio: Çatlak piksel oranı [0.0, 1.0]
          - pore_ratio: Gözenek piksel oranı [0.0, 1.0]
          - crack_area_mm2: Çatlak alanı (mm²)
          - pore_area_mm2: Gözenek alanı (mm²)
          - is_provisional: Kalibrasyon durumuna göre kesinleşmeme durumu
    """
    total_px = int(mask.size)
    if total_px == 0:
        return {
            "total_pixels": 0,
            "crack_pixels": 0,
            "pore_pixels": 0,
            "crack_ratio": 0.0,
            "pore_ratio": 0.0,
            "crack_area_mm2": 0.0,
            "pore_area_mm2": 0.0,
            "is_provisional": cal.status != "MEASURED",
        }

    crack_px = int(np.sum(mask == crack_val))
    pore_px = int(np.sum(mask == pore_val))

    crack_ratio = float(crack_px / total_px)
    pore_ratio = float(pore_px / total_px)

    crack_mm2, prov1 = px_to_mm2(float(crack_px), cal)
    pore_mm2, prov2 = px_to_mm2(float(pore_px), cal)

    return {
        "total_pixels": total_px,
        "crack_pixels": crack_px,
        "pore_pixels": pore_px,
        "crack_ratio": crack_ratio,
        "pore_ratio": pore_ratio,
        "crack_area_mm2": crack_mm2,
        "pore_area_mm2": pore_mm2,
        "is_provisional": prov1 or prov2,
    }
