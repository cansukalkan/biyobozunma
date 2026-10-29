"""Mikroskop kalibrasyon ve fiziksel birim (mm²) dönüştürücü modül."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Calibration:
    """Mikroskop kalibrasyon bilgisi.

    Attributes:
        um_per_px: Piksel başına mikrometre (µm/px).
        status: Kalibrasyon durumu ("MEASURED" veya "PENDING").
    """

    um_per_px: float
    status: str = "PENDING"


def px_to_mm2(area_px: float, cal: Calibration) -> tuple[float, bool]:
    """Piksel cinsinden alanı mm²'ye dönüştürür.

    Kural: Kalibrasyon durumu MEASURED değilse (örn. PENDING),
    sonuç provisional (geçici/kesinleşmemiş) olarak işaretlenir.

    Dönüşüm:
        1 µm = 1e-3 mm
        1 px² = (um_per_px * 1e-3)^2 mm²

    Args:
        area_px: Piksel cinsinden alan.
        cal: Kalibrasyon nesnesi.

    Returns:
        (alan_mm2, is_provisional) çifti.
    """
    scale_mm = cal.um_per_px * 1e-3
    area_mm2 = float(area_px * (scale_mm**2))
    is_provisional = cal.status != "MEASURED"
    return area_mm2, is_provisional


def load_calibration(path: Path) -> Calibration:
    """YAML dosyasından kalibrasyon ayarlarını yükler."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return Calibration(
        um_per_px=float(data.get("um_per_px", 1.0)),
        status=str(data.get("status", "PENDING")).upper(),
    )
