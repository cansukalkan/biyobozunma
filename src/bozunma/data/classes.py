"""Sınıf eşleme ve numune kimliği çözümleme modülü."""

import re
from pathlib import Path

import yaml


def load_class_map(path: Path) -> dict[int, str]:
    """YAML dosyasından ham maske sınıf eşlemesini okur.

    Args:
        path: YAML yapılandırma dosyasının yolu.

    Returns:
        Piksel değeri -> Sınıf adı eşleme sözlüğü (örn. {0: 'background', 2: 'crack'}).
    """
    if not path.exists():
        raise FileNotFoundError(f"Sınıf yapılandırma dosyası bulunamadı: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    raw_classes = data.get("raw_classes", {})
    return {int(k): str(v) for k, v in raw_classes.items()}


def parse_sample_id(filename: str) -> str:
    """Görüntü dosya adından numune kimliğini (sample ID) çıkarır.

    Desteklenen dosya adı şablonları:
      - 'ID14_470_sub_01_Image.png' -> 'ID14'
      - 'ID14_940_Image.png'        -> 'ID14'
      - '14_WG2_470_Img_cropeed.png'-> '14_WG2'
      - '256_Image_29-01-2022.png'  -> '256'
      - 'sample_A_001.png'          -> 'sample_A'

    Args:
        filename: Ayrıştırılacak dosya adı veya dosya yolu.

    Returns:
        Çözümlenen numune kimliği dizesi.

    Raises:
        ValueError: Dosya adından geçerli bir numune kimliği çözülemezse.
    """
    if not filename:
        raise ValueError("Numune kimliği çözülemedi: Dosya adı boş.")

    stem = Path(filename).name

    # 1. 'ID<numara>' formatı (örn. ID14_470_Image... -> ID14)
    m1 = re.match(r"^(ID\d+)_", stem, re.IGNORECASE)
    if m1:
        return m1.group(1).upper()

    # 2. 'sample_<ad>' formatı (örn. sample_A_001.png -> sample_A)
    m2 = re.match(r"^(sample_[A-Za-z0-9]+)_", stem, re.IGNORECASE)
    if m2:
        return m2.group(1)

    # 3. Yalnızca sayı ile başlayan ve ardından Image/Img gelenler (örn. 256_Image... -> 256)
    m3 = re.match(r"^(\d+)_(?:Image|Img|Mask)", stem, re.IGNORECASE)
    if m3:
        return m3.group(1)

    # 4. '<sayi>_<harf_sayi>' formatı (örn. 14_WG2_470_Img... -> 14_WG2)
    m4 = re.match(r"^(\d+_[A-Za-z0-9]+)_", stem)
    if m4:
        return m4.group(1)

    raise ValueError(f"Numune kimliği çözülemedi: {filename}")
