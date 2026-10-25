"""Maskeden YOLO çokgen (polygon) etiketi üretim testleri."""

import numpy as np

from bozunma.data.convert import mask_to_yolo_lines


def test_mask_to_yolo_lines_square_crack() -> None:
    # 100x100 boyutunda bir maske, ortasında 20x20 kare şeklinde çatlak (değer: 2)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 2

    # Ham değer 2 -> YOLO sınıf 0
    class_ids = {2: 0}

    lines = mask_to_yolo_lines(mask, class_ids, min_area_px=10)
    assert len(lines) == 1

    parts = lines[0].strip().split()
    cls_id = int(parts[0])
    coords = [float(x) for x in parts[1:]]

    # Sınıf 0 olmalı
    assert cls_id == 0

    # Köşeler (x, y) çiftleri olmalı ve en az 3 nokta (6 koordinat) içermeli
    assert len(coords) >= 6
    assert len(coords) % 2 == 0

    # Tüm koordinatlar [0, 1] aralığında olmalı
    for c in coords:
        assert 0.0 <= c <= 1.0

    # X koordinatları ~0.40 ile ~0.60 civarında olmalı
    xs = coords[0::2]
    ys = coords[1::2]
    assert min(xs) >= 0.38 and max(xs) <= 0.62
    assert min(ys) >= 0.38 and max(ys) <= 0.62


def test_mask_to_yolo_lines_noise_filtering() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Küçük gürültü noktası (2x2 = 4 piksel < min_area_px=10)
    mask[10:12, 10:12] = 2
    # Büyük gözenek (10x10 = 100 piksel >= min_area_px=10, değer: 3)
    mask[50:60, 50:60] = 3

    class_ids = {2: 0, 3: 1}

    lines = mask_to_yolo_lines(mask, class_ids, min_area_px=10)
    # Yalnızca gözenek (sınıf 1) üretilmeli, gürültü filtrelenmeli
    assert len(lines) == 1
    assert lines[0].startswith("1 ")


def test_mask_to_yolo_lines_empty_mask() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    class_ids = {2: 0, 3: 1}

    lines = mask_to_yolo_lines(mask, class_ids, min_area_px=10)
    assert lines == []
