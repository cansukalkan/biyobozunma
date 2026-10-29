"""Veri bütünlüğü ve dönüşüm kaybı testleri."""

from pathlib import Path

import numpy as np

from bozunma.data.convert import mask_to_yolo_lines
from bozunma.data.validate import conversion_loss, validate_dataset


def test_conversion_loss_clean_square() -> None:
    # 100x100 maske, ortada 20x20 kare (değer: 2)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 2

    lines = mask_to_yolo_lines(mask, class_ids={2: 0}, min_area_px=10)
    assert len(lines) == 1

    loss = conversion_loss(mask, lines, class_value=2)
    # Temiz bir kare için dönüşüm kaybı %5'in altında olmalı
    assert 0.0 <= loss < 5.0


def test_validate_dataset(tmp_path: Path) -> None:
    # Sahte YOLO veri seti oluştur
    root = tmp_path / "dataset"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)

    # 1. Normal etiketli dosya (sınıf 0)
    (root / "images" / "train" / "img1.png").touch()
    (root / "labels" / "train" / "img1.txt").write_text(
        "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n", encoding="utf-8"
    )

    # 2. Boş etiketli dosya
    (root / "images" / "train" / "img2.png").touch()
    (root / "labels" / "train" / "img2.txt").write_text("", encoding="utf-8")

    report = validate_dataset(root)

    assert "empty_labels" in report
    assert "class_counts" in report
    assert "total_images" in report
    assert report["total_images"] == 2
    assert len(report["empty_labels"]) == 1
    assert "img2" in str(report["empty_labels"][0])
    assert report["class_counts"].get(0, 0) == 1
