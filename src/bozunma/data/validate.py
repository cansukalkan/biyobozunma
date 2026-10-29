"""Veri bütünlüğü ve dönüşüm kaybı doğrulama modülü."""

from pathlib import Path

import cv2
import numpy as np


def conversion_loss(mask: np.ndarray, lines: list[str], class_value: int) -> float:
    """Maskeden üretilen çokgen etiketlerinin alan kaybını (yüzde) hesaplar.

    Çokgen koordinatlarını maske boyutunda ikili bir tuval üzerine yeniden çizer,
    orijinal maske sınıf alanı ile yeniden oluşturulan alan farkını yüzde olarak döner.

    Args:
        mask: Orijinal maske dizisi (H x W).
        lines: YOLO formatındaki çokgen satırları ("cls x1 y1 x2 y2 ...").
        class_value: Maskedeki hedef sınıf değeri (örn. 2 = çatlak).

    Returns:
        Yüzde olarak alan dönüşüm kaybı (|orijinal - yeniden| / orijinal * 100.0).
    """
    orig_area = float(np.sum(mask == class_value))
    if orig_area == 0:
        return 0.0 if not lines else 100.0

    h, w = mask.shape[:2]
    canvas = np.zeros((h, w), dtype=np.uint8)

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        coords = [float(x) for x in parts[1:]]
        # x, y çiftlerini piksel koordinatlarına dönüştür
        pts = []
        for i in range(0, len(coords) - 1, 2):
            px = int(round(coords[i] * w))
            py = int(round(coords[i + 1] * h))
            pts.append([px, py])

        if len(pts) >= 3:
            pts_arr = np.array([pts], dtype=np.int32)
            cv2.fillPoly(canvas, pts_arr, 1)

    reconstructed_area = float(np.sum(canvas == 1))
    loss = (abs(orig_area - reconstructed_area) / orig_area) * 100.0
    return float(loss)


def validate_dataset(root: Path) -> dict:
    """YOLO veri setini tarayarak etiket doğruluğu ve istatistikleri raporlar.

    Args:
        root: YOLO veri seti kök dizini (images/ ve labels/ içeren).

    Returns:
        Sözlük:
          - empty_labels: Boş etiket dosyalarının listesi
          - class_counts: Sınıf kimliği başına toplam çokgen sayısı
          - total_images: Toplam görsel sayısı
          - total_labels: Toplam etiket dosyası sayısı
    """
    empty_labels: list[Path] = []
    class_counts: dict[int, int] = {}

    labels_dir = root / "labels"
    label_files = list(labels_dir.rglob("*.txt")) if labels_dir.exists() else []

    for lbl_file in label_files:
        content = lbl_file.read_text(encoding="utf-8").strip()
        if not content:
            empty_labels.append(lbl_file)
            continue

        for line in content.splitlines():
            parts = line.strip().split()
            if parts:
                try:
                    cls_id = int(parts[0])
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                except ValueError:
                    pass

    images_dir = root / "images"
    image_files = []
    if images_dir.exists():
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
            image_files.extend(list(images_dir.rglob(ext)))

    return {
        "empty_labels": empty_labels,
        "class_counts": class_counts,
        "total_images": len(image_files),
        "total_labels": len(label_files),
    }
