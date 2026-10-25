"""Maskeden YOLO çokgen (segmentasyon) formatına dönüşüm modülü."""

import cv2
import numpy as np


def mask_to_yolo_lines(
    mask: np.ndarray, class_ids: dict[int, int], min_area_px: int = 10
) -> list[str]:
    """Bir piksel maskesini YOLOv8/v11 segmentasyon formatındaki satırlara dönüştürür.

    Her satır şu formattadır:
      <hedef_sinif_id> x1 y1 x2 y2 ... xn yn
    Koordinatlar [0, 1] aralığına normalize edilmiştir.

    Args:
        mask: (H, W) boyutunda tam sayı piksel değerlerine sahip maske dizisi.
        class_ids: Ham maske değeri -> Hedef YOLO sınıf kimliği eşleme sözlüğü
            (örn. {2: 0, 3: 1}).
        min_area_px: Çokgen olarak dahil edilecek minimum piksel alanı.

    Returns:
        YOLO etiket dosyası için çokgen satırları listesi.
    """
    lines: list[str] = []
    if mask is None or mask.size == 0:
        return lines

    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        return lines

    for raw_val, target_cls in class_ids.items():
        # İlgili sınıf için ikili maske oluştur
        binary_mask = (mask == raw_val).astype(np.uint8)
        if not np.any(binary_mask):
            continue

        # Yalnızca dış konturları bul
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area_px:
                continue

            # Kontur en az 3 nokta içermelidir (çokgen için)
            if len(cnt) < 3:
                continue

            # Koordinatları normalize et (x / w, y / h)
            pts = cnt.reshape(-1, 2).astype(np.float32)
            norm_pts: list[float] = []
            for pt in pts:
                nx = float(np.clip(pt[0] / w, 0.0, 1.0))
                ny = float(np.clip(pt[1] / h, 0.0, 1.0))
                norm_pts.extend([nx, ny])

            # En az 3 nokta (6 koordinat)
            if len(norm_pts) >= 6:
                coords_str = " ".join(f"{c:.6f}" for c in norm_pts)
                lines.append(f"{target_cls} {coords_str}")

    return lines
