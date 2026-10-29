"""YOLO segmentasyon veri seti yazıcısı modülü."""

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from bozunma.data.convert import mask_to_yolo_lines
from bozunma.data.split import split_by_sample


def find_matching_mask(img_path: Path) -> Path | None:
    """Verilen görüntü dosyasının aynı dizindeki maske eşini bulur.

    Örnek:
      'ID14_470_sub_01_Image.png' -> 'ID14_470_sub_01_Mask.png'
      '14_WG2_470_Img_cropeed.png' -> '14_WG2_470_Mask_cropped.tiff'
    """
    stem = img_path.stem
    parent = img_path.parent

    # En yaygın isim dönüşümleri
    replacements = [
        ("Image", "Mask"),
        ("Img", "Mask"),
        ("image", "mask"),
        ("img", "mask"),
        ("cropeed", "cropped"),
    ]

    target_stem = stem
    for old, new in replacements:
        target_stem = target_stem.replace(old, new)

    if target_stem == stem:
        return None

    # Olası uzantıları tara (.png, .tiff, .tif, .bmp)
    for ext in [img_path.suffix, ".png", ".tiff", ".tif", ".bmp"]:
        candidate = parent / f"{target_stem}{ext}"
        if candidate.exists() and candidate.resolve() != img_path.resolve():
            return candidate

    return None


def write_yolo_dataset(
    raw_images: list[Path],
    out_root: Path,
    class_ids: dict[int, int],
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    min_area_px: int = 10,
) -> dict[str, int]:
    """Ham görüntü ve maskelerden numune bazlı bölünmüş YOLOv8/v11 veri seti oluşturur.

    Dizin yapısı:
      out_root/
        images/train, images/val, images/test
        labels/train, labels/val, labels/test

    Args:
        raw_images: Ham görüntü dosyalarının listesi.
        out_root: Veri setinin yazılacağı hedef kök klasör.
        class_ids: Ham maske değeri -> Hedef YOLO sınıf kimliği eşleme sözlüğü.
        seed: Rastgelelik tohumu (bölme tekrarlanabilirliği için).
        ratios: (train, val, test) bölme oranları.
        min_area_px: Çokgen olarak dahil edilecek minimum piksel alanı.

    Returns:
        Bölme başına yazılan görüntü sayısı sözlüğü: {'train': int, 'val': int, 'test': int}.
    """
    splits = split_by_sample(raw_images, seed=seed, ratios=ratios)
    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for split_name, img_list in splits.items():
        img_dir = out_root / "images" / split_name
        lbl_dir = out_root / "labels" / split_name

        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in img_list:
            # 1. Görüntüyü kopyala
            dest_img = img_dir / img_path.name
            shutil.copy2(img_path, dest_img)

            # 2. Maskeyi bul ve çokgen etiketlerini üret
            mask_path = find_matching_mask(img_path)
            lines: list[str] = []

            if mask_path is not None and mask_path.exists():
                mask_arr = np.array(Image.open(mask_path))
                lines = mask_to_yolo_lines(
                    mask_arr, class_ids=class_ids, min_area_px=min_area_px
                )

            # 3. Etiket dosyasını yaz (.txt)
            label_file = lbl_dir / f"{img_path.stem}.txt"
            label_content = "\n".join(lines) + ("\n" if lines else "")
            label_file.write_text(label_content, encoding="utf-8")

            counts[split_name] += 1

    return counts
