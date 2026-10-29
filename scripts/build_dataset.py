"""YOLO segmentasyon veri seti oluşturucu betik.

data/raw/ altındaki gerçek görüntü ve maskeleri okur,
numune bazlı bölerek data/yolo/ dizinine YOLO formatında yazar.
"""

import sys
from pathlib import Path

# src dizinini sys.path'e ekle
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml
from bozunma.data.writer import find_matching_mask, write_yolo_dataset


def main() -> None:
    raw_dir = Path("data/raw")
    out_dir = Path("data/yolo")
    classes_cfg_path = Path("configs/classes.yaml")

    # Sınıf eşlemesini oku
    raw_to_target = {2: 0, 3: 1}  # Varsayılan: 2(crack)->0, 3(pore)->1
    if classes_cfg_path.exists():
        with open(classes_cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            if "raw_to_target" in cfg:
                raw_to_target = {int(k): int(v) for k, v in cfg["raw_to_target"].items()}

    # Ham görüntüleri tespit et (maskesi olanlar)
    candidates = sorted(raw_dir.glob("*.png")) + sorted(raw_dir.glob("*.jpg"))
    raw_images: list[Path] = []
    for c in candidates:
        if "Mask" not in c.name and "mask" not in c.name:
            mask_p = find_matching_mask(c)
            if mask_p and mask_p.exists():
                raw_images.append(c)

    print(f"[*] Toplam {len(raw_images)} geçerli görüntü-maske çifti bulundu.")
    if not raw_images:
        print("[!] data/raw/ altında veri bulunamadı.")
        return

    # YOLO veri setini oluştur
    print(f"[*] YOLO veri seti oluşturuluyor -> {out_dir}")
    counts = write_yolo_dataset(
        raw_images=raw_images,
        out_root=out_dir,
        class_ids=raw_to_target,
        seed=42,
        ratios=(0.7, 0.15, 0.15),
        min_area_px=10,
    )

    print("[+] YOLO veri seti başarıyla oluşturuldu:")
    for split_name, count in counts.items():
        print(f"    - {split_name}: {count} görsel ve etiket")


if __name__ == "__main__":
    main()
