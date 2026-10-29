"""YOLO veri seti yazıcı testleri."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from bozunma.data.writer import write_yolo_dataset


def test_write_yolo_dataset(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "yolo_dataset"

    # 10 adet sentetik görüntü ve maske oluştur (2 farklı numune: ID1, ID2)
    raw_images: list[Path] = []
    for s_idx, sample in enumerate(["ID1", "ID2"]):
        for i in range(5):
            img_path = raw_dir / f"{sample}_{i:02d}_Image.png"
            mask_path = raw_dir / f"{sample}_{i:02d}_Mask.png"

            # 64x64 gri görsel
            img = Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8))
            img.save(img_path)

            # Maske: ortada küçük bir çatlak (değer: 2)
            mask_arr = np.zeros((64, 64), dtype=np.uint8)
            mask_arr[20:40, 20:40] = 2
            Image.fromarray(mask_arr).save(mask_path)

            raw_images.append(img_path)

    # Ham değer 2 -> YOLO sınıf 0
    class_ids = {2: 0}

    counts = write_yolo_dataset(
        raw_images=raw_images,
        out_root=out_dir,
        class_ids=class_ids,
        seed=42,
        ratios=(0.5, 0.25, 0.25),
    )

    # Toplam görüntü sayısı 10 olmalı
    total_written = sum(counts.values())
    assert total_written == 10

    # Her bölmede görüntü ve etiket dosyaları eşleşmeli
    for split in ["train", "val", "test"]:
        split_img_dir = out_dir / "images" / split
        split_lbl_dir = out_dir / "labels" / split

        if counts[split] > 0:
            assert split_img_dir.exists()
            assert split_lbl_dir.exists()

            img_files = list(split_img_dir.glob("*.png"))
            lbl_files = list(split_lbl_dir.glob("*.txt"))

            assert len(img_files) == counts[split]
            assert len(lbl_files) == counts[split]

            # Etiket dosyalarının içeriği kontrol edilir
            for lbl_p in lbl_files:
                content = lbl_p.read_text(encoding="utf-8").strip()
                assert content.startswith("0 ")
