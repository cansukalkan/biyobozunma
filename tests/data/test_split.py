"""Numune bazlı veri seti bölme testleri."""

from pathlib import Path

import pytest

from bozunma.data.split import split_by_sample


def test_split_by_sample_no_leakage() -> None:
    # 4 farklı numuneye ait toplam 10 görüntü
    files = [
        Path("ID14_470_sub_01_Image.png"),
        Path("ID14_470_sub_02_Image.png"),
        Path("ID14_470_sub_03_Image.png"),
        Path("ID12_46_sub_01_Image.png"),
        Path("ID12_46_sub_02_Image.png"),
        Path("14_WG2_01_Img.png"),
        Path("14_WG2_02_Img.png"),
        Path("256_Image_01.png"),
        Path("256_Image_02.png"),
        Path("256_Image_03.png"),
    ]

    splits = split_by_sample(files, seed=42, ratios=(0.5, 0.25, 0.25))

    assert "train" in splits
    assert "val" in splits
    assert "test" in splits

    # Toplam dosya sayısı korunmalı
    total_split_files = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total_split_files == len(files)

    # Numune sızıntısı olmamalı: Aynı numunenin dosyaları farklı bölmelere dağılmamalı
    from bozunma.data.classes import parse_sample_id

    train_samples = {parse_sample_id(f.name) for f in splits["train"]}
    val_samples = {parse_sample_id(f.name) for f in splits["val"]}
    test_samples = {parse_sample_id(f.name) for f in splits["test"]}

    assert train_samples.isdisjoint(val_samples)
    assert train_samples.isdisjoint(test_samples)
    assert val_samples.isdisjoint(test_samples)


def test_split_by_sample_reproducibility() -> None:
    files = [Path(f"ID{i}_00_Image.png") for i in range(1, 20)]

    split1 = split_by_sample(files, seed=123, ratios=(0.7, 0.15, 0.15))
    split2 = split_by_sample(files, seed=123, ratios=(0.7, 0.15, 0.15))

    assert split1 == split2


def test_split_by_sample_invalid_ratios() -> None:
    files = [Path("ID1_00_Image.png")]
    with pytest.raises(ValueError, match="Oranlar toplamı"):
        split_by_sample(files, seed=42, ratios=(0.5, 0.5, 0.5))
