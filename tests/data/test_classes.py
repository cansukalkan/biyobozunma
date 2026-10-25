"""Sınıf eşleme ve numune kimliği çözümleme testleri."""

from pathlib import Path

import pytest

from bozunma.data.classes import load_class_map, parse_sample_id


def test_load_class_map(tmp_path: Path) -> None:
    yaml_file = tmp_path / "classes.yaml"
    yaml_file.write_text(
        """
raw_classes:
  0: background
  1: matrix
  2: crack
  3: pore
""",
        encoding="utf-8",
    )

    class_map = load_class_map(yaml_file)
    assert class_map == {0: "background", 1: "matrix", 2: "crack", 3: "pore"}


def test_parse_sample_id_valid() -> None:
    # Farklı CrackenPy dosya adı formatları
    assert parse_sample_id("ID14_470_sub_01_Image.png") == "ID14"
    assert parse_sample_id("ID14_940_Image.png") == "ID14"
    assert parse_sample_id("14_WG2_470_Img_cropeed.png") == "14_WG2"
    assert parse_sample_id("256_Image_29-01-2022 02-36-02.png") == "256"
    assert parse_sample_id("ID12_46_Image_sub_img_001.png") == "ID12"
    assert parse_sample_id("sample_A_001.png") == "sample_A"


def test_parse_sample_id_invalid() -> None:
    with pytest.raises(ValueError, match="Numune kimliği çözülemedi"):
        parse_sample_id("invalidfile.png")

    with pytest.raises(ValueError, match="Numune kimliği çözülemedi"):
        parse_sample_id("")
