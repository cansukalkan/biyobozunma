"""Yol güvenliği ve dizin aşımı (path traversal) testleri."""

from pathlib import Path

import pytest

from bozunma.data.security import validate_safe_path


def test_safe_path_valid(tmp_path: Path) -> None:
    base = tmp_path / "data"
    base.mkdir()
    valid_file = "sample_01.png"

    resolved = validate_safe_path(valid_file, base_dir=base)
    assert resolved == base / valid_file


def test_safe_path_traversal_rejected(tmp_path: Path) -> None:
    base = tmp_path / "data"
    base.mkdir()

    # ../ içeren girdi reddedilmeli
    with pytest.raises(ValueError, match="Geçersiz veya güvensiz dosya yolu"):
        validate_safe_path("../etc/passwd", base_dir=base)

    with pytest.raises(ValueError, match="Geçersiz veya güvensiz dosya yolu"):
        validate_safe_path("sub/../../secret.txt", base_dir=base)


def test_safe_path_absolute_rejected(tmp_path: Path) -> None:
    base = tmp_path / "data"
    base.mkdir()

    # Mutlak yol reddedilmeli
    with pytest.raises(ValueError, match="Mutlak yollar kabul edilmez"):
        validate_safe_path("C:/Windows/System32", base_dir=base)
