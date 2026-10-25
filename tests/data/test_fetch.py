"""Veri indirme ve SHA-256 bütünlük testleri."""

import hashlib
from pathlib import Path
import pytest
from bozunma.data.fetch import sha256_file, verify_checksums


def test_sha256_file(tmp_path: Path) -> None:
    test_file = tmp_path / "sample.txt"
    content = b"Mikroskop Bozunma Analizi SHA-256 Test Icerigi"
    test_file.write_bytes(content)

    expected_hash = hashlib.sha256(content).hexdigest()
    actual_hash = sha256_file(test_file)

    assert actual_hash == expected_hash


def test_verify_checksums_all_valid(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    f1 = data_dir / "file1.png"
    f2 = data_dir / "file2.tiff"
    f1.write_bytes(b"data1")
    f2.write_bytes(b"data2")

    manifest = tmp_path / "manifest.sha256"
    h1 = hashlib.sha256(b"data1").hexdigest()
    h2 = hashlib.sha256(b"data2").hexdigest()
    manifest.write_text(f"{h1}  file1.png\n{h2}  file2.tiff\n", encoding="utf-8")

    invalid_files = verify_checksums(data_dir, manifest)
    assert invalid_files == []


def test_verify_checksums_corrupted_and_missing(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    f1 = data_dir / "file1.png"
    f1.write_bytes(b"corrupted_content")

    manifest = tmp_path / "manifest.sha256"
    h1 = hashlib.sha256(b"original_content").hexdigest()
    h2 = hashlib.sha256(b"missing_content").hexdigest()
    manifest.write_text(f"{h1}  file1.png\n{h2}  file2.tiff\n", encoding="utf-8")

    invalid_files = verify_checksums(data_dir, manifest)
    assert "file1.png" in invalid_files
    assert "file2.tiff" in invalid_files
