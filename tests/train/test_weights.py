"""Model ağırlık bütünlüğü ve SHA-256 doğrulama testleri."""

import hashlib
from pathlib import Path

import pytest

from bozunma.train.weights import WeightIntegrityError, verify_weights


def test_verify_weights_valid(tmp_path: Path) -> None:
    weights_file = tmp_path / "model.pt"
    content = b"fake-weights-binary-data"
    weights_file.write_bytes(content)

    sha = hashlib.sha256(content).hexdigest()
    manifest = tmp_path / "MANIFEST.sha256"
    manifest.write_text(f"{sha}  model.pt\n", encoding="utf-8")

    verified = verify_weights(weights_file, manifest)
    assert verified == weights_file


def test_verify_weights_mismatch(tmp_path: Path) -> None:
    weights_file = tmp_path / "model.pt"
    weights_file.write_bytes(b"tampered-content")

    manifest = tmp_path / "MANIFEST.sha256"
    manifest.write_text("0123456789abcdef  model.pt\n", encoding="utf-8")

    with pytest.raises(WeightIntegrityError, match="Bütünlük doğrulaması başarısız"):
        verify_weights(weights_file, manifest)


def test_verify_weights_not_in_manifest(tmp_path: Path) -> None:
    weights_file = tmp_path / "unregistered.pt"
    weights_file.write_bytes(b"unregistered-content")

    manifest = tmp_path / "MANIFEST.sha256"
    manifest.write_text("0123456789abcdef  other.pt\n", encoding="utf-8")

    with pytest.raises(WeightIntegrityError, match="Manifest dosyasında bulunamadı"):
        verify_weights(weights_file, manifest)
