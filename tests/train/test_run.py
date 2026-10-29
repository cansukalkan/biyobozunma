"""Eğitim çalıştırıcısı ve ağırlık bütünlüğü testleri."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bozunma.train.config import TrainConfig
from bozunma.train.run import run_training
from bozunma.train.weights import WeightIntegrityError


def test_run_training_weight_integrity_fails(tmp_path: Path) -> None:
    cfg = TrainConfig(
        model_name="test.pt",
        epochs=1,
        imgsz=64,
        batch=2,
        seed=42,
    )
    weights = tmp_path / "test.pt"
    weights.write_bytes(b"bad-data")

    manifest = tmp_path / "MANIFEST.sha256"
    manifest.write_text("invalid_sha test.pt\n", encoding="utf-8")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.touch()

    # Bozuk ağırlıkla eğitim başlamamalı, WeightIntegrityError fırlatılmalı
    with pytest.raises(WeightIntegrityError):
        run_training(cfg=cfg, weights=weights, data_yaml=data_yaml, manifest=manifest)


@patch("bozunma.train.run.YOLO")
def test_run_training_valid(mock_yolo: MagicMock, tmp_path: Path) -> None:
    import hashlib

    cfg = TrainConfig(
        model_name="valid.pt",
        epochs=1,
        imgsz=64,
        batch=2,
        seed=42,
    )
    content = b"valid-weights-data"
    weights = tmp_path / "valid.pt"
    weights.write_bytes(content)

    sha = hashlib.sha256(content).hexdigest()
    manifest = tmp_path / "MANIFEST.sha256"
    manifest.write_text(f"{sha} valid.pt\n", encoding="utf-8")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.touch()

    # Mock YOLO train çıktısı
    fake_model = MagicMock()
    mock_yolo.return_value = fake_model
    fake_save_dir = tmp_path / "runs" / "train"
    (fake_save_dir / "weights").mkdir(parents=True)
    best_pt = fake_save_dir / "weights" / "best.pt"
    best_pt.write_bytes(b"best-weights")
    fake_model.trainer.save_dir = fake_save_dir

    result_path = run_training(cfg=cfg, weights=weights, data_yaml=data_yaml, manifest=manifest)
    assert result_path.exists()
    assert mock_yolo.called
