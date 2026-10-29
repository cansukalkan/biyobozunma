"""Eğitim yapılandırması testleri."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from bozunma.train.config import TrainConfig, load_train_config


def test_train_config_immutable() -> None:
    cfg = TrainConfig(
        model_name="yolov8n-seg.pt",
        epochs=10,
        imgsz=640,
        batch=4,
        seed=42,
    )
    # Değiştirilemez (frozen) olmalı
    with pytest.raises(FrozenInstanceError):
        cfg.epochs = 20  # type: ignore


def test_load_train_config(tmp_path: Path) -> None:
    yaml_file = tmp_path / "test_train.yaml"
    yaml_file.write_text(
        """model_name: yolov8n-seg.pt
epochs: 5
imgsz: 512
batch: 2
seed: 123
""",
        encoding="utf-8",
    )

    cfg = load_train_config(yaml_file)
    assert cfg.model_name == "yolov8n-seg.pt"
    assert cfg.epochs == 5
    assert cfg.imgsz == 512
    assert cfg.batch == 2
    assert cfg.seed == 123
