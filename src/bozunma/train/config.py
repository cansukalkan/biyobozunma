"""Eğitim yapılandırma veri sınıfı ve okuyucu modülü."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrainConfig:
    """YOLO segmentasyon eğitimi için değiştirilemez yapılandırma."""

    model_name: str
    epochs: int
    imgsz: int
    batch: int
    seed: int
    patience: int = 50
    lr0: float = 0.01
    optimizer: str = "auto"


def load_train_config(path: Path) -> TrainConfig:
    """YAML dosyasından eğitim yapılandırmasını yükler.

    Args:
        path: train.yaml dosya yolu.

    Returns:
        TrainConfig örneği.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return TrainConfig(
        model_name=str(data.get("model_name", "yolov8n-seg.pt")),
        epochs=int(data.get("epochs", 100)),
        imgsz=int(data.get("imgsz", 640)),
        batch=int(data.get("batch", 8)),
        seed=int(data.get("seed", 42)),
        patience=int(data.get("patience", 50)),
        lr0=float(data.get("lr0", 0.01)),
        optimizer=str(data.get("optimizer", "auto")),
    )
