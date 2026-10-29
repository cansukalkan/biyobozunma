"""YOLO segmentasyon eğitim çalıştırıcısı modülü."""

from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore

from bozunma.train.config import TrainConfig
from bozunma.train.weights import verify_weights


def run_training(
    cfg: TrainConfig,
    weights: Path,
    data_yaml: Path,
    manifest: Path,
    project: str = "runs/segment",
    name: str = "train",
) -> Path:
    """Ağırlık bütünlüğünü doğrulayarak YOLO segmentasyon eğitimini çalıştırır.

    Güvenlik kuralı: Ağırlık hash doğrulaması başarılı olmadan eğitim başlatılmaz.

    Args:
        cfg: Eğitim parametreleri (TrainConfig).
        weights: Başlangıç ağırlık dosyası yolu (.pt).
        data_yaml: Veri yapılandırma dosyası (data.yaml).
        manifest: İzin verilen ağırlık hashlerini içeren MANIFEST.sha256.
        project: Çıktıların kaydedileceği proje klasörü.
        name: Eğitim koşusunun adı.

    Returns:
        Eğitilen en iyi model ağırlık dosyasının (best.pt) yolu.

    Raises:
        WeightIntegrityError: Ağırlık özeti doğrulanamazsa.
    """
    # 1. Zorunlu ağırlık bütünlük doğrulaması
    verified_weights = verify_weights(weights, manifest)

    if YOLO is None:
        raise ImportError(
            "ultralytics kütüphanesi bulunamadı. Lütfen 'pip install ultralytics' çalıştırın."
        )

    # 2. Modeli yükle
    model = YOLO(str(verified_weights))

    # 3. Eğitimi başlat
    model.train(
        data=str(data_yaml),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        seed=cfg.seed,
        patience=cfg.patience,
        lr0=cfg.lr0,
        optimizer=cfg.optimizer,
        project=project,
        name=name,
        exist_ok=True,
        verbose=True,
    )

    # En iyi model yolunu bul
    if hasattr(model, "trainer") and model.trainer:
        save_dir = Path(model.trainer.save_dir)
    else:
        save_dir = Path(project) / name

    best_weights = save_dir / "weights" / "best.pt"

    return best_weights
