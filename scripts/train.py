"""Model eğitimi çalıştırma betiği.

YOLO segmentasyon eğitimini doğrulanmış ağırlıklar ve yapılandırma ile başlatır.
"""

import argparse
from pathlib import Path

from bozunma.train.config import load_train_config
from bozunma.train.run import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO Segmentasyon Eğitimi")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Eğitim yapılandırma dosyası (train.yaml)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Veri yapılandırma dosyası (data.yaml)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("models/yolov8n-seg.pt"),
        help="Başlangıç ağırlık dosyası",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("models/MANIFEST.sha256"),
        help="Ağırlık SHA-256 doğrulama manifesti",
    )
    args = parser.parse_args()

    cfg = load_train_config(args.config)
    print(
        f"[*] Eğitim başlatılıyor: Model={cfg.model_name}, Epochs={cfg.epochs}, "
        f"Batch={cfg.batch}, Seed={cfg.seed}"
    )

    best_model_path = run_training(
        cfg=cfg,
        weights=args.weights,
        data_yaml=args.data,
        manifest=args.manifest,
    )

    print(f"[+] Eğitim tamamlandı! En iyi ağırlıklar: {best_model_path}")


if __name__ == "__main__":
    main()
