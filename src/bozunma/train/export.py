"""Model ONNX dışa aktarım modülü."""

from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore


def export_onnx(weights: Path, imgsz: int = 640) -> Path:
    """Eğitilmiş bir YOLO modelini ONNX formatına dönüştürür.

    Args:
        weights: Dönüştürülecek .pt model dosyasının yolu.
        imgsz: Girdi görüntü boyutu (varsayılan: 640).

    Returns:
        Üretilen .onnx dosyasının yolu.

    Raises:
        FileNotFoundError: Model ağırlık dosyası mevcut değilse.
        ImportError: ultralytics kütüphanesi yüklü değilse.
    """
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model ağırlığı bulunamadı: {weights_path}")

    if YOLO is None:
        raise ImportError(
            "ultralytics kütüphanesi bulunamadı. Lütfen 'pip install ultralytics' çalıştırın."
        )

    model = YOLO(str(weights_path))
    exported_str = model.export(format="onnx", imgsz=imgsz)

    return Path(exported_str)
