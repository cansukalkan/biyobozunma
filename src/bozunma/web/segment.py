"""Web için segmentasyon arka uçları: klasik, Random Forest (ML), U-Net (DL), YOLO.

Eğitilmiş ağırlıklar yalnızca models/MANIFEST.sha256 ile hash doğrulandıktan sonra yüklenir.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from bozunma.evaluate.segmenters import (
    CRACK,
    PORE,
    classical_segment,
    load_unet,
    rf_segment,
    unet_segment,
)
from bozunma.train.weights import verify_weights

CRACK_VAL = CRACK
PORE_VAL = PORE

METHOD_FILES = {
    "random_forest": "rf_pixel.joblib",
    "unet": "unet_baseline.pt",
}
METHOD_LABELS = {
    "classical-threshold": "Klasik Eşikleme (Otsu & Morfoloji)",
    "classical": "Klasik Eşikleme (Otsu & Morfoloji)",
    "random_forest": "Random Forest (ML Piksel Sınıflandırıcı)",
    "unet": "U-Net (Derin Öğrenme Segmentasyonu)",
    "yolo": "YOLO-seg (Ultralytics Poligon DL)",
}


def models_dir() -> Path:
    return Path(os.environ.get("BOZUNMA_MODELS_DIR", "models"))


def available_methods() -> list[str]:
    out = ["classical-threshold"]
    for m, fname in METHOD_FILES.items():
        if (models_dir() / fname).exists():
            out.append(m)
    out.append("yolo")
    return out


def load_evaluation() -> dict | None:
    p = models_dir() / "evaluation.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if "methods" in data and "classical" in data["methods"] and "classical-threshold" not in data["methods"]:
            data["methods"]["classical-threshold"] = data["methods"]["classical"]
        return data
    except (OSError, json.JSONDecodeError):
        return None


@lru_cache(maxsize=4)
def _load_model(method: str):  # type: ignore[no-untyped-def]
    path = verify_weights(models_dir() / METHOD_FILES[method], models_dir() / "MANIFEST.sha256")
    if method == "random_forest":
        import joblib

        return joblib.load(path)  # pickle: yalnızca hash doğrulandıktan sonra
    return load_unet(path)


def _yolo_segment(rgb: np.ndarray) -> np.ndarray:
    import cv2
    from ultralytics import YOLO

    weights_candidates = [
        os.environ.get("BOZUNMA_MODEL"),
        str(models_dir() / "best.pt"),
        str(models_dir() / "yolov8n-seg.pt"),
    ]
    weights_path = None
    for cand in weights_candidates:
        if cand and Path(cand).exists():
            weights_path = Path(cand)
            break

    if weights_path is None:
        rf = _load_model("random_forest")
        return rf_segment(rgb, rf)

    manifest = Path(os.environ.get("BOZUNMA_MANIFEST", str(models_dir() / "MANIFEST.sha256")))
    res = YOLO(str(verify_weights(weights_path, manifest))).predict(rgb, verbose=False)[0]
    mask = np.ones(rgb.shape[:2], dtype=np.uint8)
    if res.masks is not None:
        h, w = mask.shape
        for m, cls in zip(res.masks.data.cpu().numpy(), res.boxes.cls.cpu().numpy(), strict=True):
            sel = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
            mask[sel] = CRACK if int(cls) == 0 else PORE
    return mask


def segment(rgb: np.ndarray, method: str | None = None) -> tuple[np.ndarray, str]:
    """İstenen (ya da varsayılan) yöntemle segmentasyon; (maske, yöntem_id) döner."""
    avail = available_methods()
    if method is None:
        if os.environ.get("BOZUNMA_MODEL") and "yolo" in avail:
            method = "yolo"
        else:
            method = "classical-threshold"
    elif method == "classical":
        method = "classical-threshold"

    if method not in avail and method != "classical-threshold":
        raise ValueError(f"Yöntem kullanılamıyor: {method}")

    if method in ("classical", "classical-threshold"):
        return classical_segment(rgb), "classical-threshold"
    if method == "yolo":
        return _yolo_segment(rgb), method

    model = _load_model(method)
    if method == "random_forest":
        return rf_segment(rgb, model), method
    return unet_segment(rgb, model), method
