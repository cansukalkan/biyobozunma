"""Değişmez ve tahrifata duyarlı (tamper-evident) analiz kaydı modülü."""

import hashlib
import json
from pathlib import Path

import pandas as pd


def _canonical_digest(inputs_data: list[dict], results: dict) -> str:
    """Girdi ve sonuçların deterministik SHA-256 özetini üretir."""
    canonical_payload = {
        "inputs": inputs_data,
        "results": results,
    }
    encoded = json.dumps(canonical_payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_record(inputs: pd.DataFrame, results: dict, out_dir: Path) -> Path:
    """Analiz girdilerini ve sonuçlarını kriptografik özetle mühürleyerek kaydeder.

    Args:
        inputs: Analizde kullanılan veri tablosu (DataFrame).
        results: Bozunma parametreleri ve metrikler sözlüğü.
        out_dir: Kaydın yazılacağı hedef klasör.

    Returns:
        Oluşturulan analiz kaydı JSON dosyasının yolu.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    record_file = out_dir / "analysis_record.json"

    # DataFrame'i JSON serileştirilebilir biçime dönüştür
    inputs_data = inputs.to_dict(orient="records")

    # Kriptografik özet hesapla
    digest = _canonical_digest(inputs_data, results)

    full_record = {
        "checksum": digest,
        "inputs": inputs_data,
        "results": results,
    }

    with open(record_file, "w", encoding="utf-8") as f:
        json.dump(full_record, f, indent=2, default=str)

    return record_file


def verify_record(path: Path) -> bool:
    """Bir analiz kaydının bütünlüğünü ve tahrif edilip edilmediğini doğrular.

    Args:
        path: Kontrol edilecek JSON kaydının dosya yolu.

    Returns:
        Kayıt geçerli ve değiştirilmemişse True, tahrif edilmişse False.
    """
    if not path.exists():
        return False

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        stored_checksum = data.get("checksum")
        if not stored_checksum:
            return False

        inputs_data = data.get("inputs", [])
        results = data.get("results", {})

        expected_checksum = _canonical_digest(inputs_data, results)
        return str(stored_checksum).lower() == expected_checksum.lower()
    except Exception:
        return False
