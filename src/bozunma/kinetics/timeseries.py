"""Zaman serisi tablosu oluşturma ve girdi doğrulama modülü."""

import pandas as pd


def validate_row(row: dict) -> None:
    """Tekil bir zaman serisi satırının geçerliliğini doğrular.

    Kurallar:
      - Numune kimliği (sample_id) dolu ve metin olmalı.
      - Hiçbir alan veya piksel değeri negatif olamaz.
      - Tarih/zaman alanı geçerli bir ISO biçiminde ayrıştırılabilmeli.

    Raises:
        ValueError: Satır girdisi kurallara uymuyorsa.
    """
    sample_id = row.get("sample_id")
    if sample_id is None or not str(sample_id).strip():
        raise ValueError("Numune kimliği boş olamaz")

    for k, v in row.items():
        if ("area" in k or "pixel" in k or "ratio" in k) and v is not None:
            try:
                val = float(v)
                if val < 0.0:
                    raise ValueError(f"Alan değeri negatif olamaz ({k}={v})")
            except (ValueError, TypeError) as err:
                if "negatif" in str(err):
                    raise
                raise ValueError(f"Sayısal değer geçersiz: {k}={v}") from err

    timestamp = row.get("timestamp")
    if timestamp is None:
        raise ValueError("Tarih / zaman bilgisi eksik")

    try:
        pd.to_datetime(timestamp)
    except Exception as err:
        raise ValueError(f"Tarih / zaman biçimi geçersiz: {timestamp}") from err


def build_series(rows: list[dict]) -> pd.DataFrame:
    """Doğrulanmış satırlardan numune ve tarihe göre sıralı zaman serisi tablosu üretir.

    Args:
        rows: Zaman serisi ölçüm satırları listesi.

    Returns:
        Sıralanmış pd.DataFrame.
    """
    for r in rows:
        validate_row(r)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["sample_id", "timestamp"]).reset_index(drop=True)

    return df
