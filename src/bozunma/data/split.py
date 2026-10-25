"""Numune bazlı (leakage-free) veri seti bölme modülü."""

import random
from collections import defaultdict
from pathlib import Path

from bozunma.data.classes import parse_sample_id


def split_by_sample(
    files: list[Path], seed: int = 42, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> dict[str, list[Path]]:
    """Görüntüleri numune kimliklerine göre train, val ve test bölmelerine ayırır.

    Aynı numuneye ait tüm görüntüler kesinlikle aynı bölmede yer alır. Bu sayede
    veri sızıntısı (data leakage) engellenir.

    Args:
        files: Bölünecek dosya yolları listesi.
        seed: Rastgelelik tohumu (tekrarlanabilirlik için).
        ratios: (train_orani, val_orani, test_orani) üçlüsü. Toplamı 1.0 olmalıdır.

    Returns:
        {'train': [Path, ...], 'val': [Path, ...], 'test': [Path, ...]} sözlüğü.

    Raises:
        ValueError: Oranlar toplamı 1.0 değilse.
    """
    if abs(sum(ratios) - 1.0) > 1e-4:
        raise ValueError(f"Oranlar toplamı 1.0 olmalıdır, alınan: {sum(ratios):.4f} ({ratios})")

    if not files:
        return {"train": [], "val": [], "test": []}

    # 1. Dosyaları numune kimliğine göre grupla
    sample_to_files: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        sample_id = parse_sample_id(f.name)
        sample_to_files[sample_id].append(f)

    # 2. Numuneleri tohumlu olarak karıştır
    unique_samples = sorted(sample_to_files.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_samples)

    n_samples = len(unique_samples)
    r_train, r_val, _ = ratios

    # En az birer numune varsa paylaştır
    n_train = int(round(n_samples * r_train))
    n_val = int(round(n_samples * r_val))

    # Sınır kontrolleri
    if n_samples >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        if n_train + n_val >= n_samples:
            n_train = n_samples - 2
            n_val = 1
    elif n_samples == 2:
        n_train = 1
        n_val = 1
    else:  # n_samples == 1
        n_train = 1
        n_val = 0

    train_samples = set(unique_samples[:n_train])
    val_samples = set(unique_samples[n_train : n_train + n_val])
    test_samples = set(unique_samples[n_train + n_val :])

    splits: dict[str, list[Path]] = {"train": [], "val": [], "test": []}

    for s_id in train_samples:
        splits["train"].extend(sample_to_files[s_id])
    for s_id in val_samples:
        splits["val"].extend(sample_to_files[s_id])
    for s_id in test_samples:
        splits["test"].extend(sample_to_files[s_id])

    # Kararlı sıralama
    for k in splits:
        splits[k].sort()

    return splits
