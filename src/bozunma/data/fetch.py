"""Veri indirme ve SHA-256 bütünlük kontrolü modülü."""

import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 65536) -> str:
    """Bir dosyanın SHA-256 özetini parçalı okuma yöntemiyle hesaplar.

    Args:
        path: Dosya yolu.
        chunk_size: Okuma tampon boyutu (varsayılan: 64 KB).

    Returns:
        On altılık (hex) SHA-256 dizesi.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_checksums(root: Path, manifest: Path) -> list[str]:
    """Bir klasördeki dosyaları manifest dosyasındaki SHA-256 özetleriyle doğrular.

    Args:
        root: Dosyaların bulunduğu kök klasör.
        manifest: SHA-256 özetlerini içeren manifest dosyası.
            Satır formatı: '<sha256>  <dosya_adi>' veya '<sha256> *<dosya_adi>'

    Returns:
        Bozuk veya eksik olan dosya adlarının listesi.
    """
    invalid_files: list[str] = []
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest dosyası bulunamadı: {manifest}")

    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue

            expected_hash, filename = parts
            # Windows/Unix yol ve '*' ikili işaretini temizle
            clean_filename = filename.lstrip("*").strip().replace("\\", "/")
            file_path = root / clean_filename

            if not file_path.exists():
                invalid_files.append(clean_filename)
                continue

            actual_hash = sha256_file(file_path)
            if actual_hash.lower() != expected_hash.lower():
                invalid_files.append(clean_filename)

    return invalid_files
