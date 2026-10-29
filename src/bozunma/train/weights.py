"""Model ağırlık bütünlüğü ve SHA-256 doğrulama modülü."""

import hashlib
from pathlib import Path


class WeightIntegrityError(Exception):
    """Model ağırlık bütünlüğü doğrulanamadığında fırlatılır."""

    pass


def sha256_file(path: Path, chunk_size: int = 65536) -> str:
    """Bir dosyanın SHA-256 özetini parçalı okuyarak hesaplar."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_weights(path: Path, manifest: Path) -> Path:
    """Model ağırlık dosyasının özetini manifest ile karşılaştırarak doğrular.

    Args:
        path: Kontrol edilecek .pt veya model ağırlık dosyasının yolu.
        manifest: İzin verilen özetleri içeren MANIFEST.sha256 yolu.

    Returns:
        Doğrulanan ağırlık dosyasının yolu.

    Raises:
        FileNotFoundError: Dosya veya manifest mevcut değilse.
        WeightIntegrityError: Dosya manifestte yoksa veya özeti uyuşmuyorsa.
    """
    if not path.exists():
        raise FileNotFoundError(f"Ağırlık dosyası bulunamadı: {path}")

    if not manifest.exists():
        raise FileNotFoundError(f"Manifest dosyası bulunamadı: {manifest}")

    manifest_entries: dict[str, str] = {}
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                expected_sha, fname = parts[0], parts[1].lstrip("*").strip()
                manifest_entries[fname] = expected_sha
                manifest_entries[Path(fname).name] = expected_sha

    filename = path.name
    if filename not in manifest_entries:
        raise WeightIntegrityError(f"Manifest dosyasında bulunamadı: {filename}")

    actual_sha = sha256_file(path)
    expected_sha = manifest_entries[filename]

    if actual_sha.lower() != expected_sha.lower():
        raise WeightIntegrityError(
            f"Bütünlük doğrulaması başarısız: {filename} "
            f"(beklenen: {expected_sha}, gerçek: {actual_sha})"
        )

    return path
