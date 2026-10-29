"""Yol güvenliği ve dizin aşımı (path traversal) koruma modülü."""

from pathlib import Path


def validate_safe_path(target: Path | str, base_dir: Path | str = "data") -> Path:
    """Verilen dosya yolunun belirtilen temel dizin içinde güvenli olduğunu doğrular.

    Path traversal (../) ve yetkisiz mutlak yol erişimlerini engeller.

    Args:
        target: Kontrol edilecek göreli dosya yolu.
        base_dir: İzin verilen temel dizin (varsayılan: data).

    Returns:
        Güvenli mutlak Path nesnesi.

    Raises:
        ValueError: Dosya yolu güvensizse, mutlaksa veya base_dir dışına çıkıyorsa.
    """
    target_str = str(target)
    base_path = Path(base_dir).resolve()

    # Mutlak yol kontrolü
    if Path(target_str).is_absolute() or target_str.startswith(("/", "\\")):
        raise ValueError(f"Mutlak yollar kabul edilmez: {target_str}")

    # ../ veya ..\\ kontrolü
    if ".." in target_str.replace("\\", "/").split("/"):
        raise ValueError(f"Geçersiz veya güvensiz dosya yolu: {target_str}")

    resolved = (base_path / target_str).resolve()

    # Dizin dışına taşma kontrolü
    try:
        resolved.relative_to(base_path)
    except ValueError:
        raise ValueError(f"Geçersiz veya güvensiz dosya yolu: {target_str}") from None

    return resolved
