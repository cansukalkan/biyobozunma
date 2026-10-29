"""Süresi dolan analiz sonuçlarını silen temizleyici."""

from pathlib import Path


def purge_expired(root: Path, max_age_s: int, now: float) -> list[Path]:
    """root altında değiştirilme zamanı max_age_s'den eski dosyaları siler.

    Yalnızca root içindeki normal dosyalara dokunur; sembolik bağlantıları izlemez.
    """
    if not root.exists():
        return []

    removed: list[Path] = []
    for p in root.rglob("*"):
        if p.is_symlink() or not p.is_file():
            continue
        if now - p.stat().st_mtime > max_age_s:
            p.unlink(missing_ok=True)
            removed.append(p)
    return removed
