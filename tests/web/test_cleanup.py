"""Süresi dolan sonuç dosyalarını temizleme testleri."""

import os
from pathlib import Path

from bozunma.web.cleanup import purge_expired


def test_purge_expired_removes_old_keeps_new(tmp_path: Path) -> None:
    old_f = tmp_path / "old.png"
    new_f = tmp_path / "new.png"
    old_f.write_bytes(b"x")
    new_f.write_bytes(b"y")

    now = 1_000_000.0
    os.utime(old_f, (now - 7200, now - 7200))
    os.utime(new_f, (now - 60, now - 60))

    removed = purge_expired(tmp_path, max_age_s=3600, now=now)

    assert old_f in removed
    assert not old_f.exists()
    assert new_f.exists()


def test_purge_expired_missing_root(tmp_path: Path) -> None:
    assert purge_expired(tmp_path / "yok", max_age_s=10, now=0.0) == []
