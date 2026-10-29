"""İçerik tabanlı yükleme güvenliği ve doğrulama testleri."""

import io

import numpy as np
import pytest
from PIL import Image

from bozunma.web.upload import UploadRejected, validate_upload


def test_validate_upload_valid_png() -> None:
    # Gerçek geçerli PNG üret
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    fmt = validate_upload(png_bytes)
    assert fmt.upper() == "PNG"


def test_validate_upload_reject_svg() -> None:
    # SVG içeriği (uzantısı png dense dahi içerikten yakalanmalı)
    svg_bytes = b"<svg xmlns='http://www.w3.org/2000/svg'><circle r='10'/></svg>"
    with pytest.raises(UploadRejected, match="Desteklenmeyen veya geçersiz görsel biçimi"):
        validate_upload(svg_bytes)


def test_validate_upload_reject_html() -> None:
    # HTML / XSS içeriği
    html_bytes = b"<!DOCTYPE html><html><body><script>alert(1)</script></body></html>"
    with pytest.raises(UploadRejected, match="Desteklenmeyen veya geçersiz görsel biçimi"):
        validate_upload(html_bytes)


def test_validate_upload_reject_decompression_bomb() -> None:
    # Dekompresyon bombası (aşırı büyük piksel sayısı)
    # PIL DecompressionBombError taklidi
    img = Image.new("RGB", (10000, 10000))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    bomb_bytes = buf.getvalue()

    with pytest.raises(UploadRejected, match="Görsel piksel boyutu izin verilen sınırı aşıyor"):
        validate_upload(bomb_bytes, max_pixels=5_000_000)
