"""İçerik tabanlı görsel yükleme güvenliği ve doğrulama modülü."""

import io

from PIL import Image


class UploadRejected(Exception):
    """Güvensiz veya geçersiz yükleme tespit edildiğinde fırlatılır."""

    pass


ALLOWED_FORMATS = {"PNG", "JPEG", "TIFF", "BMP"}


def validate_upload(
    data: bytes,
    max_bytes: int = 20 * 1024 * 1024,  # 20 MB
    max_pixels: int = 50_000_000,
) -> str:
    """Yüklenen dosya baytlarını içerik düzeyinde denetler ve formatını döner.

    Güvenlik önlemleri:
      1. Boyut sınırı (Varsayılan max 20 MB)
      2. SVG, HTML ve XML içeriklerinin (XSS / XXE vektörleri) tespiti ve reddi
      3. Dekompresyon bombası (Decompression Bomb) piksel sınırı kontrolü
      4. Uzantıya değil, gerçek görsel başlıklarına (magic bytes / PIL verify) güvenme

    Args:
        data: Yüklenen dosyanın ham baytları.
        max_bytes: İzin verilen azami bayt boyutu.
        max_pixels: İzin verilen azami piksel alanı (w * h).

    Returns:
        Doğrulanan görsel biçim adı (örn. 'PNG', 'JPEG').

    Raises:
        UploadRejected: Güvensiz veya uyumsuz dosya içeriğinde.
    """
    if not data or len(data) == 0:
        raise UploadRejected("Yüklenen dosya boş.")

    if len(data) > max_bytes:
        raise UploadRejected(
            f"Dosya boyutu izin verilen sınırı aşıyor (maksimum {max_bytes // (1024 * 1024)} MB)"
        )

    # Başlangıç baytlarında SVG, XML veya HTML tespiti
    head = data[:1024].lower()
    if (
        b"<svg" in head
        or b"<?xml" in head
        or b"<html" in head
        or b"<!doctype" in head
        or b"<script" in head
    ):
        raise UploadRejected(
            "Desteklenmeyen veya geçersiz görsel biçimi (SVG ve HTML yüklemeleri reddedilir)"
        )

    try:
        buf = io.BytesIO(data)
        with Image.open(buf) as img:
            fmt = str(img.format).upper() if img.format else "UNKNOWN"
            if fmt not in ALLOWED_FORMATS:
                raise UploadRejected(
                    f"Desteklenmeyen veya geçersiz görsel biçimi: {fmt} "
                    f"(İzin verilenler: {', '.join(sorted(ALLOWED_FORMATS))})"
                )

            w, h = img.size
            if w * h > max_pixels:
                raise UploadRejected(
                    f"Görsel piksel boyutu izin verilen sınırı aşıyor ({w}x{h} > {max_pixels})"
                )

            # Görsel veri bütünlüğünü doğrula
            img.verify()

        return fmt
    except (UploadRejected, Exception) as err:
        if isinstance(err, UploadRejected):
            raise
        raise UploadRejected(
            f"Desteklenmeyen veya geçersiz görsel biçimi: {err}"
        ) from err
