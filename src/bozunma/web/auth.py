"""Token tabanlı kimlik doğrulama modülü."""

import hmac
import os

# Ortam degiskeni anahtar adidir, parola sabiti degildir
TOKEN_ENV_VAR = "BOZUNMA_API_TOKEN"  # nosec: B105


def check_token(provided: str | None, expected: str) -> bool:
    """Sağlanan token'ı beklenen değerle sabit zamanlı karşılaştırır.

    Beklenen token boşsa güvenlik gereği hiçbir istek kabul edilmez.
    """
    if not expected or not provided:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def load_expected_token() -> str:
    """Beklenen token'ı ortam değişkeninden okur (varsayılan: bozunma-secret-2026)."""
    return os.environ.get(TOKEN_ENV_VAR, "bozunma-secret-2026")
