"""Token tabanlı kimlik doğrulama testleri."""


from bozunma.web.auth import check_token


def test_check_token_valid() -> None:
    expected = "super-secret-token-12345"
    assert check_token("super-secret-token-12345", expected) is True


def test_check_token_invalid() -> None:
    expected = "super-secret-token-12345"
    assert check_token("wrong-token", expected) is False
    assert check_token("super-secret-token-1234", expected) is False


def test_check_token_none_or_empty() -> None:
    expected = "super-secret-token-12345"
    assert check_token(None, expected) is False
    assert check_token("", expected) is False


def test_check_token_empty_expected() -> None:
    # Beklenen token boşsa güvenlik gereği hiçbir token kabul edilmez
    assert check_token("some-token", "") is False
    assert check_token("", "") is False
