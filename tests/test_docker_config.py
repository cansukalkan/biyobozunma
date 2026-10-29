"""Docker ve güvenlik sertleştirmesi yapılandırma testleri."""

from pathlib import Path

import yaml


def test_dockerfile_security_hardening() -> None:
    df_path = Path("Dockerfile")
    assert df_path.exists(), "Dockerfile mevcut olmalı"
    content = df_path.read_text(encoding="utf-8")

    # 1. Root-olmayan kullanıcı tanımlanmış ve geçilmiş mi?
    assert "useradd" in content or "adduser" in content
    assert "USER appuser" in content

    # 2. Çok aşamalı derleme (multi-stage) var mı?
    assert "AS builder" in content
    assert "AS runtime" in content

    # 3. Sağlık kontrolü (HEALTHCHECK) tanımlanmış mı?
    assert "HEALTHCHECK" in content

    # 4. Doğrudan fabrika üzerinden uvicorn çağrılıyor mu?
    assert "bozunma.web.app:get_app" in content


def test_docker_compose_security_hardening() -> None:
    compose_path = Path("docker-compose.yml")
    assert compose_path.exists(), "docker-compose.yml mevcut olmalı"
    data = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

    services = data.get("services", {})
    assert "bozunma-web" in services
    web = services["bozunma-web"]

    # 1. Port yalnızca yerel localhost'a mı bağlanmış?
    ports = web.get("ports", [])
    assert any("127.0.0.1:8765" in p for p in ports), "Port yalnızca 127.0.0.1 üzerinde dinlemeli"

    # 2. Yetki yükseltme engeli ve capability drop var mı?
    sec_opts = web.get("security_opt", [])
    assert "no-new-privileges:true" in sec_opts

    cap_drop = web.get("cap_drop", [])
    assert "ALL" in cap_drop

    # 3. Root olmayan kullanıcı ID tanımlı mı?
    assert str(web.get("user")).startswith("10001")


def test_security_audit_document_exists() -> None:
    doc = Path("docs/security-audit.md")
    assert doc.exists(), "docs/security-audit.md mevcut olmalı"
    text = doc.read_text(encoding="utf-8")
    assert "Bandit" in text
    assert "Pip-Audit" in text
    assert "X-Content-Type-Options" in text
    assert "timing" in text.lower()
