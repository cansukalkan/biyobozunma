"""Web uç noktası testleri."""

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from bozunma.web.app import create_app

TOKEN = "test-token-abc"


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("BOZUNMA_RESULTS_DIR", str(tmp_path))
    monkeypatch.delenv("BOZUNMA_MODEL", raising=False)
    import bozunma.web.app as app_mod

    monkeypatch.setattr(app_mod, "RESULTS_DIR", tmp_path)
    return TestClient(create_app(token=TOKEN, production=True), raise_server_exceptions=False)


def _png() -> bytes:
    arr = np.full((64, 64, 3), 200, dtype=np.uint8)
    arr[20:24, 5:60] = 20  # koyu ince çizgi (çatlak benzeri)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_analyze_requires_token(client: TestClient) -> None:
    r = client.post("/analyze", files={"file": ("a.png", _png(), "image/png")})
    assert r.status_code == 401


def test_analyze_rejects_svg(client: TestClient) -> None:
    svg = b"<svg xmlns='http://www.w3.org/2000/svg'></svg>"
    r = client.post(
        "/analyze",
        files={"file": ("x.png", svg, "image/png")},
        headers={"X-API-Token": TOKEN},
    )
    assert r.status_code == 400


def test_analyze_ok_hides_filename(client: TestClient) -> None:
    r = client.post(
        "/analyze",
        files={"file": ("gizli_numune_adi.png", _png(), "image/png")},
        headers={"X-API-Token": TOKEN},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["id"]) == 32
    assert body["method"] == "classical-threshold"
    assert body["measurement"]["crack_pixels"] > 0
    assert body["measurement"]["is_provisional"] is True
    assert "gizli_numune_adi" not in r.text


def test_errors_do_not_leak_internals(client: TestClient) -> None:
    r = client.post("/analyze", headers={"X-API-Token": TOKEN})
    assert "Traceback" not in r.text
    assert "site-packages" not in r.text
    assert "\\" not in r.text


def test_docs_disabled_in_production(client: TestClient) -> None:
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_rate_limit(client: TestClient) -> None:
    codes = [
        client.post(
            "/analyze",
            files={"file": ("a.png", _png(), "image/png")},
            headers={"X-API-Token": TOKEN},
        ).status_code
        for _ in range(11)
    ]
    assert codes[-1] == 429


def test_kinetics_sample(client: TestClient) -> None:
    r = client.get("/kinetics/sample")
    assert r.status_code == 200
    data = r.json()
    assert "rows" in data
    assert len(data["rows"]) >= 2
    assert "fit" in data
    assert "c0" in data["fit"]
    assert "k" in data["fit"]
    assert "checksum" in data
    assert "curve" in data


def test_kinetics_fit_endpoint(client: TestClient) -> None:
    payload = {
        "sample_id": "TEST-01",
        "rows": [
            {"time_days": 0.0, "matrix_area_mm2": 100.0},
            {"time_days": 10.0, "matrix_area_mm2": 80.0},
            {"time_days": 20.0, "matrix_area_mm2": 64.0},
        ],
    }
    r = client.post("/kinetics/fit", json=payload, headers={"X-API-Token": TOKEN})
    assert r.status_code == 200
    res = r.json()
    assert res["sample_id"] == "TEST-01"
    assert res["fit"]["k"] > 0
    assert len(res["curve"]) > 10
    assert len(res["checksum"]) == 64


def test_kinetics_batch_analyze_endpoint(client: TestClient) -> None:
    files = [
        ("files", ("img0.png", _png(), "image/png")),
        ("files", ("img1.png", _png(), "image/png")),
    ]
    data = {"times": "0.0, 15.0", "sample_id": "BATCH-01", "method": "classical-threshold"}
    r = client.post("/kinetics/batch_analyze", files=files, data=data, headers={"X-API-Token": TOKEN})
    assert r.status_code == 200
    res = r.json()
    assert res["sample_id"] == "BATCH-01"
    assert len(res["rows"]) == 2
    assert "fit" in res
    assert "curve" in res
    assert len(res["gallery"]) == 2

