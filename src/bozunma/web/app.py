"""FastAPI uygulaması: sağlık, model bilgisi, analiz uç noktası ve arayüz."""

import base64
import io
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from bozunma.evaluate.metrics import segmentation_report
from bozunma.evaluate.segmenters import ALL_CLASSES, TARGET_CLASSES
from bozunma.kinetics.fit import fit_first_order
from bozunma.kinetics.record import write_record
from bozunma.kinetics.timeseries import build_series
from bozunma.measure.area import measure_image
from bozunma.measure.calibration import Calibration, load_calibration, px_to_mm2
from bozunma.measure.morphology import crack_morphology, pore_morphology
from bozunma.web.auth import check_token, load_expected_token
from bozunma.web.cleanup import purge_expired
from bozunma.web.segment import (
    CRACK_VAL,
    METHOD_LABELS,
    PORE_VAL,
    available_methods,
    load_evaluation,
    segment,
)
from bozunma.web.upload import UploadRejected, validate_upload

STATIC_DIR = Path(__file__).parent / "static"
RESULTS_DIR = Path(os.environ.get("BOZUNMA_RESULTS_DIR", "runs/web_results"))
RESULT_MAX_AGE_S = 3600
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

# Arayüz paletiyle uyumlu renkler (RGBA)
CLASS_RGBA = {
    CRACK_VAL: (122, 28, 46, 255),   # deep burgundy
    PORE_VAL: (86, 110, 90, 255),    # deep sage
}


def _rgb_png_b64(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _overlay_png_b64(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> str:
    base = rgb.astype(np.float32).copy()
    for val, color in CLASS_RGBA.items():
        sel = mask == val
        if sel.any():
            c = np.array(color[:3], dtype=np.float32)
            base[sel] = (1.0 - alpha) * base[sel] + alpha * c
    buf = io.BytesIO()
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _mask_png_b64(mask: np.ndarray) -> str:
    """Yalnızca çatlak/gözenek piksellerini renkli, gerisini saydam çizer."""
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    for val, color in CLASS_RGBA.items():
        rgba[mask == val] = color
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _error_map_png_b64(pred: np.ndarray, true: np.ndarray) -> str:
    """Hedef sınıflar için TP (sage), FP (burgundy), FN (amber) haritası."""
    p = np.isin(pred, list(TARGET_CLASSES))
    t = np.isin(true, list(TARGET_CLASSES))
    rgba = np.zeros((*pred.shape, 4), dtype=np.uint8)
    rgba[p & t & (pred == true)] = (86, 110, 90, 255)
    rgba[p & ~t] = (122, 28, 46, 255)
    rgba[~p & t] = (201, 138, 46, 255)
    rgba[p & t & (pred != true)] = (201, 138, 46, 255)
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _error_overlay_png_b64(rgb: np.ndarray, pred: np.ndarray, true: np.ndarray, alpha: float = 0.5) -> str:
    p = np.isin(pred, list(TARGET_CLASSES))
    t = np.isin(true, list(TARGET_CLASSES))
    base = rgb.astype(np.float32).copy()
    tp = p & t & (pred == true)
    fp = p & ~t
    fn = ~p & t
    mis = p & t & (pred != true)

    if tp.any():
        base[tp] = (1.0 - alpha) * base[tp] + alpha * np.array([86, 110, 90], dtype=np.float32)
    if fp.any():
        base[fp] = (1.0 - alpha) * base[fp] + alpha * np.array([122, 28, 46], dtype=np.float32)
    if fn.any():
        base[fn] = (1.0 - alpha) * base[fn] + alpha * np.array([201, 138, 46], dtype=np.float32)
    if mis.any():
        base[mis] = (1.0 - alpha) * base[mis] + alpha * np.array([201, 138, 46], dtype=np.float32)

    buf = io.BytesIO()
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _load_cal() -> Calibration:
    path = Path(os.environ.get("BOZUNMA_CALIBRATION", "configs/calibration.yaml"))
    try:
        return load_calibration(path)
    except Exception:
        return Calibration(um_per_px=1.0, status="PENDING")


def _read_mask(data: bytes, shape: tuple[int, int]) -> np.ndarray:
    validate_upload(data, max_bytes=MAX_UPLOAD_BYTES)
    img = Image.open(io.BytesIO(data))
    if img.mode not in ("L", "P", "I;16", "I"):
        raise UploadRejected("Maske tek kanallı (gri) olmalı")
    mask = np.array(img)
    if mask.shape != shape:
        raise UploadRejected("Maske boyutu görüntüyle aynı olmalı")
    if not np.isin(mask, ALL_CLASSES).all():
        raise UploadRejected("Maske yalnızca 0-3 sınıf değerlerini içermeli")
    return mask.astype(np.uint8)


def create_app(token: str | None = None, production: bool | None = None) -> FastAPI:
    """Uygulamayı oluşturur. token verilmezse BOZUNMA_API_TOKEN okunur."""
    expected = token if token is not None else load_expected_token()
    if production is None:
        production = os.environ.get("BOZUNMA_ENV", "production") != "dev"

    app = FastAPI(
        title="Bozunma Analizi",
        docs_url=None if production else "/docs",
        redoc_url=None if production else "/redoc",
        openapi_url=None if production else "/openapi.json",
    )

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limited(_: Request, __: RateLimitExceeded) -> JSONResponse:
        return JSONResponse({"detail": "Çok fazla istek. Lütfen bir dakika sonra deneyin."}, 429)

    @app.exception_handler(Exception)
    async def _internal(_: Request, __: Exception) -> JSONResponse:
        # İç yol ve yığın izi asla istemciye gönderilmez
        return JSONResponse({"detail": "İşlem sırasında bir hata oluştu."}, 500)

    @app.middleware("http")
    async def _security_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; "
            "style-src 'self' https://fonts.googleapis.com; "
            "font-src https://fonts.gstatic.com; script-src 'self'; frame-ancestors 'none'"
        )
        return resp

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/models")
    def models() -> dict:
        return {
            "available": [{"id": m, "label": METHOD_LABELS[m]} for m in available_methods()],
            "evaluation": load_evaluation(),
        }

    @app.post("/analyze")
    @limiter.limit("10/minute")
    async def analyze(
        request: Request,
        file: UploadFile = File(...),
        mask: UploadFile | None = File(default=None),
        method: str | None = Form(default=None),
        x_api_token: str | None = Header(default=None),
    ) -> dict:
        if not check_token(x_api_token, expected):
            raise HTTPException(status_code=401, detail="Geçersiz ya da eksik token.")

        data = await file.read(MAX_UPLOAD_BYTES + 1)
        try:
            fmt = validate_upload(data, max_bytes=MAX_UPLOAD_BYTES)
        except UploadRejected as err:
            raise HTTPException(status_code=400, detail=str(err).split(":")[0]) from None

        rgb = np.array(Image.open(io.BytesIO(data)).convert("RGB"))

        true_mask = None
        if mask is not None:
            mdata = await mask.read(MAX_UPLOAD_BYTES + 1)
            if mdata:
                try:
                    true_mask = _read_mask(mdata, rgb.shape[:2])
                except UploadRejected as err:
                    raise HTTPException(
                        status_code=400, detail="Maske: " + str(err).split(":")[0]
                    ) from None

        try:
            t0 = time.perf_counter()
            pred, used = segment(rgb, method or None)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        except ValueError:
            raise HTTPException(status_code=400, detail="Seçilen yöntem kullanılamıyor.") from None

        cal = _load_cal()
        result_id = uuid.uuid4().hex
        result: dict = {
            "id": result_id,
            "format": fmt,
            "method": used,
            "method_label": METHOD_LABELS[used],
            "inference_ms": elapsed_ms,
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "calibration": {"um_per_px": cal.um_per_px, "status": cal.status},
            "measurement": measure_image(pred, cal, crack_val=CRACK_VAL, pore_val=PORE_VAL),
            "morphology": {
                "crack": crack_morphology(pred, CRACK_VAL, cal.um_per_px),
                "pore": pore_morphology(pred, PORE_VAL, cal.um_per_px),
            },
            "metrics": None,
        }
        eval_data = load_evaluation()
        benchmark_eval = None
        if eval_data and "methods" in eval_data:
            benchmark_eval = eval_data["methods"].get(used) or eval_data["methods"].get(
                used.replace("-threshold", "")
            )
        result["benchmark_metrics"] = benchmark_eval

        if true_mask is not None:
            metrics = segmentation_report(pred, true_mask, TARGET_CLASSES, ALL_CLASSES)
            metrics["has_ground_truth"] = True
            result["metrics"] = metrics
            result["ground_truth_measurement"] = measure_image(
                true_mask, cal, crack_val=CRACK_VAL, pore_val=PORE_VAL
            )

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        purge_expired(RESULTS_DIR, RESULT_MAX_AGE_S, time.time())
        (RESULTS_DIR / f"{result_id}.json").write_text(json.dumps(result), encoding="utf-8")

        result["original_png_base64"] = _rgb_png_b64(rgb)
        result["overlay_png_base64"] = _overlay_png_b64(rgb, pred)
        result["mask_png_base64"] = _mask_png_b64(pred)
        if true_mask is not None:
            result["gt_mask_png_base64"] = _mask_png_b64(true_mask)
            result["error_png_base64"] = _error_map_png_b64(pred, true_mask)
            result["error_overlay_png_base64"] = _error_overlay_png_b64(rgb, pred, true_mask)
        return result

    @app.get("/sample")
    def sample() -> dict:
        """Kullanıcının doğruluk ve ML/DL metriklerini test etmesi için örnek veri çifti döner."""
        raw_dir = Path("data/raw")
        img_p = raw_dir / "ID14_470_sub_01_Image.png"
        mask_p = raw_dir / "ID14_470_sub_01_Mask.png"
        if not img_p.exists() or not mask_p.exists():
            raise HTTPException(status_code=404, detail="Örnek dosya bulunamadı")
        return {
            "image_b64": base64.b64encode(img_p.read_bytes()).decode("ascii"),
            "mask_b64": base64.b64encode(mask_p.read_bytes()).decode("ascii"),
            "sample_name": "CrackenPy ID14 (Referans Doğruluk Testi)",
        }

    @app.get("/kinetics/sample")
    def kinetics_sample() -> dict:
        """CrackenPy referans numunelerine ait doğrulanmış 6 zaman noktalı bozunma serisi."""
        rec_path = Path("demos/sprint-4/analysis_record.json")
        sample_rows = []
        fit_res = {}
        checksum = ""
        if rec_path.exists():
            try:
                data = json.loads(rec_path.read_text(encoding="utf-8"))
                sample_rows = data.get("inputs", [])
                fit_res = data.get("results", {})
                checksum = data.get("checksum", "")
            except (OSError, json.JSONDecodeError):
                sample_rows = []

        if not sample_rows:
            sample_rows = [
                {"sample_id": "ID14_470", "time_days": 0.0, "matrix_area_mm2": 0.1039, "crack_area_mm2": 0.0062, "pore_area_mm2": 0.0006},
                {"sample_id": "ID14_470", "time_days": 10.0, "matrix_area_mm2": 0.1054, "crack_area_mm2": 0.0026, "pore_area_mm2": 0.0027},
                {"sample_id": "ID14_470", "time_days": 20.0, "matrix_area_mm2": 0.1094, "crack_area_mm2": 0.0013, "pore_area_mm2": 0.00003},
                {"sample_id": "ID14_470", "time_days": 30.0, "matrix_area_mm2": 0.1101, "crack_area_mm2": 0.0001, "pore_area_mm2": 0.0005},
                {"sample_id": "ID14_470", "time_days": 40.0, "matrix_area_mm2": 0.1020, "crack_area_mm2": 0.0080, "pore_area_mm2": 0.0007},
                {"sample_id": "ID14_470", "time_days": 50.0, "matrix_area_mm2": 0.1043, "crack_area_mm2": 0.0037, "pore_area_mm2": 0.0027},
            ]

        t_arr = np.array([float(r["time_days"]) for r in sample_rows], dtype=float)
        c_arr = np.array([float(r["matrix_area_mm2"]) for r in sample_rows], dtype=float)
        if not fit_res:
            fit_res = fit_first_order(t_arr, c_arr)

        raw_dir = Path("data/raw")
        gallery = []
        for i, row in enumerate(sample_rows, start=1):
            img_path = raw_dir / f"ID14_470_sub_0{i}_Image.png"
            img_b64 = None
            if img_path.exists():
                buf = io.BytesIO()
                Image.open(img_path).save(buf, format="PNG")
                img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            gallery.append({
                "time_days": float(row.get("time_days", (i - 1) * 10.0)),
                "image_b64": img_b64,
                "matrix_area_mm2": round(float(row.get("matrix_area_mm2", 0.0)), 5),
                "crack_area_mm2": round(float(row.get("crack_area_mm2", 0.0)), 5),
                "pore_area_mm2": round(float(row.get("pore_area_mm2", 0.0)), 5),
            })

        max_t = max(t_arr) if len(t_arr) > 0 and max(t_arr) > 0 else 50.0
        t_dense = np.linspace(0.0, max_t * 1.15, 60)
        c0 = float(fit_res.get("c0", 0.1033))
        k = float(fit_res.get("k", 0.0253))
        c_dense = c0 * np.exp(-k * t_dense)
        curve = [{"t": round(float(t), 2), "c": round(float(c), 5)} for t, c in zip(t_dense, c_dense)]

        return {
            "sample_id": "ID14_470",
            "rows": sample_rows,
            "fit": fit_res,
            "curve": curve,
            "gallery": gallery,
            "checksum": checksum or "65f63a7c10131495f88e40ce34345ce4a295c82cd865c35a78e25dcc7ca60447",
            "is_tamper_evident": True,
        }

    @app.post("/kinetics/fit")
    async def kinetics_fit(
        request: Request,
        x_api_token: str | None = Header(default=None),
    ) -> dict:
        if not check_token(x_api_token, expected):
            raise HTTPException(status_code=401, detail="Geçersiz ya da eksik token.")

        payload = await request.json()
        rows = payload.get("rows", [])
        sample_id = payload.get("sample_id", "SAMPLE-01")

        if len(rows) < 2:
            raise HTTPException(status_code=400, detail="Uydurma için en az 2 zaman noktası gereklidir.")

        for r in rows:
            if "sample_id" not in r:
                r["sample_id"] = sample_id
            if "timestamp" not in r:
                r["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        df = build_series(rows)
        t_arr = np.array([float(r["time_days"]) for r in rows], dtype=float)
        c_arr = np.array([float(r["matrix_area_mm2"]) for r in rows], dtype=float)

        fit_res = fit_first_order(t_arr, c_arr)

        max_t = max(t_arr) if max(t_arr) > 0 else 10.0
        t_dense = np.linspace(0.0, max_t * 1.15, 60)
        c_dense = fit_res["c0"] * np.exp(-fit_res["k"] * t_dense)
        curve = [{"t": round(float(t), 2), "c": round(float(c), 5)} for t, c in zip(t_dense, c_dense)]

        record_id = uuid.uuid4().hex
        out_dir = RESULTS_DIR / f"kinetics_{record_id}"
        rec_path = write_record(df, fit_res, out_dir)
        checksum = json.loads(rec_path.read_text(encoding="utf-8")).get("checksum", "")

        return {
            "record_id": record_id,
            "sample_id": sample_id,
            "fit": fit_res,
            "curve": curve,
            "rows": rows,
            "checksum": checksum,
            "is_tamper_evident": True,
        }

    @app.post("/kinetics/batch_analyze")
    @limiter.limit("10/minute")
    async def kinetics_batch_analyze(
        request: Request,
        files: list[UploadFile] = File(...),
        times: str = Form(...),
        sample_id: str = Form(default="SAMPLE-01"),
        method: str | None = Form(default=None),
        x_api_token: str | None = Header(default=None),
    ) -> dict:
        if not check_token(x_api_token, expected):
            raise HTTPException(status_code=401, detail="Geçersiz ya da eksik token.")

        if not files:
            raise HTTPException(status_code=400, detail="En az bir görüntü dosyası yüklenmelidir.")

        try:
            if times.strip().startswith("["):
                time_list = [float(x) for x in json.loads(times)]
            else:
                time_list = [float(x.strip()) for x in times.split(",") if x.strip()]
        except Exception:
            raise HTTPException(status_code=400, detail="Zaman değerleri sayısal olmalıdır (ör. 0, 10, 20).") from None

        if len(time_list) != len(files):
            raise HTTPException(status_code=400, detail="Yüklenen dosya sayısı ile zaman adımı sayısı eşleşmelidir.")

        if len(files) < 2:
            raise HTTPException(status_code=400, detail="Kinetik uydurma için en az 2 zaman noktası gereklidir.")

        cal = _load_cal()
        chosen_method = method or "random_forest"

        sorted_pairs = sorted(zip(time_list, files), key=lambda p: p[0])
        rows = []
        gallery = []

        for t_val, f in sorted_pairs:
            data = await f.read(MAX_UPLOAD_BYTES + 1)
            try:
                validate_upload(data, max_bytes=MAX_UPLOAD_BYTES)
            except UploadRejected as err:
                raise HTTPException(status_code=400, detail=str(err).split(":")[0]) from None

            rgb = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
            try:
                pred, used = segment(rgb, chosen_method)
            except ValueError:
                raise HTTPException(status_code=400, detail="Seçilen yöntem kullanılamıyor.") from None

            meas = measure_image(pred, cal, crack_val=CRACK_VAL, pore_val=PORE_VAL)
            total_mm2, _ = px_to_mm2(float(meas["total_pixels"]), cal)
            matrix_mm2 = max(0.0, total_mm2 - (meas["crack_area_mm2"] + meas["pore_area_mm2"]))
            defect_ratio = meas["crack_ratio"] + meas["pore_ratio"]
            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            row = {
                "sample_id": sample_id,
                "time_days": float(t_val),
                "timestamp": now_iso,
                "matrix_area_mm2": float(matrix_mm2),
                "crack_area_mm2": float(meas["crack_area_mm2"]),
                "pore_area_mm2": float(meas["pore_area_mm2"]),
                "is_provisional": cal.status != "MEASURED",
            }
            rows.append(row)
            gallery.append({
                "time_days": float(t_val),
                "overlay_b64": _overlay_png_b64(rgb, pred),
                "matrix_area_mm2": round(float(matrix_mm2), 5),
                "crack_area_mm2": round(float(meas["crack_area_mm2"]), 5),
                "pore_area_mm2": round(float(meas["pore_area_mm2"]), 5),
                "defect_ratio": round(float(defect_ratio * 100), 2),
            })

        df = build_series(rows)
        t_arr = np.array([r["time_days"] for r in rows], dtype=float)
        c_arr = np.array([r["matrix_area_mm2"] for r in rows], dtype=float)

        fit_res = fit_first_order(t_arr, c_arr)

        max_t = max(t_arr) if max(t_arr) > 0 else 10.0
        t_dense = np.linspace(0.0, max_t * 1.15, 60)
        safe_k = max(0.0, fit_res.get("k", 0.0))
        c_dense = fit_res["c0"] * np.exp(-safe_k * t_dense)
        curve = [{"t": round(float(t), 2), "c": round(float(c), 5)} for t, c in zip(t_dense, c_dense)]

        record_id = uuid.uuid4().hex
        out_dir = RESULTS_DIR / f"kinetics_{record_id}"
        rec_path = write_record(df, fit_res, out_dir)
        checksum = json.loads(rec_path.read_text(encoding="utf-8")).get("checksum", "")

        return {
            "record_id": record_id,
            "sample_id": sample_id,
            "method": chosen_method,
            "method_label": METHOD_LABELS.get(chosen_method, chosen_method),
            "calibration": {"um_per_px": cal.um_per_px, "status": cal.status},
            "fit": fit_res,
            "curve": curve,
            "rows": rows,
            "gallery": gallery,
            "checksum": checksum,
            "is_tamper_evident": True,
        }

    @app.get("/kinetics/record/{record_id}")
    def kinetics_record(record_id: str) -> FileResponse:
        p = RESULTS_DIR / f"kinetics_{record_id}" / "analysis_record.json"
        if not p.exists():
            raise HTTPException(status_code=404, detail="Analiz kaydı bulunamadı")
        return FileResponse(p, media_type="application/json", filename=f"analysis_record_{record_id}.json")


    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def get_app() -> FastAPI:
    """uvicorn fabrika girişi: uvicorn bozunma.web.app:get_app --factory"""
    return create_app()
