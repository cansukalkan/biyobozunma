"""Gerçek veri üzerinde yöntem değerlendirmesi (leave-one-image-out) ve son model eğitimi.

Çıktılar:
  models/rf_pixel.joblib, models/unet_baseline.pt   (git'e girmez)
  models/MANIFEST.sha256                             (özetler güncellenir)
  models/evaluation.json, demos/sprint-5/evaluation.json

Kullanım: python scripts/evaluate_models.py [--unet-iters 600]
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bozunma.data.classes import parse_sample_id  # noqa: E402
from bozunma.data.writer import find_matching_mask  # noqa: E402
from bozunma.evaluate.metrics import segmentation_report  # noqa: E402
from bozunma.evaluate.segmenters import (  # noqa: E402
    ALL_CLASSES,
    TARGET_CLASSES,
    classical_segment,
    rf_segment,
    train_random_forest,
    train_unet,
    unet_segment,
)
from bozunma.train.weights import sha256_file  # noqa: E402


def load_pairs(raw: Path) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    out = []
    for p in sorted(raw.iterdir()):
        if "mask" in p.name.lower() or p.suffix.lower() not in {".png", ".tif", ".tiff", ".jpg"}:
            continue
        m = find_matching_mask(p)
        if m is None:
            continue
        rgb = np.array(Image.open(p).convert("RGB"))
        mask = np.array(Image.open(m))
        if mask.shape != rgb.shape[:2]:
            continue
        try:
            sid = parse_sample_id(p.name)
        except ValueError:
            sid = "bilinmiyor"
        out.append((p.stem, sid, rgb, mask))
    return out


def pooled_report(cm_total: np.ndarray) -> dict:
    """Katlar boyunca toplanmış karışıklık matrisinden metrikler."""
    labels = ALL_CLASSES
    per_class = {}
    for c, name in TARGET_CLASSES.items():
        i = labels.index(c)
        tp = cm_total[i, i]
        fp = cm_total[:, i].sum() - tp
        fn = cm_total[i, :].sum() - tp
        per_class[name] = {
            "iou": float(tp / (tp + fp + fn)) if tp + fp + fn else 1.0,
            "dice": float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else 1.0,
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "support_px": int(cm_total[i, :].sum()),
        }
    return {
        "per_class": per_class,
        "pixel_accuracy": float(np.trace(cm_total) / cm_total.sum()),
        "confusion_matrix": cm_total.astype(int).tolist(),
        "labels": labels,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--unet-iters", type=int, default=600)
    args = ap.parse_args()

    pairs = load_pairs(args.raw)
    n = len(pairs)
    print(f"[*] {n} görüntü-maske çifti; numuneler: {sorted({s for _, s, _, _ in pairs})}")
    if n < 3:
        raise SystemExit("Değerlendirme için en az 3 çift gerekir.")

    methods = ["classical", "random_forest", "unet"]
    cms = {m: np.zeros((4, 4), dtype=np.int64) for m in methods}
    per_image = {m: [] for m in methods}
    timings = {m: 0.0 for m in methods}

    for k, (name, _, rgb, mask) in enumerate(pairs):
        train = [(r, m) for j, (_, _, r, m) in enumerate(pairs) if j != k]
        print(f"[fold {k + 1}/{n}] test={name}")

        t0 = time.perf_counter()
        preds = {"classical": classical_segment(rgb)}
        timings["classical"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        preds["random_forest"] = rf_segment(rgb, train_random_forest(train))
        timings["random_forest"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        preds["unet"] = unet_segment(rgb, train_unet(train, iters=args.unet_iters))
        timings["unet"] += time.perf_counter() - t0

        for m in methods:
            rep = segmentation_report(preds[m], mask, TARGET_CLASSES, ALL_CLASSES)
            cms[m] += np.array(rep["confusion_matrix"])
            per_image[m].append({"image_index": k, **{
                f"{c}_iou": rep["per_class"][c]["iou"] for c in TARGET_CLASSES.values()
            }, "pixel_accuracy": rep["pixel_accuracy"]})
            print(f"    {m:14s} crack IoU={rep['per_class']['crack']['iou']:.3f} "
                  f"pore IoU={rep['per_class']['pore']['iou']:.3f} "
                  f"acc={rep['pixel_accuracy']:.3f}")

    results = {}
    for m in methods:
        rep = pooled_report(cms[m])
        ious = np.array([[r["crack_iou"], r["pore_iou"]] for r in per_image[m]])
        rep["mean_iou_per_image"] = {"crack": float(ious[:, 0].mean()),
                                     "pore": float(ious[:, 1].mean())}
        rep["std_iou_per_image"] = {"crack": float(ious[:, 0].std()),
                                    "pore": float(ious[:, 1].std())}
        rep["seconds_total"] = round(timings[m], 1)
        results[m] = rep

    # Son modeller: tüm veriyle
    print("[*] Son modeller tüm veriyle eğitiliyor...")
    all_pairs = [(r, m) for _, _, r, m in pairs]
    args.models.mkdir(exist_ok=True)
    import joblib
    import torch

    rf_path = args.models / "rf_pixel.joblib"
    joblib.dump(train_random_forest(all_pairs), rf_path, compress=3)
    unet_path = args.models / "unet_baseline.pt"
    torch.save(train_unet(all_pairs, iters=args.unet_iters).state_dict(), unet_path)

    manifest = args.models / "MANIFEST.sha256"
    keep = [ln for ln in manifest.read_text(encoding="utf-8").splitlines()
            if ln.startswith("#") or (ln.strip() and not ln.strip().endswith(
                (rf_path.name, unet_path.name)))] if manifest.exists() else []
    keep += [f"{sha256_file(rf_path)}  {rf_path.name}", f"{sha256_file(unet_path)}  {unet_path.name}"]
    manifest.write_text("\n".join(keep) + "\n", encoding="utf-8")

    samples = sorted({s for _, s, _, _ in pairs})
    evaluation = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": "leave-one-image-out",
        "n_images": n,
        "n_samples": len(samples),
        "caveat": (
            "Görüntülerin çoğu aynı numuneden alınmış alt kesitlerdir; bu nedenle sonuçlar "
            "yeni numunelere genelleme performansını olduğundan iyi gösterebilir. "
            "Küçük veri: metrikler ön değerlendirme niteliğindedir."
        ),
        "unet_iters": args.unet_iters,
        "methods": results,
    }
    out = args.models / "evaluation.json"
    out.write_text(json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8")
    demo = Path("demos/sprint-5")
    demo.mkdir(parents=True, exist_ok=True)
    shutil.copy(out, demo / "evaluation.json")

    print("\n=== Havuzlanmış sonuçlar ===")
    for m, r in results.items():
        pc = r["per_class"]
        print(f"{m:14s} crack IoU={pc['crack']['iou']:.3f} Dice={pc['crack']['dice']:.3f} | "
              f"pore IoU={pc['pore']['iou']:.3f} | acc={r['pixel_accuracy']:.3f} | "
              f"{r['seconds_total']}s")


if __name__ == "__main__":
    main()
