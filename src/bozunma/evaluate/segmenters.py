"""Segmentasyon yöntemleri: klasik eşikleme, Random Forest (ML) ve U-Net (DL).

Tüm yöntemler RGB görüntü alır ve 0=arka plan, 1=matris, 2=çatlak, 3=gözenek
değerlerinden oluşan maske döndürür.
"""

from pathlib import Path

import cv2
import numpy as np

BG, MATRIX, CRACK, PORE = 0, 1, 2, 3
ALL_CLASSES = [BG, MATRIX, CRACK, PORE]
TARGET_CLASSES = {CRACK: "crack", PORE: "pore"}


# ---------------------------------------------------------------- klasik
def classical_segment(rgb: np.ndarray, min_area_px: int = 10) -> np.ndarray:
    """Otsu eşikleme + yuvarlaklık ölçütü. Model değildir; temel karşılaştırma içindir."""
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = np.full(gray.shape, MATRIX, dtype=np.uint8)
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_px:
            continue
        perim = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (perim * perim) if perim > 0 else 0.0
        cv2.drawContours(mask, [c], -1, PORE if circ > 0.6 else CRACK, thickness=cv2.FILLED)
    return mask


# ---------------------------------------------------------------- Random Forest
def pixel_features(rgb: np.ndarray) -> np.ndarray:
    """Piksel başına çok ölçekli doku öznitelikleri (H*W, F)."""
    img = rgb.astype(np.float32) / 255.0
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    feats = [img[..., 0], img[..., 1], img[..., 2], gray]
    for s in (1.0, 2.0, 4.0, 8.0):
        g = cv2.GaussianBlur(gray, (0, 0), s)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        lap = cv2.Laplacian(g, cv2.CV_32F)
        mean = g
        sq = cv2.GaussianBlur(gray * gray, (0, 0), s)
        std = np.sqrt(np.maximum(sq - mean * mean, 0))
        feats += [g, np.hypot(gx, gy), lap, std, gray - g]
    return np.stack(feats, axis=-1).reshape(-1, len(feats))


def sample_training_pixels(
    rgb: np.ndarray, mask: np.ndarray, per_class: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sınıf dengesizliğini azaltmak için sınıf başına en fazla per_class piksel örnekler."""
    X = pixel_features(rgb)
    y = mask.ravel()
    idx = []
    for c in ALL_CLASSES:
        pos = np.flatnonzero(y == c)
        if pos.size:
            idx.append(rng.choice(pos, size=min(per_class, pos.size), replace=False))
    sel = np.concatenate(idx)
    return X[sel], y[sel]


def train_random_forest(pairs: list[tuple[np.ndarray, np.ndarray]], seed: int = 42):
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(seed)
    Xs, ys = zip(*(sample_training_pixels(r, m, 4000, rng) for r, m in pairs), strict=True)
    clf = RandomForestClassifier(
        n_estimators=80, max_depth=18, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=seed,
    )
    clf.fit(np.concatenate(Xs), np.concatenate(ys))
    return clf


def rf_segment(rgb: np.ndarray, clf) -> np.ndarray:  # type: ignore[no-untyped-def]
    pred = clf.predict(pixel_features(rgb))
    return pred.reshape(rgb.shape[:2]).astype(np.uint8)


# ---------------------------------------------------------------- U-Net
def _to_tensor(rgb: np.ndarray):  # type: ignore[no-untyped-def]
    import torch

    x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    return (x - 0.5) / 0.25


def train_unet(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    iters: int = 600,
    crop: int = 128,
    batch: int = 8,
    seed: int = 42,
):  # type: ignore[no-untyped-def]
    """bozunma.evaluate.baseline_unet.UNetBaseline'ı rastgele kırpmalarla eğitir (CPU)."""
    import torch
    import torch.nn as nn

    from bozunma.evaluate.baseline_unet import UNetBaseline

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = UNetBaseline(in_channels=3, num_classes=len(ALL_CLASSES))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    counts = np.bincount(np.concatenate([m.ravel() for _, m in pairs]), minlength=4) + 1
    w = torch.tensor((counts.sum() / counts) ** 0.5, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=w / w.mean())

    model.train()
    for _ in range(iters):
        xs, ys = [], []
        for _ in range(batch):
            rgb, m = pairs[rng.integers(len(pairs))]
            h, wd = m.shape
            y0 = rng.integers(0, h - crop + 1)
            x0 = rng.integers(0, wd - crop + 1)
            r = rgb[y0:y0 + crop, x0:x0 + crop]
            mm = m[y0:y0 + crop, x0:x0 + crop]
            k = int(rng.integers(4))
            r, mm = np.rot90(r, k), np.rot90(mm, k)
            if rng.random() < 0.5:
                r, mm = r[:, ::-1], mm[:, ::-1]
            xs.append(_to_tensor(np.ascontiguousarray(r)))
            ys.append(torch.from_numpy(np.ascontiguousarray(mm).astype(np.int64)))
        opt.zero_grad()
        loss = loss_fn(model(torch.stack(xs)), torch.stack(ys))
        loss.backward()
        opt.step()
    model.eval()
    return model


def unet_segment(rgb: np.ndarray, model) -> np.ndarray:  # type: ignore[no-untyped-def]
    import torch

    h, w = rgb.shape[:2]
    ph, pw = (-h) % 4, (-w) % 4
    padded = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    with torch.no_grad():
        out = model(_to_tensor(padded).unsqueeze(0))[0].argmax(0).numpy()
    return out[:h, :w].astype(np.uint8)


def load_unet(path: Path):  # type: ignore[no-untyped-def]
    import torch

    from bozunma.evaluate.baseline_unet import UNetBaseline

    model = UNetBaseline(in_channels=3, num_classes=len(ALL_CLASSES))
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model
