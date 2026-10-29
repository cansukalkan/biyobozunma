"""Çatlak ve gözenek morfolojisi: sayı, iskelet uzunluğu, ortalama genişlik, çap."""

import cv2
import numpy as np
from skimage.morphology import skeletonize


def crack_morphology(mask: np.ndarray, crack_val: int = 2, um_per_px: float = 1.0) -> dict:
    """Çatlak bileşen sayısı, toplam iskelet uzunluğu ve ortalama genişlik.

    Ortalama genişlik = çatlak alanı / iskelet uzunluğu (piksel).
    """
    binary = (mask == crack_val).astype(np.uint8)
    area = int(binary.sum())
    if area == 0:
        return {"count": 0, "length_px": 0, "mean_width_px": 0.0,
                "length_um": 0.0, "mean_width_um": 0.0, "longest_px": 0}

    n, labels = cv2.connectedComponents(binary, connectivity=8)
    skel = skeletonize(binary.astype(bool))
    length = int(skel.sum())
    longest = int(np.bincount(labels[skel], minlength=n)[1:].max()) if length else 0
    width = area / length if length else 0.0
    return {
        "count": int(n - 1),
        "length_px": length,
        "mean_width_px": round(float(width), 3),
        "length_um": round(length * um_per_px, 2),
        "skeleton_length_um": round(length * um_per_px, 2),
        "mean_width_um": round(float(width) * um_per_px, 3),
        "longest_px": longest,
    }


def pore_morphology(mask: np.ndarray, pore_val: int = 3, um_per_px: float = 1.0) -> dict:
    """Gözenek sayısı, eşdeğer çap ortalaması/maksimumu ve ortalama yuvarlaklık."""
    binary = (mask == pore_val).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    diams, circs = [], []
    for c in contours:
        a = cv2.contourArea(c)
        if a <= 0:
            a = 1.0  # tek piksel bileşen
        p = cv2.arcLength(c, True)
        diams.append(2.0 * np.sqrt(a / np.pi))
        circs.append(min(1.0, 4 * np.pi * a / (p * p)) if p > 0 else 1.0)
    if not diams:
        return {
            "count": 0, "mean_diameter_px": 0.0, "max_diameter_px": 0.0,
            "mean_diameter_um": 0.0, "mean_equivalent_diameter_um": 0.0,
            "mean_circularity": 0.0, "circularity": 0.0,
        }
    md = float(np.mean(diams))
    circ_val = round(float(np.mean(circs)), 3)
    diam_um = round(md * um_per_px, 3)
    return {
        "count": len(diams),
        "mean_diameter_px": round(md, 3),
        "max_diameter_px": round(float(np.max(diams)), 3),
        "mean_diameter_um": diam_um,
        "mean_equivalent_diameter_um": diam_um,
        "mean_circularity": circ_val,
        "circularity": circ_val,
    }
