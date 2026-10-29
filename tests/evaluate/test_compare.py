"""Model karşılaştırma (YOLO vs U-Net) testleri."""

import numpy as np
import pytest

from bozunma.evaluate.compare import compare_models


def test_compare_models_structure_and_values() -> None:
    # 2 görsel için gerçek zemin maskesi
    mask1 = np.zeros((20, 20), dtype=np.uint8)
    mask1[5:15, 5:15] = 2  # crack

    mask2 = np.zeros((20, 20), dtype=np.uint8)
    mask2[2:8, 2:8] = 2

    truths = {
        "img1": mask1,
        "img2": mask2,
    }

    # Model A: Mükemmel tahmin
    # Model B: Tamamen boş tahmin
    preds = {
        "YOLOv8": {
            "img1": mask1.copy(),
            "img2": mask2.copy(),
        },
        "UNet": {
            "img1": np.zeros((20, 20), dtype=np.uint8),
            "img2": np.zeros((20, 20), dtype=np.uint8),
        },
    }

    df = compare_models(preds=preds, truths=truths, classes=[2])

    assert len(df) == 2  # 2 model x 1 sınıf = 2 satır
    assert set(df["model"]) == {"YOLOv8", "UNet"}
    assert set(df["class_id"]) == {2}

    # YOLOv8 mükemmel olduğu için IoU ve Dice 1.0 olmalı
    yolo_row = df[df["model"] == "YOLOv8"].iloc[0]
    assert yolo_row["mean_iou"] == pytest.approx(1.0)
    assert yolo_row["mean_dice"] == pytest.approx(1.0)

    # UNet sıfır çakışma olduğu için 0.0 olmalı
    unet_row = df[df["model"] == "UNet"].iloc[0]
    assert unet_row["mean_iou"] == pytest.approx(0.0)
    assert unet_row["mean_dice"] == pytest.approx(0.0)
