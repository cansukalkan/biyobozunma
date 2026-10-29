"""Model karşılaştırma ve toplu metrik raporlama modülü."""

import numpy as np
import pandas as pd

from bozunma.evaluate.metrics import dice, iou


def compare_models(
    preds: dict[str, dict[str, np.ndarray]],
    truths: dict[str, np.ndarray],
    classes: list[int],
) -> pd.DataFrame:
    """Farklı segmentasyon modellerinin (YOLO, U-Net vb.) tahminlerini değerlendirir.

    Args:
        preds: Model adı -> {görsel_kimliği: tahmin_maskesi} sözlüğü.
        truths: Görsel kimliği -> gerçek_maske sözlüğü.
        classes: Değerlendirilecek sınıf kimlikleri listesi.

    Returns:
        pd.DataFrame sütunları: model, class_id, mean_iou, mean_dice.
    """
    rows = []

    for model_name, model_preds in preds.items():
        for cls_id in classes:
            ious = []
            dices = []

            for img_id, true_mask in truths.items():
                pred_mask = model_preds.get(img_id)
                if pred_mask is None:
                    continue

                ious.append(iou(pred_mask, true_mask, cls=cls_id))
                dices.append(dice(pred_mask, true_mask, cls=cls_id))

            mean_iou = float(np.mean(ious)) if ious else 0.0
            mean_dice = float(np.mean(dices)) if dices else 0.0

            rows.append(
                {
                    "model": model_name,
                    "class_id": cls_id,
                    "mean_iou": mean_iou,
                    "mean_dice": mean_dice,
                }
            )

    return pd.DataFrame(rows)
