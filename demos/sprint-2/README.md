# Sprint 2 Demosu: Temel Model Eğitimi ve Ağırlık Güvenliği

## Yapılandırma ve Parametreler
- **Model:** `yolov8n-seg.pt`
- **Görüntü Boyutu (imgsz):** 640
- **Epoch Sayısı:** 25
- **Batch Boyutu:** 4
- **Deterministik Tohum (Seed):** 42
- **Ağırlık Hash Doğrulaması:** SHA-256 MANIFEST denetimi ZORUNLU
- **İki Tekrarlı Eğitim Arasındaki Metrik Farkı (mAP50):** < 0.003 (Belgelenen eşik: < 0.01)

## Eğitim Eğrileri
![Eğitim Eğrileri](training_curves.png)

## Doğrulama Tahminleri
- ![Tahmin 1](val_prediction_01.png)
- ![Tahmin 2](val_prediction_02.png)
- ![Tahmin 3](val_prediction_03.png)
