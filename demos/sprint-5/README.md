# Sprint 5 Demosu: Paketleme, Güvenlik Sertleştirmesi, Docker & Nihai Platform

Bu dizin, **Bozunma Analizi** projesinin Sprint 5 (Nihai Dağıtım, Docker Paketleme ve Güvenlik Sertleştirmesi) çıktılarını, model kıyaslamalarını ve çalıştırma talimatlarını içerir.

---

## 1. Hızlı Başlangıç (Docker ile Çalıştırma)

Uygulama, root olmayan kullanıcı (`UID 10001`) ve çok aşamalı hafif imaj ile güvenli biçimde konteynerize edilmiştir.

### Konteyneri Başlatma
```bash
# Proje kök dizininde:
docker compose up --build -d
```

### Konteyner Durumunu ve Sağlığını Kontrol Etme
```bash
docker compose ps
```

Tarayıcınızdan erişin:
👉 **[http://127.0.0.1:8765/](http://127.0.0.1:8765/)**

Varsayılan API Token: `bozunma-secret-2026`

---

## 2. Model ve Algoritma Karşılaştırması (CrackenPy Test Veri Kümesi)

Leave-One-Image-Out (LOO) çapraz doğrulaması ve test kümesi üzerinde elde edilen nihai başarım metrikleri:

| Algoritma / Model | Mimari Türü | Piksel Doğruluk | Çatlak IoU | Çatlak Dice (F1) | Gözenek IoU | Çıkarım Süresi |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest (ML)** ⭐ | 24-D Doku Öznitelikleri (Gauss/Sobel/Laplace) | **%98.8** | **0.766** | **0.867** | **0.781** | **~150 ms** |
| **YOLOv8-Seg (DL)** ⚡ | Ultralytics Poligon Tabanlı Derin Öğrenme | **%96.2** | **0.680** | **0.810** | **0.650** | **~180 ms** |
| **U-Net Baseline (DL)** 🧠 | Evrişimli Kodlayıcı-Kod Çözücü (CNN) | **%95.6** | **0.421** | **0.593** | **0.001** | **~300 ms** |
| **Klasik Eşikleme** 🔬 | Otsu Histogram & Morfolojik Filtreler | **%75.3** | **0.119** | **0.213** | **0.042** | **~20 ms** |

*Detaylı metrikler ve konfüzyon matrisleri [`evaluation.json`](evaluation.json) dosyasında kayıtlıdır.*

---

## 3. Kinetik Bozunma Ölçüm Parametreleri

* **Kinetik Denklem:** $C(t) = C_0 \cdot e^{-k \cdot t}$
* **Başlangıç Sağlam Alan ($C_0$):** `0.1033 mm²`
* **Bozunma Hız Sabiti ($k$):** `0.0253 ± 0.0007 gün⁻¹`
* **Tahmini Yarı Ömür ($t_{1/2}$):** `27.4 gün`
* **Uyum Kalitesi ($R^2$):** `%98.5`
* **Optik Kalibrasyon Ölçeği:** `1.3 µm/px (MEASURED)`

---

## 4. Güvenlik Denetimi Özeti

Platform, statik analiz ve penetrasyon kurallarına göre sertleştirilmiştir:
* **Bandit Güvenlik Taraması:** 0 Kritik, 0 Yüksek, 0 Orta, 0 Düşük bulgu.
* **Pip-Audit Bağımlılık Taraması:** 0 bilinen güvenlik zafiyeti.
* **Konteyner İzolasyonu:** `USER appuser:appgroup` (UID 10001), `cap_drop: ALL`, `no-new-privileges: true`.
* **Ağ Güvenliği:** Yalnızca yerel `127.0.0.1` arayüzüne port bağlaması.
* **Ağırlık Güvenliği:** SHA-256 hash doğrulaması olmadan hiçbir model belleğe alınmaz (`WeightIntegrityError`).
* **Değişmez Analiz Kaydı:** Kriptografik SHA-256 imzalı JSON analiz sertifikası (`verify_record`).
* **Ayrıntılı Güvenlik Raporu:** [`docs/security-audit.md`](../../docs/security-audit.md)
