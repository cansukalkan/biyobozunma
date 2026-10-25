# Mikroskop Bozunma Analizi

CrackenPy veri seti üzerinde çatlak ve gözenek segmentasyonu yapan, piksel ölçümlerini zaman serisine ve bozunma hızına dönüştüren, token korumalı bir web arayüzü sunan ve Docker ile paketlenen analiz aracı.

## Özellikler

- **Veri Hattı:** Maskelerden YOLO çokgen etiketleme formatına otomatik ve kayıpsız dönüştürme.
- **Segmentasyon Modelleri:** YOLOv8/v11 ve U-Net tabanlı çatlak ve gözenek tespiti.
- **Kalibrasyon & Alan Ölçümü:** Piksel alanlarından mm² birimine güvenli dönüşüm (provisional/measured takip bayrağı).
- **Kinetik Bozunma Analizi:** Zaman serisi üstel bozunma uydurma ($C(t) = C_0 e^{-kt}$) ve değişmez SHA-256 kayıtları.
- **Güvenli Web Arayüzü:** Sabit zamanlı token doğrulaması, içerik tabanlı dosya yükleme doğrulaması, hız sınırı (rate limit).
- **Konteyner Desteği:** Güvenlik sıkılaştırmalı Docker & docker-compose dağıtımı.

## Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleme
uv sync

# Testleri çalıştırma
pytest

# Lint ve format
ruff check .
```

## Lisans

Bu proje BSD-3-Clause lisansı ile lisanslanmıştır. Detaylar için [docs/data-license.md](docs/data-license.md) dosyasına bakınız.
