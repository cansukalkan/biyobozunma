# Sprint 0: Gerçek Veriyle İlk Demo

Bu klasör, Mikroskop Bozunma Analizi projesinin Sprint 0 aşamasında gerçek mikroskop verileriyle üretilmiş ilk görselleştirme çıktılarını içerir.

## 1. Veri Kaynağı
- **Kaynak:** CrackenPy Resmi Araştırma Veri Seti (Dvořák ve ark., 2024, Zenodo DOI: 10.5281/zenodo.13969747).
- **Lisans:** BSD-3-Clause (Doğrulandı, ticari ve yeniden dağıtıma uygun).
- **Veri Tipi:** Alkali aktif cüruf / beton numunelerinin optik mikroskop altındaki gerçek yüzey fotoğrafları ve karşılık gelen piksel etiketli zemin gerçeği (ground truth) maskeleri.
- **Bütünlük:** SHA-256 özetleri `docs/checksums.sha256` dosyasında kayıtlıdır ve doğrulanmıştır.

## 2. Üretilen Demo Çıktıları
Klasörde 6 adet gerçek numune alt bölgesi için Orijinal Görüntü | Zemin Gerçeği Maskesi | Renkli Bindirme (Overlay) panelleri yer almaktadır:
- `overlay_01.png`: `ID14` numunesi (470. saat), alt bölge 1.
- `overlay_02.png`: `ID14` numunesi (470. saat), alt bölge 2.
- `overlay_03.png`: `ID14` numunesi (470. saat), alt bölge 3.
- `overlay_04.png`: `ID14` numunesi (470. saat), alt bölge 4.
- `overlay_05.png`: `ID14` numunesi (470. saat), alt bölge 5.
- `overlay_06.png`: `ID14` numunesi (470. saat), alt bölge 6.

## 3. Renk Lejantı ve Anlamları
- **Kırmızı (`#FF2828`):** Çatlak (Crack) sınıfı (Ham piksel değeri: `2`).
- **Mavi (`#1E8CFF`):** Gözenek (Pore / Air void) sınıfı (Ham piksel değeri: `3`).
- **Gri (`#A0A0A0`):** Malzeme Matrisi (Matrix) sınıfı (Ham piksel değeri: `1`).
- **Siyah (`#000000`):** Arka plan (Background) sınıfı (Ham piksel değeri: `0`).

## 4. Bilinen Eksikler ve Gelecek Sprintler
- **Fiziksel Boyutlandırma:** Henüz mikrometre/piksel kalibrasyonu entegre edilmemiştir; alanlar piksel cinsindendir. Metrik ($mm^2$) dönüşümü Sprint 3 kalibrasyon modülüyle devreye alınacaktır.
- **Model Tahminleri:** Bu aşamada zemin gerçeği maskeleri gösterilmektedir; otomatik YOLO segmentasyonu Sprint 1 ve 2'de eğitilecek ve entegre edilecektir.
