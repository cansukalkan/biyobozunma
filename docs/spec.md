# Mikroskop Bozunma Analizi Spesifikasyonu (docs/spec.md)

## 1. Amaç
CrackenPy veri seti (ve mikroskobik malzeme yüzey görüntüleri) üzerinde çatlak ve gözenek yapılarını semantik olarak segmente eden, tespit edilen piksel alanlarını kalibre edilmiş fiziksel birimlere ($mm^2$) dönüştüren, numune bazında zaman serisi oluşturup birinci derece üstel kinetik bozunma modeli uyduran, token korumalı ve sertleştirilmiş bir web arayüzü ile sunulan uçtan uca analiz yazılımı geliştirmek.

## 2. Kapsam
- **Veri Hattı:** Maskelerden ikili kontur ve poligon çıkarımı, YOLO formatına normalizasyon, numune sızıntısını önleyen (leak-free) train/val/test bölmesi, dönüşüm kayıp analizi.
- **Modelleme & Karşılaştırma:** YOLO segmentasyon modeli ve U-Net referans modeli ile mIoU ve Dice metrikleri üzerinden kıyaslama.
- **Ağırlık Güvenliği:** Model ağırlıklarının (.pt) SHA-256 hash doğrulaması olmadan kesinlikle yüklenmemesi (WeightIntegrityError).
- **Ölçüm Katmanı:** Piksel -> $mm^2$ dönüşümü, kalibrasyon durumuna göre (PENDING/MEASURED) değerlerin "provisional" olarak etiketlenmesi.
- **Kinetik Modül:** $C(t) = C_0 \cdot e^{-k \cdot t}$ üstel bozunma uydurması ($k$ bozunma hızı sabiti ve standart hatası) ve SHA-256 imzalı değişmez analiz kaydı.
- **Web API & Arayüz:** FastAPI tabanlı, sabit zamanlı token kimlik doğrulamalı (`BOZUNMA_API_TOKEN`), magic byte içerik doğrulamalı (SVG/HTML/dekompresyon bombası korumalı), hız sınırlamalı (dakikada 10 istek) arayüz.
- **Dağıtım:** Çok aşamalı derleme, non-root kullanıcı ve yerel localhost bağlaması ile Docker / docker-compose dağıtımı.

## 3. Riskler ve Azaltma Stratejileri
- **Lisans Riski:** CrackenPy veri kaynağının lisansı ve atıf gereksinimleri (BSD-3-Clause) doğrulanmadan veri işlenmez.
- **Veri Sızıntısı:** Aynı numuneye ait farklı zaman adımı görüntülerinin hem eğitim hem test setine dağılması engellenir (`split_by_sample`).
- **Ağırlık Enjeksiyonu:** Güvenilmeyen pickle/PyTorch ağırlık dosyaları SHA-256 doğrulaması olmadan belleğe alınmaz.
- **Yükleme Güvenliği:** Uzantı manipülasyonu (polyglot/SVG/HTML) içerik baytları (magic bytes) taranarak engellenir; aşırı büyük görüntüler dekompresyon bombası filtresine takılır.

## 4. Ürün Metrikleri ve Başarı Kriterleri
- YOLO ve U-Net segmentasyonu için IoU > 0.65 ve Dice > 0.75.
- Dönüşüm kaybı (conversion loss) poligon alanında <%5.
- Kinetik uydurma standart hatası ve $R^2$ raporlaması.
- Statik güvenlik taramalarında (bandit, pip-audit, gitleaks, trivy) kritik ve yüksek seviyeli bulgu bulunmaması.
