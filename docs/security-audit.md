# Güvenlik ve Uyumluluk Sertleştirme Denetim Raporu (docs/security-audit.md)

Bu belge, **Bozunma Analizi** platformunun yazılım güvenliği, veri gizliliği, konteyner izolasyonu ve bağımlılık taraması denetim sonuçlarını belgeler.

---

## 1. Statik Güvenlik Taramaları (Static Analysis & Vulnerability Scans)

| Tarama Aracı | Kapsam | Bulgu Sayısı | Durum |
| :--- | :--- | :--- | :--- |
| **Bandit v1.9.4** | `src/` altındaki tüm Python kaynak kodu | **0 Kritik / 0 Yüksek / 0 Orta / 0 Düşük** | **GEÇTİ (Temiz)** |
| **Pip-Audit v2.6.0** | Yüklü Python paketleri ve bağımlılıklar | **0 Bilinen Güvenlik Zafiyeti** | **GEÇTİ (Temiz)** |
| **Ruff Linter** | Kod stili, güvensiz bloklar (`B`) | **0 Hata** | **GEÇTİ (Temiz)** |
| **Pytest Suite** | Tüm birim, entegrasyon ve web testleri | **73 / 73 Test Başarılı** | **GEÇTİ (Temiz)** |

---

## 2. Web API ve İletişim Güvenliği Sertleştirmesi

### 2.1. Kimlik Doğrulama & Zamanlama Saldırısı Koruması (Timing-Safe Auth)
* API istekleri `X-API-Token` başlığı üzerinden doğrulanır.
* Doğrulama `hmac.compare_digest(provided, expected)` ile sabit zamanda (**constant-time**) yapılır. Bu sayede parola uzunluğu veya karakter eşleşmesi üzerinden zamanlama saldırıları (**timing attack / side-channel**) imkansız kılınmıştır.

### 2.2. Girdi ve Dosya Yükleme Güvenliği (Input Sanitization)
* **Magic Byte Taraması:** Yalnızca gerçek PNG (`\x89PNG\r\n\x1a\n`), JPEG (`\xff\xd8\xff`) ve TIFF baytlarına izin verilir. Uzantı manipülasyonu ile gizlenmiş SVG, HTML veya ikili yürütülebilir dosyalar derhal reddedilir.
* **XSS Koruması:** SVG ve HTML dosyaları yükleme seviyesinde engellenir (`UploadRejected`).
* **Dekompresyon Bombası Koruması (Zip/Pixel Bomb):** `Image.MAX_IMAGE_PIXELS = 89,478,485` ve `MAX_UPLOAD_BYTES = 20 MB` sınırı ile bellek tüketimini hedefleyen DoS saldırıları engellenir.
* **Hız Sınırlaması (Rate Limiting):** Kötüye kullanımı ve DoS saldırılarını önlemek için IP başına dakikada maksimum 10 istek sınırı (`@limiter.limit("10/minute")`) uygulanır.
* **Gizlilik ve Bilgi İfşası (Privacy Leak Prevention):** Yüklenen orijinal dosya adı sistem içinde asla saklanmaz veya istemciye geri yansıtılmaz (`UUIDv4` kullanılır).

### 2.3. HTTP Güvenlik Başlıkları (Secure Headers)
FastAPI ara yazılımı (middleware) her yanıta aşağıdaki başlıkları ekler:
* `X-Content-Type-Options: nosniff` (MIME sniffing engelleme)
* `X-Frame-Options: DENY` (Clickjacking / IFrame içine alma engelleme)
* `Referrer-Policy: no-referrer` (Yönlendiren bilgi sızıntısını önleme)
* `Content-Security-Policy`:
  `default-src 'self'; img-src 'self' data: blob:; style-src 'self' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; script-src 'self'; frame-ancestors 'none'`
* Üretim ortamında (`BOZUNMA_ENV=production`) `/docs`, `/redoc` ve `/openapi.json` Swagger uç noktaları tamamen kapatılır.

---

## 3. Yapay Zekâ & Model Güvenliği (AI Model Integrity)

### 3.1. Kriptografik Model Ağırlığı Doğrulaması (SHA-256)
* Hiçbir derin öğrenme (`.pt`) veya makine öğrenmesi (`.joblib`) ağırlık dosyası, `models/MANIFEST.sha256` içindeki kriptografik özetle birebir eşleşmeden belleğe alınmaz.
* Tahrif edilmiş, bozulmuş veya pickle enjeksiyonu riski taşıyan modeller `WeightIntegrityError` ile engellenir.

### 3.2. Değişmez Analiz Kayıtları (Tamper-Evident Audit Trail)
* Zaman serisi ve kinetik bozunma hesaplamaları, tüm girdi dizisi ve kinetik parametreleri deterministik sıralama ile JSON olarak birleştirilip SHA-256 ile mühürlenir (`verify_record`).
* Tek bir piksel veya parametre değiştirilirse kayıt bütünlüğü geçersiz sayılır.

---

## 4. Konteyner Güvenliği (Docker Hardening)

`Dockerfile` ve `docker-compose.yml` şu güvenlik kontrollerini sağlar:
1. **Çok Aşamalı Derleme (Multi-Stage Build):** Derleyiciler (`gcc`, `build-essential`) üretim imajından tamamen dışlanır.
2. **Root Olmayan Kullanıcı (Non-Root User):**
   * Kullanıcı: `appuser` (UID: 10001, GID: 10001)
   * Kabuk: `/sbin/nologin`
3. **Yetki Yükseltme Koruması:** `security_opt: ["no-new-privileges:true"]`
4. **Linux Yeteneklerinin Kaldırılması:** `cap_drop: ["ALL"]`
5. **Yerel Ağ İzolasyonu:** Port dış dünyaya `0.0.0.0` yerine yalnızca yerel makineye bağlanır: `127.0.0.1:8765:8765`.
6. **Dahili Sağlık Denetimi:** `HEALTHCHECK` direktifi dış paket gerektirmeden saf Python standard kütüphanesi (`urllib.request`) ile çalışır.
