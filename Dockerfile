# ==============================================================================
# Aşama 1: Derleyici (Builder)
# ==============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Güvenlik & Derleme gereksinimleri (derleme sonrası son imajdan atılır)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Sanal ortam oluştur
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Bağımlılıkları yükle
COPY pyproject.toml /build/
RUN pip install --upgrade pip setuptools wheel && \
    pip install ".[ml]"

# Proje kaynak kodunu yükle
COPY src/ /build/src/
RUN pip install --no-deps -e .

# ==============================================================================
# Aşama 2: Güvenli Çalışma Zamanı (Runtime)
# ==============================================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="Bozunma Analizi Ekibi" \
      description="Mikroskop Bozunma Analizi & Yapay Zekâ Destekli Kinetik Ölçüm Platformu" \
      version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BOZUNMA_ENV=production \
    BOZUNMA_RESULTS_DIR=/app/runs/web_results \
    PATH="/opt/venv/bin:$PATH"

# Çalışma zamanı için yalnızca gerekli kütüphaneler (libglib2.0 vb. OpenCV için)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# GÜVENLİK SERTLEŞTİRMESİ: Root olmayan kullanıcı ve grup (UID 10001)
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /sbin/nologin -m -d /home/appuser appuser

WORKDIR /app

# Derleyiciden sanal ortamı kopyala
COPY --from=builder /opt/venv /opt/venv

# Proje dosyalarını root yetkisi olmadan sadece okunabilir olarak kopyala
COPY --chown=appuser:appgroup src/ /app/src/
COPY --chown=appuser:appgroup configs/ /app/configs/
COPY --chown=appuser:appgroup models/ /app/models/
COPY --chown=appuser:appgroup data/ /app/data/
COPY --chown=appuser:appgroup pyproject.toml /app/

# Yazılabilir sonuç dizini oluştur ve sahibini appuser yap
RUN mkdir -p /app/runs/web_results && \
    chown -R appuser:appgroup /app/runs

# Root olmayan kullanıcıya geç
USER appuser:appgroup

# Yalnızca localhost/iç port
EXPOSE 8765

# Güvenli sağlık kontrolü (ek curl paketi olmadan saf Python ile)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/health').getcode() == 200 else 1)"

# Uygulama başlatma komutu (Fabrika fonksiyonu)
ENTRYPOINT ["uvicorn", "bozunma.web.app:get_app", "--factory", "--host", "0.0.0.0", "--port", "8765"]
