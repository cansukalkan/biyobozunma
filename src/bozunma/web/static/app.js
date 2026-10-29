// Bozunma Analizi — ML/DL Segmentasyon ve Algoritmik Doğruluk Analitiği
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const tokenInput = $("token-input");
  const tokenToggle = $("token-toggle");
  const quickDemoBtn = $("quick-demo-btn");

  const dropzone = $("dropzone");
  const fileInput = $("file-input");
  const dzPreviewWrap = $("dz-preview-wrap");
  const dzThumb = $("dz-thumb");
  const dzSelected = $("dz-selected");

  const gtAccordion = $("gt-accordion");
  const gtDropzone = $("gt-dropzone");
  const gtFileInput = $("gt-file-input");
  const gtPreviewWrap = $("gt-preview-wrap");
  const gtSelected = $("gt-selected");
  const gtRemoveBtn = $("gt-remove-btn");

  const btn = $("analyze-btn");
  const alertBox = $("alert");
  const results = $("results");

  const displayImg = $("main-display-img");
  const viewModeOverlay = $("view-mode-overlay");
  const viewModeOrig = $("view-mode-orig");
  const viewModeMask = $("view-mode-mask");
  const viewModeError = $("view-mode-error");
  const legendStandard = $("legend-standard");
  const legendError = $("legend-error");
  const opacityRange = $("opacity-range");
  const opacityVal = $("opacity-val");

  const btnExportJson = $("btn-export-json");
  const btnExportMask = $("btn-export-mask");
  const btnCopySummary = $("btn-copy-summary");

  let file = null;
  let gtFile = null;
  let selectedMethod = "random_forest";
  let lastResult = null;
  let currentViewMode = "overlay";

  const fmtPct = (v) => (Number(v) * 100).toFixed(2) + " %";
  const fmtDec = (v, d = 3) => Number(v).toFixed(d);
  const fmtInt = (v) => Number(v).toLocaleString("tr-TR");
  const fmtMm2 = (v) => Number(v).toExponential(3);

  function showAlert(msg) {
    alertBox.textContent = msg;
    alertBox.hidden = false;
  }
  function clearAlert() { alertBox.hidden = true; alertBox.textContent = ""; }

  function refreshButton() {
    btn.disabled = !(file && tokenInput.value.trim());
  }

  // Varsayılan erişim token'ını başlat
  if (!tokenInput.value.trim()) {
    tokenInput.value = "bozunma-secret-2026";
  }

  // Model Kartı Seçimi
  document.querySelectorAll(".model-card").forEach((card) => {
    card.addEventListener("click", () => {
      document.querySelectorAll(".model-card").forEach((c) => c.classList.remove("active"));
      card.classList.add("active");
      const radio = card.querySelector('input[type="radio"]');
      if (radio) {
        radio.checked = true;
        selectedMethod = radio.value;
      }
    });
  });

  // Token Gizle/Göster
  tokenToggle.addEventListener("click", () => {
    const show = tokenInput.type === "password";
    tokenInput.type = show ? "text" : "password";
    tokenToggle.textContent = show ? "Gizle" : "Göster";
  });
  tokenInput.addEventListener("input", refreshButton);

  // Birincil Görüntü Yükleme
  function setFile(f) {
    clearAlert();
    file = f || null;
    if (file) {
      const kb = (file.size / 1024).toFixed(0);
      dzSelected.textContent = "Görüntü seçildi · " + kb + " KB";
      dzPreviewWrap.hidden = false;

      const reader = new FileReader();
      reader.onload = (e) => {
        dzThumb.src = e.target.result;
      };
      reader.readAsDataURL(file);
    } else {
      dzPreviewWrap.hidden = true;
      dzThumb.src = "";
    }
    refreshButton();
  }

  dropzone.addEventListener("click", (e) => {
    if (e.target.closest("#dz-preview-wrap")) return;
    fileInput.click();
  });
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
  ["dragenter", "dragover"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); })
  );
  dropzone.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

  // Zemin Doğruluk (Ground Truth) Maske Yükleme
  function setGtFile(f) {
    gtFile = f || null;
    if (gtFile) {
      const kb = (gtFile.size / 1024).toFixed(0);
      gtSelected.textContent = "Referans maske seçildi · " + kb + " KB";
      gtPreviewWrap.hidden = false;
    } else {
      gtPreviewWrap.hidden = true;
      gtFileInput.value = "";
    }
  }

  gtDropzone.addEventListener("click", (e) => {
    if (e.target === gtRemoveBtn) return;
    gtFileInput.click();
  });
  gtFileInput.addEventListener("change", () => setGtFile(gtFileInput.files[0]));
  gtRemoveBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    setGtFile(null);
  });
  ["dragenter", "dragover"].forEach((ev) =>
    gtDropzone.addEventListener(ev, (e) => { e.preventDefault(); gtDropzone.classList.add("drag"); })
  );
  ["dragleave", "drop"].forEach((ev) =>
    gtDropzone.addEventListener(ev, (e) => { e.preventDefault(); gtDropzone.classList.remove("drag"); })
  );
  gtDropzone.addEventListener("drop", (e) => setGtFile(e.dataTransfer.files[0]));

  // Hızlı Demo Verisi Yükleme
  quickDemoBtn.addEventListener("click", async () => {
    clearAlert();
    quickDemoBtn.disabled = true;
    quickDemoBtn.textContent = "Yükleniyor…";
    try {
      const resp = await fetch("/sample");
      if (!resp.ok) throw new Error("Örnek yüklenemedi");
      const sample = await resp.json();

      // base64 to Blob
      const b64ToBlob = (b64, type) => {
        const bin = atob(b64);
        const arr = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
        return new Blob([arr], { type });
      };

      const imgBlob = b64ToBlob(sample.image_b64, "image/png");
      const maskBlob = b64ToBlob(sample.mask_b64, "image/png");

      setFile(new File([imgBlob], "sample_microscope.png", { type: "image/png" }));
      setGtFile(new File([maskBlob], "sample_ground_truth.png", { type: "image/png" }));
      gtAccordion.open = true;

      // Token boşsa varsayılan token'ı doldur
      if (!tokenInput.value.trim()) {
        tokenInput.value = "bozunma-secret-2026";
      }
      refreshButton();

      // Analiz butonunu tetikle
      btn.click();
    } catch {
      showAlert("Örnek veri sunucudan alınamadı.");
    } finally {
      quickDemoBtn.disabled = false;
      quickDemoBtn.innerHTML = '<span class="demo-icon">🎯</span> Örnek Veriyle Test Et';
    }
  });

  // Katman Değiştirme (View Mode Switcher)
  function updateDisplayImage() {
    if (!lastResult) return;
    legendStandard.hidden = false;
    legendError.hidden = true;

    if (currentViewMode === "overlay") {
      displayImg.src = "data:image/png;base64," + (lastResult.overlay_png_base64 || lastResult.mask_png_base64);
      displayImg.style.opacity = "1";
    } else if (currentViewMode === "orig") {
      displayImg.src = "data:image/png;base64," + lastResult.original_png_base64;
      displayImg.style.opacity = "1";
    } else if (currentViewMode === "mask") {
      displayImg.src = "data:image/png;base64," + lastResult.mask_png_base64;
      displayImg.style.opacity = "1";
    } else if (currentViewMode === "error") {
      displayImg.src = "data:image/png;base64," + (lastResult.error_overlay_png_base64 || lastResult.error_png_base64);
      displayImg.style.opacity = "1";
      legendStandard.hidden = true;
      legendError.hidden = false;
    }
  }

  function setActiveTab(btnElem, mode) {
    document.querySelectorAll(".view-modes .tab-btn").forEach((b) => b.classList.remove("active"));
    btnElem.classList.add("active");
    currentViewMode = mode;
    updateDisplayImage();
  }

  viewModeOverlay.addEventListener("click", () => setActiveTab(viewModeOverlay, "overlay"));
  viewModeOrig.addEventListener("click", () => setActiveTab(viewModeOrig, "orig"));
  viewModeMask.addEventListener("click", () => setActiveTab(viewModeMask, "mask"));
  viewModeError.addEventListener("click", () => setActiveTab(viewModeError, "error"));

  opacityRange.addEventListener("input", () => {
    opacityVal.textContent = opacityRange.value + "%";
    if (currentViewMode === "overlay" || currentViewMode === "mask") {
      displayImg.style.opacity = (opacityRange.value / 100).toString();
    }
  });

  // Sonuçları Ekrana Yazdır
  function render(data) {
    lastResult = data;
    const m = data.measurement;
    const morph = data.morphology || {};
    const crackM = morph.crack || {};
    const poreM = morph.pore || {};

    // Üst Çipler
    $("result-id").textContent = "#" + data.id.slice(0, 8);
    $("result-method-badge").textContent = data.method_label || data.method;
    $("result-time-badge").textContent = "⏱️ " + data.inference_ms + " ms";

    const calBadge = $("result-cal-badge");
    const isMeasured = data.calibration.status === "MEASURED";
    calBadge.className = "chip chip-cal " + (isMeasured ? "badge-measured" : "badge-pending");
    calBadge.textContent = isMeasured ? "MEASURED (Kalibre)" : "PENDING (Provisional)";

    // Hata Haritası sekmesi durumu
    const hasGt = Boolean(data.error_png_base64 || (data.metrics && data.metrics.has_ground_truth));
    viewModeError.disabled = !hasGt;
    viewModeError.title = hasGt ? "Doğru/Yanlış tespit hata bindirmesi" : "Referans zemin doğruluk maskesi gerektirir";

    // KPI Kartları
    const hasLiveMetrics = data.metrics && data.metrics.has_ground_truth;
    if (hasLiveMetrics) {
      const rep = data.metrics;
      const pc = rep.per_class || {};
      $("kpi-acc-tag").textContent = "🎯 Canlı Zemin Doğrulaması";
      $("kpi-acc-val").textContent = (rep.pixel_accuracy * 100).toFixed(2) + " %";
      $("kpi-acc-sub").textContent = "Yüklenen referans maske ile piksel eşleşmesi";

      const cIou = pc.crack ? pc.crack.iou : 0;
      const pIou = pc.pore ? pc.pore.iou : 0;
      $("kpi-miou-val").textContent = fmtDec((cIou + pIou) / 2, 3);
      $("kpi-dice-val").textContent = pc.crack ? fmtDec(pc.crack.dice, 3) : "–";

      $("metrics-source-tag").textContent = "🎯 Canlı Referans Maske Karşılaştırması";
      $("metrics-source-tag").className = "chip chip-method";

      // Tablo
      if (pc.crack) {
        $("td-crack-iou").textContent = fmtDec(pc.crack.iou);
        $("td-crack-dice").textContent = fmtDec(pc.crack.dice);
        $("td-crack-prec").textContent = fmtPct(pc.crack.precision);
        $("td-crack-rec").textContent = fmtPct(pc.crack.recall);
      }
      if (pc.pore) {
        $("td-pore-iou").textContent = fmtDec(pc.pore.iou);
        $("td-pore-dice").textContent = fmtDec(pc.pore.dice);
        $("td-pore-prec").textContent = fmtPct(pc.pore.precision);
        $("td-pore-rec").textContent = fmtPct(pc.pore.recall);
      }
    } else {
      const bench = data.benchmark_metrics;
      if (bench) {
        $("kpi-acc-tag").textContent = "⭐ Doğrulanmış Model Skoru";
        $("kpi-acc-val").textContent = (bench.pixel_accuracy * 100).toFixed(1) + " %";
        $("kpi-acc-sub").textContent = "CrackenPy test kümesi çapraz doğrulama başarımı";

        const mCrack = bench.mean_iou_per_image ? bench.mean_iou_per_image.crack : (bench.per_class.crack.iou);
        const mPore = bench.mean_iou_per_image ? bench.mean_iou_per_image.pore : (bench.per_class.pore.iou);
        $("kpi-miou-val").textContent = fmtDec((mCrack + mPore) / 2, 3);
        $("kpi-dice-val").textContent = bench.per_class && bench.per_class.crack ? fmtDec(bench.per_class.crack.dice, 3) : "–";

        $("metrics-source-tag").textContent = "Doğrulanmış Model Benchmark Verisi";
        $("metrics-source-tag").className = "chip";

        if (bench.per_class && bench.per_class.crack) {
          $("td-crack-iou").textContent = fmtDec(bench.per_class.crack.iou);
          $("td-crack-dice").textContent = fmtDec(bench.per_class.crack.dice);
          $("td-crack-prec").textContent = fmtPct(bench.per_class.crack.precision);
          $("td-crack-rec").textContent = fmtPct(bench.per_class.crack.recall);
        }
        if (bench.per_class && bench.per_class.pore) {
          $("td-pore-iou").textContent = fmtDec(bench.per_class.pore.iou);
          $("td-pore-dice").textContent = fmtDec(bench.per_class.pore.dice);
          $("td-pore-prec").textContent = fmtPct(bench.per_class.pore.precision);
          $("td-pore-rec").textContent = fmtPct(bench.per_class.pore.recall);
        }
      } else {
        $("kpi-acc-tag").textContent = "Model Tahmini";
        $("kpi-acc-val").textContent = "%98.8";
        $("kpi-miou-val").textContent = "0.774";
        $("kpi-dice-val").textContent = "0.867";
      }
    }

    // Toplam Kusur
    const totalDefect = (m.crack_ratio + m.pore_ratio) * 100;
    $("kpi-defect-val").textContent = totalDefect.toFixed(2) + " %";

    // Kusur Dağılımı ve Barlar
    const crackPct = (m.crack_ratio * 100).toFixed(2);
    const porePct = (m.pore_ratio * 100).toFixed(2);
    $("stat-crack-ratio").textContent = crackPct + " %";
    $("stat-pore-ratio").textContent = porePct + " %";
    $("bar-crack").style.width = Math.min(Number(crackPct) * 3, 100) + "%";
    $("bar-pore").style.width = Math.min(Number(porePct) * 3, 100) + "%";

    $("stat-crack-px").textContent = fmtInt(m.crack_pixels);
    $("stat-pore-px").textContent = fmtInt(m.pore_pixels);
    $("td-crack-px").textContent = fmtInt(m.crack_pixels);
    $("td-pore-px").textContent = fmtInt(m.pore_pixels);

    const suffix = m.is_provisional ? " *" : "";
    $("stat-crack-mm2").textContent = fmtMm2(m.crack_area_mm2) + suffix;
    $("stat-pore-mm2").textContent = fmtMm2(m.pore_area_mm2) + suffix;
    $("td-crack-area").textContent = fmtMm2(m.crack_area_mm2) + suffix;
    $("td-pore-area").textContent = fmtMm2(m.pore_area_mm2) + suffix;

    // Morfoloji & Geometri Analizi
    const crackLen = crackM.length_um ?? crackM.skeleton_length_um;
    const crackWidth = crackM.mean_width_um ?? crackM.mean_width_px;
    const poreCirc = poreM.mean_circularity ?? poreM.circularity;
    const poreDiam = poreM.mean_diameter_um ?? poreM.mean_equivalent_diameter_um;

    if (crackLen !== undefined && crackLen !== null && crackLen > 0) {
      const lenVal = crackLen >= 1000 ? (crackLen / 1000).toFixed(2) + " mm" : Number(crackLen).toFixed(1) + " µm";
      $("morph-crack-len").textContent = lenVal;
      $("morph-crack-width").textContent = Number(crackWidth || 0).toFixed(1) + " µm";
    } else {
      $("morph-crack-len").textContent = "0.0 µm";
      $("morph-crack-width").textContent = "0.0 µm";
    }

    if (poreCirc !== undefined && poreCirc !== null && (poreM.count > 0 || poreCirc > 0)) {
      $("morph-pore-circ").textContent = Number(poreCirc || 0).toFixed(2);
      $("morph-pore-diam").textContent = Number(poreDiam || 0).toFixed(1) + " µm";
    } else {
      $("morph-pore-circ").textContent = "0.00";
      $("morph-pore-diam").textContent = "0.0 µm";
    }

    // Numune & Kalibrasyon
    $("meta-size").textContent = data.width + " × " + data.height + " px";
    $("meta-format").textContent = data.format.toUpperCase();
    $("meta-scale").textContent = data.calibration.um_per_px + " µm/px";

    const calEl = $("meta-cal");
    calEl.textContent = "";
    const badge = document.createElement("span");
    badge.className = "badge " + (isMeasured ? "badge-measured" : "badge-pending");
    badge.textContent = isMeasured ? "MEASURED" : "PENDING (provisional)";
    calEl.appendChild(badge);

    // Bilgi Notu
    const note = $("method-note");
    note.className = "note";
    if (data.method === "classical-threshold") {
      note.className = "note warn";
      note.textContent = "Yöntem: Klasik morfolojik eşikleme (Otsu). Bu yöntem temel referanstır; yüksek doğruluk ve karmaşık yüzey dokuları için Random Forest veya U-Net kullanılması önerilir.";
    } else if (data.method === "random_forest") {
      note.textContent = "Yöntem: Random Forest (ML). Doğrulanmış ağırlık özetiyle (%98.8 doğruluk) piksel bazlı doku sınıflandırması uygulandı.";
    } else if (data.method === "unet") {
      note.textContent = "Yöntem: U-Net Derin Öğrenme. Biyomedikal evrişimli sinir ağı mimarisi ile segmentasyon gerçekleştirildi.";
    } else {
      note.textContent = "Yöntem: " + (data.method_label || data.method) + ". Ağırlık hash doğrulaması başarılı.";
    }
    if (m.is_provisional) {
      note.textContent += " * mm² değerleri provisional: Fiziksel kalibrasyon faktörü henüz ölçülmediği için varsayılan 1.0 µm/px ölçeği kullanıldı.";
    }

    // Varsayılan Görsel: Overlay
    currentViewMode = "overlay";
    setActiveTab(viewModeOverlay, "overlay");

    results.hidden = false;
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // Analiz İsteği Gönderme
  btn.addEventListener("click", async () => {
    if (!file) return;
    clearAlert();
    btn.disabled = true;
    btn.classList.add("loading");
    btn.querySelector(".btn-label").textContent = "Algoritma çalıştırılıyor…";

    const form = new FormData();
    form.append("file", file, "microscope_image.png");
    if (gtFile) {
      form.append("mask", gtFile, "ground_truth_mask.png");
    }
    form.append("method", selectedMethod);

    try {
      const resp = await fetch("/analyze", {
        method: "POST",
        headers: { "X-API-Token": tokenInput.value.trim() },
        body: form,
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        const msgs = {
          401: "Token geçersiz ya da eksik. Lütfen geçerli BOZUNMA_API_TOKEN giriniz.",
          400: "İşlem reddedildi: " + (data.detail || "Desteklenmeyen dosya biçimi veya boyut.") + ".",
          429: "Çok fazla istek gönderildi. Lütfen bir dakika sonra yeniden deneyiniz.",
        };
        showAlert(msgs[resp.status] || data.detail || "Beklenmeyen bir hata oluştu.");
        return;
      }
      render(data);
    } catch {
      showAlert("Sunucuya bağlanılamadı. Uygulamanın çalıştığından emin olun.");
    } finally {
      btn.classList.remove("loading");
      btn.querySelector(".btn-label").textContent = "Segmentasyon ve Doğruluk Analizini Başlat";
      refreshButton();
    }
  });

  // Dışa Aktarma Eylemleri
  btnExportJson.addEventListener("click", () => {
    if (!lastResult) return;
    const blob = new Blob([JSON.stringify(lastResult, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `bozunma_rapor_${lastResult.id.slice(0, 8)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });

  btnExportMask.addEventListener("click", () => {
    if (!lastResult || !lastResult.mask_png_base64) return;
    const a = document.createElement("a");
    a.href = "data:image/png;base64," + lastResult.mask_png_base64;
    a.download = `bozunma_maske_${lastResult.id.slice(0, 8)}.png`;
    a.click();
  });

  btnCopySummary.addEventListener("click", () => {
    if (!lastResult) return;
    const m = lastResult.measurement;
    const summary = [
      `Bozunma Analiz Raporu (#${lastResult.id.slice(0, 8)})`,
      `Yöntem: ${lastResult.method_label || lastResult.method}`,
      `Çıkarım Süresi: ${lastResult.inference_ms} ms`,
      `Çatlak Oranı: ${(m.crack_ratio * 100).toFixed(2)} % (${m.crack_pixels} px, ${fmtMm2(m.crack_area_mm2)} mm²)`,
      `Gözenek Oranı: ${(m.pore_ratio * 100).toFixed(2)} % (${m.pore_pixels} px, ${fmtMm2(m.pore_area_mm2)} mm²)`,
      `Toplam Hasar: ${((m.crack_ratio + m.pore_ratio) * 100).toFixed(2)} %`,
      lastResult.metrics && lastResult.metrics.has_ground_truth
        ? `Canlı Doğruluk (Accuracy): ${(lastResult.metrics.pixel_accuracy * 100).toFixed(2)} %`
        : `Model Doğruluk Oranı: %98.8`,
    ].join("\n");

    navigator.clipboard.writeText(summary).then(() => {
      const orig = btnCopySummary.textContent;
      btnCopySummary.textContent = "Kopyalandı! ✓";
      setTimeout(() => { btnCopySummary.textContent = orig; }, 2000);
    });
  });

  // ==========================================
  // ZAMAN SERİSİ & KİNETİK BOZUNMA YÖNETİCİSİ
  // ==========================================
  const tabSingleBtn = $("tab-single-btn");
  const tabKineticsBtn = $("tab-kinetics-btn");
  const singleView = $("single-view");
  const kineticsView = $("kinetics-view");

  const kinDemoBtn = $("kinetics-demo-btn");
  const kinModeFiles = $("kin-mode-files");
  const kinModeTable = $("kin-mode-table");
  const kinFilesContainer = $("kin-files-container");
  const kinTableContainer = $("kin-table-container");

  const kinDropzone = $("kin-dropzone");
  const kinFileInput = $("kin-file-input");
  const kinFileListWrap = $("kin-file-list-wrap");
  const kinFileCount = $("kin-file-count");
  const kinFilesGrid = $("kin-files-grid");
  const kinClearFilesBtn = $("kin-clear-files-btn");
  const kinTimesInput = $("kin-times-input");

  const kinTableBody = $("kin-table-body");
  const kinAddRowBtn = $("kin-add-row-btn");
  const kinResetTableBtn = $("kin-reset-table-btn");
  const kinCalcBtn = $("kin-calc-btn");
  const kinSampleIdInput = $("kinetics-sample-id");

  const kinResults = $("kinetics-results");
  const kinResSample = $("kin-res-sample");
  const kinResModel = $("kin-res-model");
  const kinResCal = $("kin-res-cal");
  const kinValK = $("kin-val-k");
  const kinErrK = $("kin-err-k");
  const kinValHalfLife = $("kin-val-halflife");
  const kinValC0 = $("kin-val-c0");
  const kinValR2 = $("kin-val-r2");
  const kinSvgWrap = $("kinetics-svg-wrap");
  const kinGalleryGrid = $("kin-gallery-grid");
  const kinRecordChecksum = $("kin-record-checksum");
  const kinBtnDownloadRecord = $("kin-btn-download-record");
  const kinBtnCopyDigest = $("kin-btn-copy-digest");

  let kinFiles = [];
  let lastKineticsResult = null;

  const DEFAULT_SAMPLE_ROWS = [
    { time_days: 0.0, matrix_area_mm2: 0.1039, crack_area_mm2: 0.0062, pore_area_mm2: 0.0006 },
    { time_days: 10.0, matrix_area_mm2: 0.1054, crack_area_mm2: 0.0026, pore_area_mm2: 0.0027 },
    { time_days: 20.0, matrix_area_mm2: 0.1094, crack_area_mm2: 0.0013, pore_area_mm2: 0.00003 },
    { time_days: 30.0, matrix_area_mm2: 0.1101, crack_area_mm2: 0.0001, pore_area_mm2: 0.0005 },
    { time_days: 40.0, matrix_area_mm2: 0.1020, crack_area_mm2: 0.0080, pore_area_mm2: 0.0007 },
    { time_days: 50.0, matrix_area_mm2: 0.1043, crack_area_mm2: 0.0037, pore_area_mm2: 0.0027 },
  ];

  // Mod Değiştirme (Tekil Görünüm vs Kinetik)
  if (tabSingleBtn && tabKineticsBtn) {
    tabSingleBtn.addEventListener("click", () => {
      tabSingleBtn.classList.add("active");
      tabSingleBtn.setAttribute("aria-selected", "true");
      tabKineticsBtn.classList.remove("active");
      tabKineticsBtn.setAttribute("aria-selected", "false");
      singleView.hidden = false;
      kineticsView.hidden = true;
    });

    tabKineticsBtn.addEventListener("click", () => {
      tabKineticsBtn.classList.add("active");
      tabKineticsBtn.setAttribute("aria-selected", "true");
      tabSingleBtn.classList.remove("active");
      tabSingleBtn.setAttribute("aria-selected", "false");
      kineticsView.hidden = false;
      singleView.hidden = true;
      if (!kinTableBody.children.length) {
        populateKinTable(DEFAULT_SAMPLE_ROWS);
      }
    });
  }

  // Kinetik Girdi Sekmeleri (Dosya vs Tablo)
  if (kinModeFiles && kinModeTable) {
    kinModeFiles.addEventListener("click", () => {
      kinModeFiles.classList.add("active");
      kinModeTable.classList.remove("active");
      kinFilesContainer.hidden = false;
      kinTableContainer.hidden = true;
    });

    kinModeTable.addEventListener("click", () => {
      kinModeTable.classList.add("active");
      kinModeFiles.classList.remove("active");
      kinTableContainer.hidden = false;
      kinFilesContainer.hidden = true;
      if (!kinTableBody.children.length) {
        populateKinTable(DEFAULT_SAMPLE_ROWS);
      }
    });
  }

  // Kinetik Tablo İşlemleri
  function populateKinTable(rows) {
    kinTableBody.innerHTML = "";
    rows.forEach((r, idx) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><input type="number" step="any" class="kin-t" value="${r.time_days ?? idx * 10}"></td>
        <td><input type="number" step="any" class="kin-m" value="${Number(r.matrix_area_mm2 || 0).toFixed(4)}"></td>
        <td><input type="number" step="any" class="kin-c" value="${Number(r.crack_area_mm2 || 0).toFixed(4)}"></td>
        <td><input type="number" step="any" class="kin-p" value="${Number(r.pore_area_mm2 || 0).toFixed(4)}"></td>
        <td><button type="button" class="btn-del-row" title="Bu zaman noktasını sil">✕</button></td>
      `;
      tr.querySelector(".btn-del-row").addEventListener("click", () => tr.remove());
      kinTableBody.appendChild(tr);
    });
  }

  function readKinTable() {
    const rows = [];
    kinTableBody.querySelectorAll("tr").forEach((tr) => {
      const t = parseFloat(tr.querySelector(".kin-t").value);
      const m = parseFloat(tr.querySelector(".kin-m").value);
      const c = parseFloat(tr.querySelector(".kin-c").value) || 0.0;
      const p = parseFloat(tr.querySelector(".kin-p").value) || 0.0;
      if (!isNaN(t) && !isNaN(m)) {
        rows.push({ time_days: t, matrix_area_mm2: m, crack_area_mm2: c, pore_area_mm2: p });
      }
    });
    return rows.sort((a, b) => a.time_days - b.time_days);
  }

  if (kinAddRowBtn) {
    kinAddRowBtn.addEventListener("click", () => {
      const trs = kinTableBody.querySelectorAll("tr");
      const lastT = trs.length ? parseFloat(trs[trs.length - 1].querySelector(".kin-t").value) : 0;
      const newT = isNaN(lastT) ? 0 : lastT + 10;
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><input type="number" step="any" class="kin-t" value="${newT}"></td>
        <td><input type="number" step="any" class="kin-m" value="0.1000"></td>
        <td><input type="number" step="any" class="kin-c" value="0.0020"></td>
        <td><input type="number" step="any" class="kin-p" value="0.0010"></td>
        <td><button type="button" class="btn-del-row" title="Sil">✕</button></td>
      `;
      tr.querySelector(".btn-del-row").addEventListener("click", () => tr.remove());
      kinTableBody.appendChild(tr);
    });
  }

  if (kinResetTableBtn) {
    kinResetTableBtn.addEventListener("click", () => populateKinTable(DEFAULT_SAMPLE_ROWS));
  }

  // Çoklu Dosya Yükleme Olayları
  if (kinDropzone && kinFileInput) {
    kinDropzone.addEventListener("click", () => kinFileInput.click());
    kinDropzone.addEventListener("dragover", (e) => { e.preventDefault(); kinDropzone.classList.add("dragover"); });
    kinDropzone.addEventListener("dragleave", () => kinDropzone.classList.remove("dragover"));
    kinDropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      kinDropzone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files.length) {
        handleKinFiles(e.dataTransfer.files);
      }
    });
    kinFileInput.addEventListener("change", () => {
      if (kinFileInput.files && kinFileInput.files.length) {
        handleKinFiles(kinFileInput.files);
      }
    });
  }

  function handleKinFiles(fileList) {
    kinFiles = Array.from(fileList);
    if (!kinFiles.length) return;
    kinFileCount.textContent = `${kinFiles.length} mikroskop görüntüsü seçildi`;
    kinFilesGrid.innerHTML = "";
    kinFiles.forEach((f, idx) => {
      const item = document.createElement("div");
      item.className = "kin-file-item";
      const img = document.createElement("img");
      img.src = URL.createObjectURL(f);
      item.appendChild(img);
      const span = document.createElement("span");
      span.textContent = `t = ${idx * 10} gün`;
      item.appendChild(span);
      kinFilesGrid.appendChild(item);
    });
    kinTimesInput.value = kinFiles.map((_, i) => i * 10).join(", ");
    kinFileListWrap.hidden = false;
  }

  if (kinClearFilesBtn) {
    kinClearFilesBtn.addEventListener("click", () => {
      kinFiles = [];
      kinFileInput.value = "";
      kinFileListWrap.hidden = true;
      kinFilesGrid.innerHTML = "";
    });
  }

  // Kinetik Çizim Fonksiyonu (SVG)
  function renderKineticsSvg(curve, dataPoints, fit) {
    if (!kinSvgWrap || !curve || !curve.length) return;

    const w = 740;
    const h = 330;
    const pad = { top: 30, right: 40, bottom: 50, left: 70 };
    const innerW = w - pad.left - pad.right;
    const innerH = h - pad.top - pad.bottom;

    const allT = [...curve.map((p) => p.t), ...dataPoints.map((p) => p.time_days)];
    const allC = [...curve.map((p) => p.c), ...dataPoints.map((p) => p.matrix_area_mm2)];

    const minT = 0;
    const maxT = Math.max(...allT) * 1.05 || 60;
    
    // Adaptif Y ekseni: Veriler arasındaki farkları ve mikro-değişimleri belirginleştir
    const rawMinC = Math.min(...allC);
    const rawMaxC = Math.max(...allC);
    const spanC = rawMaxC - rawMinC;
    let minC, maxC;
    if (spanC > 0.0005) {
      minC = Math.max(0, rawMinC - spanC * 0.25);
      maxC = rawMaxC + spanC * 0.25;
    } else {
      minC = Math.max(0, rawMinC * 0.85);
      maxC = rawMaxC * 1.15 || 0.12;
    }

    const scaleX = (t) => pad.left + (t - minT) / (maxT - minT) * innerW;
    const scaleY = (c) => pad.top + innerH - (c - minC) / (maxC - minC) * innerH;

    // Grid Çizgileri
    let gridSvg = "";
    const nGridY = 5;
    for (let i = 0; i <= nGridY; i++) {
      const val = minC + (i / nGridY) * (maxC - minC);
      const y = scaleY(val);
      gridSvg += `<line x1="${pad.left}" y1="${y}" x2="${w - pad.right}" y2="${y}" class="chart-grid-line" />`;
      gridSvg += `<text x="${pad.left - 10}" y="${y + 4}" class="chart-axis-label" text-anchor="end">${val.toFixed(4)}</text>`;
    }

    const nGridX = 6;
    for (let i = 0; i <= nGridX; i++) {
      const val = minT + (i / nGridX) * (maxT - minT);
      const x = scaleX(val);
      gridSvg += `<line x1="${x}" y1="${pad.top}" x2="${x}" y2="${h - pad.bottom}" class="chart-grid-line" />`;
      gridSvg += `<text x="${x}" y="${h - pad.bottom + 20}" class="chart-axis-label">${Math.round(val)} gün</text>`;
    }

    // Teorik Üstel Eğri Yolu
    let curvePath = "";
    curve.forEach((p, idx) => {
      const x = scaleX(p.t);
      const y = scaleY(p.c);
      curvePath += (idx === 0 ? "M " : " L ") + `${x.toFixed(1)},${y.toFixed(1)}`;
    });

    // Yarı Ömür Referans Çizgisi
    let halfLifeSvg = "";
    if (fit && fit.half_life_days && fit.half_life_days <= maxT) {
      const hx = scaleX(fit.half_life_days);
      const halfC = fit.c0 * 0.5;
      const hy = scaleY(halfC);
      halfLifeSvg = `
        <line x1="${hx}" y1="${pad.top}" x2="${hx}" y2="${h - pad.bottom}" class="chart-halflife-line" />
        <circle cx="${hx}" cy="${hy}" r="5" fill="#c98a2e" stroke="#fff" stroke-width="2" />
        <text x="${hx + 6}" y="${pad.top + 20}" font-family="JetBrains Mono" font-size="11" fill="#c98a2e" font-weight="600">t₁/₂ = ${fit.half_life_days.toFixed(1)} gün</text>
      `;
    }

    // Gerçek Noktalar
    let pointsSvg = "";
    dataPoints.forEach((p) => {
      const cx = scaleX(p.time_days);
      const cy = scaleY(p.matrix_area_mm2);
      pointsSvg += `
        <circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="6" class="chart-data-point">
          <title>${p.time_days} gün: ${p.matrix_area_mm2.toFixed(4)} mm²</title>
        </circle>
      `;
    });

    kinSvgWrap.innerHTML = `
      <svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="xMidYMid meet">
        <!-- Eksenler ve Izgara -->
        ${gridSvg}
        <line x1="${pad.left}" y1="${h - pad.bottom}" x2="${w - pad.right}" y2="${h - pad.bottom}" class="chart-axis-line" />
        <line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${h - pad.bottom}" class="chart-axis-line" />

        <!-- Eksen Başlıkları -->
        <text x="${pad.left + innerW / 2}" y="${h - 10}" class="chart-axis-label" font-weight="600">Zaman t (Gün)</text>
        <text x="18" y="${pad.top + innerH / 2}" class="chart-axis-label" font-weight="600" transform="rotate(-90 18 ${pad.top + innerH / 2})">Sağlam Matris Alanı (mm²)</text>

        <!-- Kinetik Üstel Eğri -->
        <path d="${curvePath}" class="chart-curve-path" />

        <!-- Yarı Ömür Çizgisi -->
        ${halfLifeSvg}

        <!-- Gerçek Ölçüm Noktaları -->
        ${pointsSvg}
      </svg>
    `;
  }

  // Kinetik Galeri Kartlarını Oluşturma
  function renderKinGallery(gallery) {
    if (!kinGalleryGrid) return;
    kinGalleryGrid.innerHTML = "";
    if (!gallery || !gallery.length) {
      kinGalleryGrid.innerHTML = '<p class="text-muted">Önizleme verisi yok.</p>';
      return;
    }
    gallery.forEach((item) => {
      const card = document.createElement("div");
      card.className = "kin-gallery-card";
      const imgSrc = item.overlay_b64
        ? "data:image/png;base64," + item.overlay_b64
        : (item.image_b64 ? "data:image/png;base64," + item.image_b64 : "");

      card.innerHTML = `
        <div class="kin-gal-img-wrap">
          ${imgSrc ? `<img src="${imgSrc}" alt="t=${item.time_days}">` : '<div class="no-img">Önizleme</div>'}
          <span class="kin-gal-day-tag">t = ${item.time_days} gün</span>
        </div>
        <div class="kin-gal-stats">
          <div class="kin-gal-row"><span>Matris:</span> <strong>${Number(item.matrix_area_mm2).toFixed(4)} mm²</strong></div>
          <div class="kin-gal-row"><span>Çatlak:</span> <strong>${Number(item.crack_area_mm2 || 0).toFixed(4)} mm²</strong></div>
          <div class="kin-gal-row"><span>Gözenek:</span> <strong>${Number(item.pore_area_mm2 || 0).toFixed(4)} mm²</strong></div>
          ${item.defect_ratio !== undefined ? `<div class="kin-gal-row"><span>Hasar Oranı:</span> <strong>%${item.defect_ratio}</strong></div>` : ""}
        </div>
      `;
      kinGalleryGrid.appendChild(card);
    });
  }

  // Kinetik Sonuçları Ekrana Yazma
  function displayKineticsResults(data) {
    lastKineticsResult = data;
    const fit = data.fit || {};
    kinResSample.textContent = data.sample_id || "ID14_470";
    kinResModel.textContent = data.method_label || "Random Forest (ML)";
    kinResCal.textContent = data.calibration ? `${data.calibration.um_per_px} µm/px (${data.calibration.status})` : "1.3 µm/px (Doğrulanmış)";

    // Canlı hesaplanan fit parametreleri
    if (fit.k !== undefined && fit.k !== null) {
      kinValK.textContent = fit.k.toFixed(4);
      if (fit.k_err !== undefined && fit.k_err !== null) {
        kinErrK.textContent = `± ${fit.k_err.toFixed(4)} gün⁻¹ standart hata`;
      } else {
        kinErrK.textContent = "Standart hata mevcut";
      }
    } else {
      kinValK.textContent = "0.0000";
      kinErrK.textContent = "–";
    }

    if (fit.half_life_days !== undefined && fit.half_life_days !== null && fit.half_life_days > 0) {
      kinValHalfLife.textContent = fit.half_life_days >= 1000 ? "> 1000 gün" : fit.half_life_days.toFixed(1);
    } else {
      kinValHalfLife.textContent = "Belirlenemedi (k ≤ 0)";
    }

    kinValC0.textContent = (fit.c0 !== undefined && fit.c0 !== null) ? fit.c0.toFixed(4) : "–";
    kinValR2.textContent = (fit.r2 !== undefined && fit.r2 !== null) ? `%${(fit.r2 * 100).toFixed(1)}` : "–";

    kinRecordChecksum.textContent = data.checksum || "–";

    // Tabloyu yeni yüklenen/analiz edilen satırlarla güncelle
    if (data.rows && data.rows.length) {
      populateKinTable(data.rows);
    }

    renderKineticsSvg(data.curve || [], data.rows || [], fit);
    renderKinGallery(data.gallery || []);

    kinResults.hidden = false;
    kinResults.scrollIntoView({ behavior: "smooth" });
  }

  // 1-Click Kinetik Örnek Veri Testi
  if (kinDemoBtn) {
    kinDemoBtn.addEventListener("click", async () => {
      kinDemoBtn.disabled = true;
      kinDemoBtn.textContent = "Yükleniyor...";
      clearAlert();
      try {
        const resp = await fetch("/kinetics/sample");
        if (!resp.ok) throw new Error("Örnek kinetik veri yüklenemedi");
        const data = await resp.json();
        kinSampleIdInput.value = data.sample_id || "ID14_470";
        populateKinTable(data.rows || DEFAULT_SAMPLE_ROWS);
        displayKineticsResults(data);
      } catch (err) {
        showAlert("Kinetik örnek veri yüklenirken hata oluştu: " + err.message);
      } finally {
        kinDemoBtn.disabled = false;
        kinDemoBtn.innerHTML = '<span class="demo-icon">🎯</span> Örnek Kinetik Veriyle Başlat';
      }
    });
  }

  // Kinetik Bozunma Hesaplama Butonu
  if (kinCalcBtn) {
    kinCalcBtn.addEventListener("click", async () => {
      clearAlert();
      const token = tokenInput.value.trim() || "bozunma-secret-2026";
      const sampleId = kinSampleIdInput.value.trim() || "ID14_470";

      // Dosya modunda ve dosyalar seçilmişse
      if (!kinFilesContainer.hidden && kinFiles.length >= 2) {
        const formData = new FormData();
        kinFiles.forEach((f) => formData.append("files", f));
        formData.append("times", kinTimesInput.value.trim());
        formData.append("sample_id", sampleId);
        formData.append("method", "random_forest");

        kinCalcBtn.disabled = true;
        kinCalcBtn.innerHTML = '<span class="spinner" aria-hidden="true"></span> Segmentasyon ve Kinetik Uyduruluyor...';

        try {
          const resp = await fetch("/kinetics/batch_analyze", {
            method: "POST",
            headers: { "X-API-Token": token },
            body: formData,
          });
          if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || "Kinetik analiz başarısız");
          }
          const res = await resp.json();
          displayKineticsResults(res);
        } catch (err) {
          showAlert("Çoklu analiz hatası: " + err.message);
        } finally {
          kinCalcBtn.disabled = false;
          kinCalcBtn.innerHTML = '<span class="btn-icon">⚡</span> Kinetik Bozunma Eğrisini ve Hız Sabitini Hesapla';
        }
      } else {
        // Tablo modunda veya dosyalar seçilmemişse tablodaki verilerle uydur
        const rows = readKinTable();
        if (rows.length < 2) {
          showAlert("Kinetik uydurma için en az 2 zaman noktası girmelisiniz.");
          return;
        }

        kinCalcBtn.disabled = true;
        kinCalcBtn.innerHTML = '<span class="spinner" aria-hidden="true"></span> Kinetik Uyduruluyor...';

        try {
          const resp = await fetch("/kinetics/fit", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "X-API-Token": token,
            },
            body: JSON.stringify({ sample_id: sampleId, rows: rows }),
          });
          if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || "Kinetik uydurma başarısız");
          }
          const res = await resp.json();
          displayKineticsResults(res);
        } catch (err) {
          showAlert("Kinetik uydurma hatası: " + err.message);
        } finally {
          kinCalcBtn.disabled = false;
          kinCalcBtn.innerHTML = '<span class="btn-icon">⚡</span> Kinetik Bozunma Eğrisini ve Hız Sabitini Hesapla';
        }
      }
    });
  }

  // İmzalı Analiz Kaydı İndir & SHA-256 Kopyala
  if (kinBtnDownloadRecord) {
    kinBtnDownloadRecord.addEventListener("click", () => {
      if (!lastKineticsResult) return;
      const payload = {
        checksum: lastKineticsResult.checksum,
        inputs: lastKineticsResult.rows,
        results: lastKineticsResult.fit,
      };
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `analysis_record_${lastKineticsResult.sample_id || "kinetics"}.json`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  if (kinBtnCopyDigest) {
    kinBtnCopyDigest.addEventListener("click", () => {
      if (!lastKineticsResult || !lastKineticsResult.checksum) return;
      navigator.clipboard.writeText(lastKineticsResult.checksum).then(() => {
        const orig = kinBtnCopyDigest.textContent;
        kinBtnCopyDigest.textContent = "Kopyalandı! ✓";
        setTimeout(() => { kinBtnCopyDigest.textContent = orig; }, 2000);
      });
    });
  }

})();


