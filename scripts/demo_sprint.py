"""Sprint demo üretim betiği.

Farklı sprintler için görselleştirme ve doğrulama çıktıları üretir.
Sprint 0: Gerçek mikroskop verisinden 6 örnek görüntü ve maske bindirmesi (overlay).
"""

import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw


def create_overlay(
    image: Image.Image,
    mask: np.ndarray,
    alpha: float = 0.55,
    colors: dict[int, tuple[int, int, int]] | None = None,
) -> Image.Image:
    """Orijinal mikroskop görüntüsü üzerine maske sınıflarını renkli bindirir.

    Varsayılan renkler:
      - 2 (Crack / Çatlak): Kırmızı (255, 40, 40)
      - 3 (Pore / Gözenek): Mavi (30, 140, 255)
    """
    if colors is None:
        colors = {
            2: (255, 40, 40),  # Çatlak (Kırmızı)
            3: (30, 140, 255),  # Gözenek (Mavi)
        }

    img_rgb = image.convert("RGB")
    img_arr = np.array(img_rgb).astype(np.float32)

    overlay_arr = img_arr.copy()

    for cls_val, color in colors.items():
        cls_mask = mask == cls_val
        if np.any(cls_mask):
            for c in range(3):
                overlay_arr[cls_mask, c] = (1.0 - alpha) * img_arr[cls_mask, c] + alpha * color[c]

    return Image.fromarray(np.clip(overlay_arr, 0, 255).astype(np.uint8))


def create_side_by_side(
    image: Image.Image, mask: np.ndarray, overlay: Image.Image, title: str
) -> Image.Image:
    """Orijinal, Maske ve Renkli Bindirmeyi yan yana tek bir panelde birleştirir."""
    w, h = image.size

    # Maskeyi görsel RGB paletine çevir
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask_rgb[mask == 1] = (160, 160, 160)  # Matris (Gri)
    mask_rgb[mask == 2] = (255, 40, 40)  # Çatlak (Kırmızı)
    mask_rgb[mask == 3] = (30, 140, 255)  # Gözenek (Mavi)
    mask_img = Image.fromarray(mask_rgb)

    # Başlık alanı için üst boşluk
    header_h = 40
    total_w = w * 3 + 20
    total_h = h + header_h + 10

    canvas = Image.new("RGB", (total_w, total_h), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)

    # Panelleri yerleştir
    canvas.paste(image, (5, header_h))
    canvas.paste(mask_img, (w + 10, header_h))
    canvas.paste(overlay, (w * 2 + 15, header_h))

    # Etiketler
    draw.text((10, 12), f"{title} - Orijinal", fill=(220, 220, 220))
    draw.text((w + 15, 12), "Maske (Kirmizi: Catlak, Mavi: Gozenek)", fill=(220, 220, 220))
    draw.text((w * 2 + 20, 12), "Bindirme (Overlay)", fill=(220, 220, 220))

    return canvas


def run_sprint_0(
    data_dir: Path, demo_dir: Path, classes_cfg: Path, num_samples: int = 6
) -> list[Path]:
    """Sprint 0 demosu: İlk 6 gerçek veri çiftinden görsel bindirmeleri üretir."""
    demo_dir.mkdir(parents=True, exist_ok=True)

    with open(classes_cfg, "r", encoding="utf-8") as f:
        _ = yaml.safe_load(f)

    # data/raw altındaki çiftleri bul
    pairs: list[tuple[Path, Path]] = []
    for i in range(1, num_samples + 1):
        img_p = data_dir / f"ID14_470_sub_{i:02d}_Image.png"
        mask_p = data_dir / f"ID14_470_sub_{i:02d}_Mask.png"
        if img_p.exists() and mask_p.exists():
            pairs.append((img_p, mask_p))

    # Yeterli çift yoksa data/raw içinde arama yap
    if len(pairs) < num_samples:
        imgs = sorted(data_dir.glob("*_Image.png")) + sorted(data_dir.glob("*_Img*.png"))
        for img_p in imgs:
            # Maske eşini ara
            stem = img_p.stem.replace("Image", "Mask").replace("Img", "Mask")
            candidates = list(data_dir.glob(f"{stem}.*"))
            if candidates:
                mask_p = candidates[0]
                if (img_p, mask_p) not in pairs:
                    pairs.append((img_p, mask_p))
            if len(pairs) >= num_samples:
                break

    if not pairs:
        err_msg = (
            f"Gerçek veri bulunamadı! Lütfen önce `python scripts/fetch_data.py` "
            f"çalıştırın. Dizin: {data_dir}"
        )
        raise FileNotFoundError(err_msg)

    output_files: list[Path] = []
    for idx, (img_p, mask_p) in enumerate(pairs[:num_samples], start=1):
        img = Image.open(img_p).convert("RGB")
        mask = np.array(Image.open(mask_p))

        overlay = create_overlay(img, mask)
        composite = create_side_by_side(img, mask, overlay, title=img_p.stem)

        out_path = demo_dir / f"overlay_{idx:02d}.png"
        composite.save(out_path)
        output_files.append(out_path)
        print(f" [Demo 0] Üretildi: {out_path.name} ({img_p.name})")

    return output_files


def draw_yolo_polygons(
    image: Image.Image,
    lines: list[str],
    colors: dict[int, tuple[int, int, int]] | None = None,
) -> Image.Image:
    """YOLO çokgen çizgilerini görüntü üzerine çizer."""
    if colors is None:
        colors = {
            0: (255, 40, 40),   # Çatlak (Kırmızı)
            1: (30, 140, 255),  # Gözenek (Mavi)
        }

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    w, h = image.size

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        color = colors.get(cls_id, (0, 255, 0))

        coords = [float(x) for x in parts[1:]]
        pts = [(coords[i] * w, coords[i + 1] * h) for i in range(0, len(coords) - 1, 2)]

        if len(pts) >= 3:
            fill_col = (color[0], color[1], color[2], 120)
            draw.polygon(pts, fill=fill_col, outline=color)

    return canvas


def run_sprint_1(
    raw_dir: Path,
    yolo_dir: Path,
    demo_dir: Path,
    num_samples: int = 6,
) -> dict:
    """Sprint 1 demosu: Çokgen bindirmeleri, dönüşüm kaybı ve sınıf dağılımı."""
    from bozunma.data.convert import mask_to_yolo_lines
    from bozunma.data.validate import conversion_loss, validate_dataset
    from bozunma.data.writer import find_matching_mask

    demo_dir.mkdir(parents=True, exist_ok=True)

    # 1. Gerçek verileri bul ve çokgenleri oluştur
    images = sorted(raw_dir.glob("*_Image.png"))[:num_samples]
    if not images:
        images = [p for p in raw_dir.glob("*.png") if "mask" not in p.name.lower()][:num_samples]

    losses = []
    output_files = []

    for idx, img_p in enumerate(images, start=1):
        mask_p = find_matching_mask(img_p)
        if not mask_p or not mask_p.exists():
            continue

        img = Image.open(img_p)
        mask_arr = np.array(Image.open(mask_p))

        # Çatlak (2->0) ve Gözenek (3->1)
        lines = mask_to_yolo_lines(mask_arr, class_ids={2: 0, 3: 1}, min_area_px=10)

        # Dönüşüm kaybı
        sample_loss = conversion_loss(mask_arr, lines, class_value=2)
        losses.append(sample_loss)

        # Görsel bindirme
        poly_img = draw_yolo_polygons(img, lines)

        # Yan yana panel
        w, h = img.size
        header_h = 40
        panel = Image.new("RGB", (w * 2 + 10, h + header_h + 10), (25, 25, 25))
        pdraw = ImageDraw.Draw(panel)
        panel.paste(img, (5, header_h))
        panel.paste(poly_img, (w + 10, header_h))
        pdraw.text((10, 12), f"{img_p.stem} - Orijinal", fill=(220, 220, 220))
        pdraw.text(
            (w + 15, 12),
            f"YOLO Cokgenleri (Kayıp: %{sample_loss:.2f})",
            fill=(220, 220, 220),
        )

        out_path = demo_dir / f"polygon_overlay_{idx:02d}.png"
        panel.save(out_path)
        output_files.append(out_path)

    # Veri seti doğrulama raporu
    val_report = validate_dataset(yolo_dir)
    avg_loss = float(np.mean(losses)) if losses else 0.0

    # README oluştur
    readme_path = demo_dir / "README.md"
    readme_content = f"""# Sprint 1 Demosu: Veri Hattı ve Çokgen Segmentasyonu

## Özet Metrikler
- **Toplam Görüntü Sayısı:** {val_report.get('total_images', 0)}
- **Toplam Etiket Dosyası:** {val_report.get('total_labels', 0)}
- **Ortalama Dönüşüm Kaybı:** %{avg_loss:.2f}
- **Boş Etiket Sayısı:** {len(val_report.get('empty_labels', []))}

## Sınıf Dağılımı (Çokgen Adetleri)
| Sınıf ID | Sınıf Adı | Çokgen Sayısı |
| :--- | :--- | :--- |
| 0 | crack (çatlak) | {val_report.get('class_counts', {}).get(0, 0)} |
| 1 | pore (gözenek) | {val_report.get('class_counts', {}).get(1, 0)} |

## Örnek Çokgen Bindirme Görüntüleri
{chr(10).join([f"- ![{p.name}]({p.name})" for p in output_files])}
"""
    readme_path.write_text(readme_content, encoding="utf-8")

    return {
        "output_files": output_files,
        "avg_loss": avg_loss,
        "report": val_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint demo çıktıları üretici")
    parser.add_argument("--sprint", type=int, default=0, help="Sprint numarası (0-6)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--configs", type=Path, default=Path("configs/classes.yaml"))
    args = parser.parse_args()

    if args.sprint == 0:
        demo_dir = Path("demos/sprint-0")
        print(f"[*] Sprint 0 demosu çalıştırılıyor -> {demo_dir}")
        outputs = run_sprint_0(args.data_dir, demo_dir, args.configs, num_samples=6)
        print(f"[+] Başarılı: {len(outputs)} demo bindirme görüntüsü üretildi.")
    elif args.sprint == 1:
        demo_dir = Path("demos/sprint-1")
        print(f"[*] Sprint 1 demosu çalıştırılıyor -> {demo_dir}")
        res = run_sprint_1(args.data_dir, Path("data/yolo"), demo_dir, num_samples=6)
        print(
            f"[+] Başarılı: {len(res['output_files'])} görsel üretildi. "
            f"Ortalama kayıp: %{res['avg_loss']:.2f}"
        )
    else:
        print(f"Sprint {args.sprint} için demo betiği bu sprint kapsamında tanımlanacak.")


if __name__ == "__main__":
    main()

