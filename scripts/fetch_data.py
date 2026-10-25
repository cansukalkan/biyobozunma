"""Resmi CrackenPy veri kaynağından gerçek verileri indirme ve doğrulama betiği."""

import argparse
import hashlib
import urllib.request
from pathlib import Path

from PIL import Image

RAW_REPO_BASE = "https://raw.githubusercontent.com/Rievil/CrackenPy/main"

# İndirilecek resmi örnek dosyalar
REMOTE_FILES = [
    {
        "url": f"{RAW_REPO_BASE}/Examples/Img/14_WG2_470_Img_cropeed.png",
        "dest": "14_WG2_470_Img_cropeed.png",
    },
    {
        "url": f"{RAW_REPO_BASE}/Examples/Img/14_WG2_470_Mask_cropped.tiff",
        "dest": "14_WG2_470_Mask_cropped.tiff",
    },
    {
        "url": f"{RAW_REPO_BASE}/Examples/Img/DeepLearning_example.png",
        "dest": "DeepLearning_example.png",
    },
]


def download_file(url: str, dest_path: Path) -> bytes:
    """Verilen URL'den dosyayı indirir ve hedefe yazar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (BozunmaAnalizi/1.0)"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    dest_path.write_bytes(data)
    return data


def create_real_sample_crops(
    img_path: Path, mask_path: Path, out_dir: Path, num_crops: int = 6, crop_size: int = 256
) -> list[tuple[Path, Path]]:
    """Gerçek 561x561 CrackenPy görüntüsünden ve maskesinden gerçek alt bölgeler üretir.

    Her alt bölge çatlak, gözenek ve matris sınıflarını içeren gerçek piksel verisidir.
    Numune adlandırma kuralı: ID14_470_sub_01_Image.png, ID14_470_sub_01_Mask.png
    """
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path)

    w, h = img.size
    crops = []

    # Belirlenmiş koordinatlardan gerçek bölgeler kes
    # (561x561 boyutunda 256x256 pencereler)
    positions = [
        (0, 0),
        (w - crop_size, 0),
        (0, h - crop_size),
        (w - crop_size, h - crop_size),
        ((w - crop_size) // 2, (h - crop_size) // 2),
        ((w - crop_size) // 2, 0),
    ]

    for idx, (x, y) in enumerate(positions[:num_crops], start=1):
        box = (x, y, x + crop_size, y + crop_size)
        crop_img = img.crop(box)
        crop_mask = mask.crop(box)

        img_name = f"ID14_470_sub_{idx:02d}_Image.png"
        mask_name = f"ID14_470_sub_{idx:02d}_Mask.png"

        img_out = out_dir / img_name
        mask_out = out_dir / mask_name

        crop_img.save(img_out)
        crop_mask.save(mask_out)
        crops.append((img_out, mask_out))

    return crops


def generate_checksums(data_dir: Path, manifest_path: Path) -> list[tuple[str, str]]:
    """Verilen klasördeki tüm dosyaların SHA-256 özetini çıkarıp manifest dosyasına yazar."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    records = []

    # Alfabetik sıralı tüm dosyalar
    for file_path in sorted(data_dir.rglob("*")):
        if file_path.is_file():
            rel_path = file_path.relative_to(data_dir).as_posix()
            hasher = hashlib.sha256()
            with open(file_path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            records.append((digest, rel_path))

    with open(manifest_path, "w", encoding="utf-8") as f:
        for digest, rel_path in records:
            f.write(f"{digest}  {rel_path}\n")

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="CrackenPy veri indirme ve doğrulama")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("docs/checksums.sha256"))
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Veriler indiriliyor -> {args.data_dir}")
    for item in REMOTE_FILES:
        dest = args.data_dir / item["dest"]
        print(f" - {item['dest']} indiriliyor...")
        download_file(item["url"], dest)

    img_ref = args.data_dir / "14_WG2_470_Img_cropeed.png"
    mask_ref = args.data_dir / "14_WG2_470_Mask_cropped.tiff"

    if img_ref.exists() and mask_ref.exists():
        print("[*] Gerçek örnek alt görüntüleri oluşturuluyor...")
        crops = create_real_sample_crops(img_ref, mask_ref, args.data_dir, num_crops=6)
        print(f" - {len(crops)} gerçek görüntü/maske çifti oluşturuldu.")

    print(f"[*] SHA-256 manifest üretiliyor -> {args.manifest}")
    records = generate_checksums(args.data_dir, args.manifest)
    print(f"[+] Başarılı: {len(records)} dosya özetlendi.")


if __name__ == "__main__":
    main()
