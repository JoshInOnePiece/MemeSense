"""
Image augmentation script.
Takes all images in an input folder and outputs 4x the amount via augmentation.

Usage:
    python dataset_creater.py --input_dir <path> --output_dir <path>
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def random_flip(img: Image.Image) -> Image.Image:
    return ImageOps.mirror(img) if random.random() > 0.5 else img


def random_rotate(img: Image.Image, max_degrees: int = 20) -> Image.Image:
    angle = random.uniform(-max_degrees, max_degrees)
    return img.rotate(angle, expand=False, fillcolor=(128, 128, 128))


def random_brightness(img: Image.Image) -> Image.Image:
    factor = random.uniform(0.6, 1.4)
    return ImageEnhance.Brightness(img).enhance(factor)


def random_contrast(img: Image.Image) -> Image.Image:
    factor = random.uniform(0.6, 1.4)
    return ImageEnhance.Contrast(img).enhance(factor)


def random_saturation(img: Image.Image) -> Image.Image:
    factor = random.uniform(0.5, 1.5)
    return ImageEnhance.Color(img).enhance(factor)


def random_blur(img: Image.Image) -> Image.Image:
    if random.random() > 0.5:
        radius = random.uniform(0.5, 1.5)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    return img


def random_crop_and_resize(img: Image.Image, crop_fraction: float = 0.85) -> Image.Image:
    w, h = img.size
    new_w = int(w * random.uniform(crop_fraction, 1.0))
    new_h = int(h * random.uniform(crop_fraction, 1.0))
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    return img.crop((left, top, left + new_w, top + new_h)).resize((w, h), Image.LANCZOS)


# Each augmentation variant applies a distinct combination of transforms
AUGMENTATION_VARIANTS = [
    lambda img: random_flip(random_brightness(img)),
    lambda img: random_rotate(random_contrast(img)),
    lambda img: random_crop_and_resize(random_saturation(img)),
    lambda img: random_blur(random_flip(random_rotate(random_brightness(img)))),
]


def augment_image(img: Image.Image, variant_index: int) -> Image.Image:
    return AUGMENTATION_VARIANTS[variant_index % len(AUGMENTATION_VARIANTS)](img)


def get_save_format(extension: str) -> str:
    mapping = {
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".png": "PNG",
        ".bmp": "BMP",
        ".webp": "WEBP",
        ".tiff": "TIFF",
    }
    return mapping.get(extension.lower(), "PNG")


def process_directory(input_dir: str, output_dir: str) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_files:
        print(f"No supported images found in '{input_dir}'.")
        return

    print(f"Found {len(image_files)} image(s). Generating {len(image_files) * 4} augmented images...")

    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")

            stem = img_path.stem
            ext = img_path.suffix.lower()
            save_format = get_save_format(ext)

            # Copy the original
            original_out = output_path / img_path.name
            with Image.open(img_path) as orig:
                orig.convert("RGB").save(original_out, format=save_format)

            # Generate 3 augmented variants (original + 3 = 4x per image)
            for i in range(3):
                with Image.open(img_path) as orig:
                    augmented = augment_image(orig.convert("RGB"), variant_index=i)
                out_name = f"{stem}_aug{i + 1}{ext}"
                augmented.save(output_path / out_name, format=save_format)

        except Exception as e:
            print(f"  Skipping '{img_path.name}': {e}")
            continue

        print(f"  Processed: {img_path.name}")

    total_original = len(image_files)
    total_output = total_original * 4
    print(f"\nDone. {total_original} original(s) → {total_output} total images saved to '{output_dir}'.")


def main() -> None:
    # --- Change these two paths ---
    INPUT_DIR  = "C:\\Users\\samil\\Documents\\deeplearningoscs\\inputaugmenter"
    OUTPUT_DIR = "C:\\Users\\samil\\Documents\\deeplearningoscs\\outputaugmenter"
    # ------------------------------

    # Command-line args override the hardcoded paths above if provided
    parser = argparse.ArgumentParser(description="Augment images to 4x the original count.")
    parser.add_argument("--input_dir", default=INPUT_DIR, help="Folder containing source images.")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Folder to save augmented images.")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
