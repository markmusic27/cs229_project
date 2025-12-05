"""
Ensure an image is at least 1024x1024:
- If smaller, scale up until both sides >= 1024
- Then center-crop to exactly 1024x1024
- Never scale down
- Supports PNG and JPG inputs
- ALWAYS outputs PNG
"""

from PIL import Image
import sys
from pathlib import Path

TARGET = 1024


def upscale_and_crop(input_path: Path, output_path: Path) -> None:
    img = Image.open(input_path)

    # Convert image modes appropriately
    if img.mode not in ("RGB", "RGBA"):
        # If palette or other, convert to RGBA for safety
        img = img.convert("RGBA")
    else:
        # Leave RGB as RGB, PNGs often RGBA
        img = img.convert(img.mode)

    w, h = img.size

    # Step 1: Scale up if needed (never scale down)
    scale = max(TARGET / w, TARGET / h, 1)
    if scale > 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        w, h = new_w, new_h

    # Step 2: Center crop to 1024x1024
    left = (w - TARGET) // 2
    top = (h - TARGET) // 2
    right = left + TARGET
    bottom = top + TARGET

    cropped = img.crop((left, top, right, bottom))

    # Always output PNG
    output_path = output_path.with_suffix(".png")
    cropped.save(output_path, format="PNG")

    print(f"Processed {input_path} → {output_path} ({w}x{h} → 1024x1024 PNG)")


def main():
    if len(sys.argv) != 3:
        print("Usage: python scale_crop_1024.py input_image output_image")
        sys.exit(1)

    upscale_and_crop(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
