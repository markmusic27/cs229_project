#!/usr/bin/env python3
"""
Generate training data for LoRA finetuning.

Takes original images, compresses them through the VAE pipeline,
and creates train/val/test splits (80/10/10).

Usage:
    uv run python scripts/generate_training_data.py --input path/to/images
    uv run python scripts/generate_training_data.py --input datasets/lighthouses --output training_data
"""

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from tqdm import tqdm
from vae import VAE


def get_image_files(input_dir: str) -> List[Path]:
    """Find all image files in directory."""
    input_path = Path(input_dir)
    
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    image_files = []
    
    for ext in extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    return sorted(image_files)


def split_dataset(
    files: List[Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split files into train/val/test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    random.seed(seed)
    shuffled = files.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]
    
    return train, val, test


def process_images(
    image_files: List[Path],
    output_dir: Path,
    split_name: str,
    vae: VAE,
    target_size: Tuple[int, int] = (1024, 1024),
):
    """Process images through VAE and save original/compressed pairs."""
    
    originals_dir = output_dir / split_name / "originals"
    compressed_dir = output_dir / split_name / "compressed"
    
    originals_dir.mkdir(parents=True, exist_ok=True)
    compressed_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in tqdm(image_files, desc=f"Processing {split_name}"):
        try:
            # Load and resize original
            original = Image.open(img_path).convert("RGB")
            original = original.resize(target_size, Image.LANCZOS)
            
            # Save original as PNG
            original_out = originals_dir / f"{img_path.stem}.png"
            original.save(original_out, "PNG")
            
            # Compress through VAE
            image_tensor, original_size = vae.load_image(str(original_out))
            packed, latent_min, latent_max, original_shape = vae.encode_and_pack(image_tensor)
            
            # Decode and save compressed
            compressed_out = compressed_dir / f"{img_path.stem}.png"
            vae.unpack_and_decode(
                packed, original_shape, latent_min, latent_max,
                target_size, str(compressed_out)
            )
            
        except Exception as e:
            print(f"  Warning: Failed to process {img_path.name}: {e}")
            continue


def generate_training_data(
    input_dir: str,
    output_dir: str = "training_data",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    target_size: Tuple[int, int] = (1024, 1024),
    vae_quant: int = 3,
    seed: int = 42,
):
    """
    Generate training data with train/val/test splits.
    
    Args:
        input_dir: Directory containing original images
        output_dir: Output directory for training data
        train_ratio: Fraction for training set (default 0.8)
        val_ratio: Fraction for validation set (default 0.1)
        test_ratio: Fraction for test set (default 0.1)
        target_size: Target image size (default 1024x1024)
        vae_quant: VAE quantization bits (default 3)
        seed: Random seed for reproducibility
    """
    
    print("=" * 60)
    print("Training Data Generator")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    print(f"Target size: {target_size}")
    print(f"VAE quantization: {vae_quant}-bit")
    print()
    
    # Find all images
    image_files = get_image_files(input_dir)
    print(f"Found {len(image_files)} images")
    
    if len(image_files) == 0:
        print(f"Error: No images found in {input_dir}")
        print("Supported formats: PNG, JPG, JPEG, WebP, BMP")
        return
    
    # Split dataset
    train_files, val_files, test_files = split_dataset(
        image_files, train_ratio, val_ratio, test_ratio, seed
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_files)} images ({train_ratio*100:.0f}%)")
    print(f"  Val:   {len(val_files)} images ({val_ratio*100:.0f}%)")
    print(f"  Test:  {len(test_files)} images ({test_ratio*100:.0f}%)")
    
    # Initialize VAE
    print(f"\nInitializing VAE ({vae_quant}-bit quantization)...")
    vae = VAE(quant=vae_quant, model_dims=(512, 512))
    
    # Create output directory
    output_path = Path(output_dir)
    if output_path.exists():
        print(f"\nWarning: {output_dir} already exists")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return
        shutil.rmtree(output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    print("\n" + "=" * 60)
    print("Processing images...")
    print("=" * 60)
    
    process_images(train_files, output_path, "train", vae, target_size)
    process_images(val_files, output_path, "val", vae, target_size)
    process_images(test_files, output_path, "test", vae, target_size)
    
    # Save split info
    split_info = {
        "train": [f.name for f in train_files],
        "val": [f.name for f in val_files],
        "test": [f.name for f in test_files],
    }
    
    import json
    with open(output_path / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("✓ Training data generated successfully!")
    print("=" * 60)
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── originals/   ({len(train_files)} images)")
    print(f"  │   └── compressed/  ({len(train_files)} images)")
    print(f"  ├── val/")
    print(f"  │   ├── originals/   ({len(val_files)} images)")
    print(f"  │   └── compressed/  ({len(val_files)} images)")
    print(f"  ├── test/")
    print(f"  │   ├── originals/   ({len(test_files)} images)")
    print(f"  │   └── compressed/  ({len(test_files)} images)")
    print(f"  └── split_info.json")
    
    # Estimate storage
    avg_size_mb = 4  # Approximate MB per image pair
    total_mb = len(image_files) * avg_size_mb * 2  # originals + compressed
    print(f"\nEstimated storage: ~{total_mb} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data for LoRA finetuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    uv run python scripts/generate_training_data.py --input datasets/lighthouses
    
    # Custom output directory
    uv run python scripts/generate_training_data.py --input my_images --output my_training_data
    
    # Different split ratios
    uv run python scripts/generate_training_data.py --input images --train 0.7 --val 0.15 --test 0.15
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input directory containing original images"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="training_data",
        help="Output directory for training data (default: training_data)"
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--test",
        type=float,
        default=0.1,
        help="Test set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Target image size (default: 1024)"
    )
    parser.add_argument(
        "--quant",
        type=int,
        default=3,
        help="VAE quantization bits (default: 3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        print(f"Error: Split ratios must sum to 1.0, got {total}")
        return
    
    generate_training_data(
        input_dir=args.input,
        output_dir=args.output,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        target_size=(args.size, args.size),
        vae_quant=args.quant,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

