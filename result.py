#!/usr/bin/env python3
"""
Generate paper-ready figure comparison of compression methods.

Compares: Original | CLS (VAE + CLS token) | LoRA Fine-tuned (VAE only)
Includes bpp (bits per pixel) compression rates in headers.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np

# Configuration
IMAGE_IDS = [9, 177]
ORIGINAL_DIR = Path("datasets/lighthouses")
CLS_DIR = Path("evals/cls")
LORA_DIR = Path("evals/lora_finetuned_no_cls")

# Image dimensions
ORIGINAL_SIZE = (1024, 1024)  # Original image size
VAE_INPUT_SIZE = (512, 512)   # VAE processes at this size
LATENT_SIZE = (64, 64)        # VAE latent space (512/8)
LATENT_CHANNELS = 4
QUANT_BITS = 3                # 3-bit quantization for VAE latents
CLS_BYTES = 1288              # CLS token size: 1280 int8 + 8 bytes metadata


def compute_bpp_cls():
    """
    Compute bpp for CLS method (VAE latents + CLS token).
    
    Transmitted data:
    - VAE latents: 4 channels × 64 × 64 spatial × 3 bits = 49,152 bits
    - CLS token: 1288 bytes = 10,304 bits
    - Total: 59,456 bits
    
    Original: 1024 × 1024 pixels
    """
    latent_bits = LATENT_CHANNELS * LATENT_SIZE[0] * LATENT_SIZE[1] * QUANT_BITS
    cls_bits = CLS_BYTES * 8
    total_bits = latent_bits + cls_bits
    total_pixels = ORIGINAL_SIZE[0] * ORIGINAL_SIZE[1]
    return total_bits / total_pixels


def compute_bpp_no_cls():
    """
    Compute bpp for LoRA method without CLS (VAE latents only).
    
    Transmitted data:
    - VAE latents: 4 channels × 64 × 64 spatial × 3 bits = 49,152 bits
    
    Original: 1024 × 1024 pixels
    """
    latent_bits = LATENT_CHANNELS * LATENT_SIZE[0] * LATENT_SIZE[1] * QUANT_BITS
    total_pixels = ORIGINAL_SIZE[0] * ORIGINAL_SIZE[1]
    return latent_bits / total_pixels


def create_comparison_figure():
    """Create paper-ready comparison figure."""
    
    # Compute bpp values
    bpp_cls = compute_bpp_cls()
    bpp_lora = compute_bpp_no_cls()
    
    # Figure setup: 2 rows (one per image), 3 columns (Original, CLS, LoRA)
    fig, axes = plt.subplots(
        len(IMAGE_IDS), 3,
        figsize=(12, 8.5),
        squeeze=False
    )
    
    # Reduce spacing
    plt.subplots_adjust(
        left=0.02, right=0.98,
        top=0.88, bottom=0.02,
        wspace=0.05, hspace=0.08
    )
    
    # Compression ratios (vs 24-bit RGB)
    ratio_cls = 24 / bpp_cls
    ratio_lora = 24 / bpp_lora
    
    # Column headers with bpp and compression ratio
    headers = [
        "Original",
        f"CLS (VAE + CLS)\n{bpp_cls:.3f} bpp",
        f"LoRA Fine-tuned \n{bpp_lora:.3f} bpp"
    ]
    
    # Process each image
    for row_idx, img_id in enumerate(IMAGE_IDS):
        # Load images
        orig_path = ORIGINAL_DIR / f"{img_id}.png"
        cls_path = CLS_DIR / f"{img_id}.png"
        lora_path = LORA_DIR / f"{img_id}.png"
        
        paths = [orig_path, cls_path, lora_path]
        
        for col_idx, path in enumerate(paths):
            ax = axes[row_idx, col_idx]
            
            if path.exists():
                img = mpimg.imread(str(path))
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, f"Missing:\n{path.name}", 
                       ha='center', va='center', fontsize=10,
                       transform=ax.transAxes)
            
            ax.axis('off')
            
            # Add headers to top row only
            if row_idx == 0:
                ax.set_title(headers[col_idx], fontsize=13, fontweight='bold', 
                           pad=10, fontfamily='serif')
    
    # Main title
    fig.suptitle(
        "Image Compression Comparison: CLS-Guided vs LoRA Fine-tuned Reconstruction",
        fontsize=15, fontweight='bold', fontfamily='serif', y=0.96
    )
    
    # Save figure
    output_path = Path("comparison_figure.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path.absolute()}")
    
    # Also save as PNG for quick viewing
    output_png = Path("comparison_figure.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_png.absolute()}")
    
    plt.close()
    
    # Print compression stats
    print("\n" + "=" * 50)
    print("COMPRESSION STATISTICS")
    print("=" * 50)
    print(f"\nCLS Method (VAE latents + CLS token):")
    print(f"  VAE latents: {LATENT_CHANNELS}×{LATENT_SIZE[0]}×{LATENT_SIZE[1]}×{QUANT_BITS} bits = {LATENT_CHANNELS * LATENT_SIZE[0] * LATENT_SIZE[1] * QUANT_BITS:,} bits")
    print(f"  CLS token:   {CLS_BYTES} bytes = {CLS_BYTES * 8:,} bits")
    print(f"  Total:       {LATENT_CHANNELS * LATENT_SIZE[0] * LATENT_SIZE[1] * QUANT_BITS + CLS_BYTES * 8:,} bits")
    print(f"  bpp:         {bpp_cls:.4f}")
    
    print(f"\nLoRA Method (VAE latents only):")
    print(f"  VAE latents: {LATENT_CHANNELS}×{LATENT_SIZE[0]}×{LATENT_SIZE[1]}×{QUANT_BITS} bits = {LATENT_CHANNELS * LATENT_SIZE[0] * LATENT_SIZE[1] * QUANT_BITS:,} bits")
    print(f"  bpp:         {bpp_lora:.4f}")
    
    print(f"\nOriginal image: {ORIGINAL_SIZE[0]}×{ORIGINAL_SIZE[1]} = {ORIGINAL_SIZE[0] * ORIGINAL_SIZE[1]:,} pixels")
    print(f"Compression ratio (CLS):  {24 / bpp_cls:.1f}× (vs 24-bit RGB)")
    print(f"Compression ratio (LoRA): {24 / bpp_lora:.1f}× (vs 24-bit RGB)")


if __name__ == "__main__":
    create_comparison_figure()

