#!/usr/bin/env python3
"""
Evaluation script: Generate img2img outputs for validation set using CLS conditioning.
"""

import os
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from img2img import Img2ImgConfig, Img2ImgPipeline, extract_and_quantize_cls, dequantize_cls


# Configuration from user
CONFIG = Img2ImgConfig(
    strength=0.5,
    num_inference_steps=100,
    guidance_scale=3.0,
    seed=42,
    use_ilvr=False,
    use_cls_conditioning=True,
    ip_adapter_scale=0.6,
)


def load_val_pairs(data_dir: str = "training_data"):
    """Load validation image pairs."""
    val_dir = Path(data_dir) / "val"
    originals_dir = val_dir / "originals"
    compressed_dir = val_dir / "compressed"
    
    pairs = []
    for orig_file in sorted(originals_dir.glob("*.png")):
        comp_file = compressed_dir / orig_file.name
        if comp_file.exists():
            pairs.append({
                "original": str(orig_file),
                "compressed": str(comp_file),
                "name": orig_file.stem,
            })
    
    return pairs


def main():
    print("=" * 70)
    print("VALIDATION SET EVALUATION - IMG2IMG WITH CLS CONDITIONING")
    print("=" * 70)
    
    # Setup paths
    output_dir = Path("evals")
    output_dir.mkdir(exist_ok=True)
    
    # Print config
    print("\nConfiguration:")
    print(f"  strength:           {CONFIG.strength}")
    print(f"  num_inference_steps: {CONFIG.num_inference_steps}")
    print(f"  guidance_scale:     {CONFIG.guidance_scale}")
    print(f"  seed:               {CONFIG.seed}")
    print(f"  use_ilvr:           {CONFIG.use_ilvr}")
    print(f"  use_cls_conditioning: {CONFIG.use_cls_conditioning}")
    print(f"  ip_adapter_scale:   {CONFIG.ip_adapter_scale}")
    print(f"\nOutput directory: {output_dir.absolute()}")
    
    # Load validation pairs
    val_pairs = load_val_pairs()
    print(f"\nFound {len(val_pairs)} validation image pairs")
    
    if len(val_pairs) == 0:
        print("Error: No validation pairs found in training_data/val/")
        return
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"\nDevice: {device}")
    print(f"Dtype: {dtype}")
    
    # Initialize pipeline
    print("\nInitializing Img2Img pipeline...")
    pipeline = Img2ImgPipeline(config=CONFIG, device=device, dtype=dtype)
    
    # Process each validation image
    print(f"\nProcessing {len(val_pairs)} images...")
    print("-" * 70)
    
    for i, pair in enumerate(tqdm(val_pairs, desc="Generating")):
        name = pair["name"]
        original_path = pair["original"]
        compressed_path = pair["compressed"]
        
        print(f"\n[{i+1}/{len(val_pairs)}] Processing: {name}")
        
        # Load original image for CLS extraction
        original_img = Image.open(original_path).convert("RGB")
        
        # Extract and quantize CLS from original (simulating sender)
        print(f"  Extracting CLS from original...")
        quantized_cls = extract_and_quantize_cls(original_img, device=device)
        
        # Dequantize CLS (simulating receiver)
        cls_embedding = dequantize_cls(quantized_cls, device=device, dtype=dtype)
        
        # Load compressed image as reference
        pipeline.load_reference(compressed_path)
        
        # Generate with CLS conditioning
        print(f"  Generating with img2img + CLS...")
        generated = pipeline.generate(cls_embedding=cls_embedding)
        
        # Save generated image
        output_path = output_dir / f"{name}.png"
        generated.save(output_path)
        print(f"  Saved: {output_path}")
    
    print("\n" + "=" * 70)
    print(f"DONE! Generated {len(val_pairs)} images in {output_dir.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()

