#!/usr/bin/env python3
"""
Evaluate image quality metrics comparing reconstructions to originals.

Metrics computed:
- MSE (Mean Squared Error) - lower is better
- MAE (Mean Absolute Error) - lower is better  
- PSNR (Peak Signal-to-Noise Ratio) - higher is better
- SSIM (Structural Similarity Index) - higher is better (0-1)
- LPIPS (Learned Perceptual Image Patch Similarity) - lower is better

Usage:
    uv run python evaluate_metrics.py
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
torch.backends.cudnn.enabled = False

# Suppress deprecation warnings from lpips/torchvision
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="lpips")

# Try to import LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not installed. Run 'pip install lpips' to enable LPIPS metric.")

# =============================================================================
# CONFIGURATION
# =============================================================================

ORIGINALS_DIR = Path("training_data/val/originals")
COMPRESSED_DIR = Path("training_data/val/compressed")
CLS_BASE_DIR = Path("evals/cls_base")
CLS_LORA_DIR = Path("evals/cls_lora")

OUTPUT_FILE = Path("evals/metrics_results.json")

# =============================================================================
# METRIC FUNCTIONS
# =============================================================================

def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Mean Squared Error between two images."""
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))

def compute_mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Mean Absolute Error between two images."""
    return float(np.mean(np.abs(img1.astype(np.float64) - img2.astype(np.float64))))

def compute_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images."""
    mse = compute_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return float(10 * np.log10((max_val ** 2) / mse))

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Structural Similarity Index between two images."""
    # SSIM expects images in (H, W, C) format
    return float(ssim(img1, img2, channel_axis=2, data_range=255))

class LPIPSMetric:
    """LPIPS metric wrapper with lazy loading."""
    
    def __init__(self):
        self.model = None
        self.device = None
    
    def _load_model(self):
        if self.model is None and LPIPS_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = lpips.LPIPS(net='alex').to(self.device)
            self.model.eval()
    
    def compute(self, img1: np.ndarray, img2: np.ndarray) -> Optional[float]:
        """Compute LPIPS distance between two images."""
        if not LPIPS_AVAILABLE:
            return None
        
        self._load_model()
        
        # Convert to torch tensors, normalize to [-1, 1]
        def to_tensor(img):
            t = torch.from_numpy(img).float().permute(2, 0, 1) / 127.5 - 1.0
            return t.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            t1 = to_tensor(img1)
            t2 = to_tensor(img2)
            dist = self.model(t1, t2)
        
        return float(dist.item())

# =============================================================================
# EVALUATION
# =============================================================================

def load_image(path: Path) -> np.ndarray:
    """Load image as numpy array in RGB format."""
    img = Image.open(path).convert("RGB")
    return np.array(img)

def compute_all_metrics(
    original: np.ndarray,
    comparison: np.ndarray,
    lpips_metric: LPIPSMetric
) -> Dict[str, float]:
    """Compute all metrics between original and comparison image."""
    metrics = {
        "mse": compute_mse(original, comparison),
        "mae": compute_mae(original, comparison),
        "psnr": compute_psnr(original, comparison),
        "ssim": compute_ssim(original, comparison),
    }
    
    lpips_val = lpips_metric.compute(original, comparison)
    if lpips_val is not None:
        metrics["lpips"] = lpips_val
    
    return metrics

def evaluate_method(
    method_name: str,
    method_dir: Path,
    originals_dir: Path,
    lpips_metric: LPIPSMetric,
    image_ids: List[str]
) -> Dict[str, Dict]:
    """Evaluate a reconstruction method against originals."""
    results = {}
    
    for img_id in tqdm(image_ids, desc=f"Evaluating {method_name}"):
        original_path = originals_dir / f"{img_id}.png"
        method_path = method_dir / f"{img_id}.png"
        
        if not method_path.exists():
            print(f"  Warning: {method_path} not found, skipping")
            continue
        
        original = load_image(original_path)
        comparison = load_image(method_path)
        
        # Ensure same size
        if original.shape != comparison.shape:
            print(f"  Warning: Size mismatch for {img_id}, skipping")
            continue
        
        metrics = compute_all_metrics(original, comparison, lpips_metric)
        results[img_id] = metrics
    
    return results

def compute_summary_stats(method_results: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
    """Compute summary statistics (mean, std) for each metric."""
    if not method_results:
        return {}
    
    # Collect all values per metric
    metric_values = {}
    for img_metrics in method_results.values():
        for metric_name, value in img_metrics.items():
            if metric_name not in metric_values:
                metric_values[metric_name] = []
            metric_values[metric_name].append(value)
    
    # Compute stats
    summary = {}
    for metric_name, values in metric_values.items():
        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    
    return summary

def main():
    print("=" * 70)
    print("IMAGE QUALITY METRICS EVALUATION")
    print("=" * 70)
    
    # Get list of validation images
    original_files = sorted(ORIGINALS_DIR.glob("*.png"))
    image_ids = [f.stem for f in original_files]
    
    print(f"\nFound {len(image_ids)} validation images")
    print(f"Image IDs: {image_ids[:5]}..." if len(image_ids) > 5 else f"Image IDs: {image_ids}")
    
    # Check which methods are available
    methods = {}
    
    if COMPRESSED_DIR.exists():
        methods["compressed"] = COMPRESSED_DIR
        print(f"\n✓ Compressed images: {COMPRESSED_DIR}")
    
    if CLS_BASE_DIR.exists():
        methods["cls_base"] = CLS_BASE_DIR
        print(f"✓ CLS Base images: {CLS_BASE_DIR}")
    
    if CLS_LORA_DIR.exists():
        methods["cls_lora"] = CLS_LORA_DIR
        print(f"✓ CLS LoRA images: {CLS_LORA_DIR}")
    else:
        print(f"✗ CLS LoRA images not found at {CLS_LORA_DIR}")
    
    if not methods:
        print("\nError: No methods found to evaluate!")
        return
    
    # Initialize LPIPS metric
    lpips_metric = LPIPSMetric()
    
    # Evaluate each method
    print("\n" + "=" * 70)
    print("COMPUTING METRICS")
    print("=" * 70)
    
    all_results = {
        "methods": {},
        "summary": {},
        "config": {
            "originals_dir": str(ORIGINALS_DIR),
            "num_images": len(image_ids),
            "image_ids": image_ids,
        }
    }
    
    for method_name, method_dir in methods.items():
        print(f"\n--- {method_name.upper()} ---")
        results = evaluate_method(
            method_name, method_dir, ORIGINALS_DIR, lpips_metric, image_ids
        )
        all_results["methods"][method_name] = results
        all_results["summary"][method_name] = compute_summary_stats(results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY (Mean ± Std)")
    print("=" * 70)
    
    metrics_order = ["mse", "mae", "psnr", "ssim", "lpips"]
    header = f"{'Metric':<10}"
    for method in methods.keys():
        header += f"{method:>20}"
    print(header)
    print("-" * (10 + 20 * len(methods)))
    
    for metric in metrics_order:
        row = f"{metric.upper():<10}"
        for method in methods.keys():
            summary = all_results["summary"].get(method, {})
            if metric in summary:
                mean = summary[metric]["mean"]
                std = summary[metric]["std"]
                row += f"{mean:>12.4f} ± {std:<6.4f}"
            else:
                row += f"{'N/A':>20}"
        print(row)
    
    # Save results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to {OUTPUT_FILE}")
    
    # Print interpretation guide
    print("\n" + "=" * 70)
    print("METRIC INTERPRETATION")
    print("=" * 70)
    print("  MSE   : Lower is better (0 = identical)")
    print("  MAE   : Lower is better (0 = identical)")
    print("  PSNR  : Higher is better (∞ = identical, >30dB is good)")
    print("  SSIM  : Higher is better (1 = identical, >0.9 is good)")
    print("  LPIPS : Lower is better (0 = identical)")

if __name__ == "__main__":
    main()