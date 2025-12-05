#!/usr/bin/env python3
"""
LoRA Training for SDXL Img2Img Compression Reconstruction.

This script trains a LoRA adapter to improve image reconstruction
from VAE-compressed inputs.

Usage:
    uv run python scripts/train_lora.py --data training_data
    
    # With custom settings:
    uv run python scripts/train_lora.py --data training_data --epochs 50 --lr 1e-4
    
    # Resume from checkpoint:
    uv run python scripts/train_lora.py --data training_data --resume lora_output/checkpoint-20
"""

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionXLImg2ImgPipeline, DDIMScheduler
from peft import LoraConfig, get_peft_model, PeftModel


@dataclass
class TrainingConfig:
    """Training configuration with all hyperparameters."""
    # Data
    data_dir: str = "training_data"
    output_dir: str = "lora_output"
    
    # Training
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    seed: int = 42
    
    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Checkpointing
    save_every: int = 20
    validate_every: int = 10
    
    # Model
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    prompt: str = "a high quality photograph"
    
    def save(self, path: str):
        """Save config to JSON."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load config from JSON."""
        with open(path) as f:
            return cls(**json.load(f))


@dataclass  
class TrainingMetrics:
    """Metrics tracked during training."""
    epoch: int
    train_loss: float
    val_loss: Optional[float]
    learning_rate: float
    epoch_time: float
    total_time: float
    best_val_loss: float
    
    def to_dict(self) -> dict:
        return asdict(self)


class TrainingLogger:
    """Logger for training metrics."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "training_log.jsonl"
        self.metrics_history: List[Dict] = []
        
    def log(self, metrics: TrainingMetrics):
        """Log metrics to file and history."""
        self.metrics_history.append(metrics.to_dict())
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(metrics.to_dict()) + "\n")
    
    def print_epoch(self, metrics: TrainingMetrics):
        """Print formatted epoch summary."""
        print("\n" + "=" * 70)
        print(f"EPOCH {metrics.epoch} SUMMARY")
        print("=" * 70)
        print(f"  Train Loss:      {metrics.train_loss:.6f}")
        if metrics.val_loss is not None:
            print(f"  Val Loss:        {metrics.val_loss:.6f}")
        print(f"  Learning Rate:   {metrics.learning_rate:.2e}")
        print(f"  Epoch Time:      {metrics.epoch_time:.1f}s")
        print(f"  Total Time:      {metrics.total_time/60:.1f}min")
        print(f"  Best Val Loss:   {metrics.best_val_loss:.6f}")
        print("=" * 70)
    
    def save_summary(self, config: TrainingConfig):
        """Save final training summary."""
        summary = {
            "config": asdict(config),
            "metrics_history": self.metrics_history,
            "final_train_loss": self.metrics_history[-1]["train_loss"] if self.metrics_history else None,
            "best_val_loss": min([m["val_loss"] for m in self.metrics_history if m["val_loss"] is not None], default=None),
            "total_epochs": len(self.metrics_history),
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(self.output_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)


def load_image_pairs(data_dir: str, split: str) -> List[Dict[str, str]]:
    """Load image pairs from a split directory."""
    split_dir = Path(data_dir) / split
    originals_dir = split_dir / "originals"
    compressed_dir = split_dir / "compressed"
    
    if not originals_dir.exists() or not compressed_dir.exists():
        return []
    
    pairs = []
    for orig_file in originals_dir.glob("*.png"):
        comp_file = compressed_dir / orig_file.name
        if comp_file.exists():
            pairs.append({
                "original": str(orig_file),
                "compressed": str(comp_file),
                "name": orig_file.stem,
            })
    
    return pairs


def compute_loss(
    pipe,
    original_path: str,
    compressed_path: str,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    add_time_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute training loss for a single image pair."""
    
    # Load images
    original = Image.open(original_path).convert("RGB")
    compressed = Image.open(compressed_path).convert("RGB")
    
    # Preprocess
    original_tensor = pipe.image_processor.preprocess(original).to(device, dtype)
    
    with torch.no_grad():
        # Encode original to latents (target)
        original_latents = pipe.vae.encode(original_tensor).latent_dist.sample()
        original_latents = original_latents * pipe.vae.config.scaling_factor
    
    # Sample random timestep
    timesteps = torch.randint(0, 1000, (1,), device=device).long()
    
    # Add noise to original
    noise = torch.randn_like(original_latents)
    noisy_latents = pipe.scheduler.add_noise(original_latents, noise, timesteps)
    
    # Predict noise
    noise_pred = pipe.unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs={
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids,
        },
    ).sample
    
    # MSE loss
    loss = F.mse_loss(noise_pred, noise)
    return loss


def validate(
    pipe,
    val_pairs: List[Dict],
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    add_time_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """Compute validation loss."""
    pipe.unet.eval()
    total_loss = 0
    
    with torch.no_grad():
        for pair in val_pairs:
            loss = compute_loss(
                pipe, pair["original"], pair["compressed"],
                prompt_embeds, pooled_prompt_embeds, add_time_ids,
                device, dtype
            )
            total_loss += loss.item()
    
    pipe.unet.train()
    return total_loss / len(val_pairs) if val_pairs else 0


def save_checkpoint(
    pipe,
    optimizer,
    scheduler,
    epoch: int,
    config: TrainingConfig,
    metrics: TrainingMetrics,
    output_dir: str,
    name: str = None,
):
    """Save a training checkpoint."""
    checkpoint_name = name or f"checkpoint-{epoch}"
    checkpoint_dir = Path(output_dir) / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save LoRA weights
    pipe.unet.save_pretrained(checkpoint_dir / "lora")
    
    # Save training state
    torch.save({
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics.to_dict(),
    }, checkpoint_dir / "training_state.pt")
    
    # Save config
    config.save(checkpoint_dir / "config.json")
    
    print(f"  ✓ Checkpoint saved: {checkpoint_dir}")
    return checkpoint_dir


def load_checkpoint(
    pipe,
    optimizer,
    scheduler,
    checkpoint_dir: str,
    device: torch.device,
):
    """Load a training checkpoint."""
    checkpoint_path = Path(checkpoint_dir)
    
    # Load LoRA weights
    lora_path = checkpoint_path / "lora"
    if lora_path.exists():
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
        pipe.unet.to(device)
    
    # Load training state
    state_path = checkpoint_path / "training_state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location=device)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        return state["epoch"], state.get("metrics", {})
    
    return 0, {}


def train(config: TrainingConfig, resume_from: str = None):
    """Main training function."""
    
    # Setup
    print("\n" + "=" * 70)
    print("LORA TRAINING FOR SDXL IMG2IMG")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Data directory:    {config.data_dir}")
    print(f"  Output directory:  {config.output_dir}")
    print(f"  Epochs:            {config.num_epochs}")
    print(f"  Learning rate:     {config.learning_rate}")
    print(f"  LoRA rank:         {config.lora_rank}")
    print(f"  Save every:        {config.save_every} epochs")
    print(f"  Validate every:    {config.validate_every} epochs")
    
    # Device setup
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    print(f"\nDevice: {device}")
    print(f"Dtype: {dtype}")
    
    # Load training data
    print("\nLoading training data...")
    train_pairs = load_image_pairs(config.data_dir, "train")
    val_pairs = load_image_pairs(config.data_dir, "val")
    
    print(f"  Train pairs: {len(train_pairs)}")
    print(f"  Val pairs:   {len(val_pairs)}")
    
    if len(train_pairs) == 0:
        print(f"\nError: No training data found in {config.data_dir}/train/")
        print("Run generate_training_data.py first!")
        return
    
    # Load model
    print(f"\nLoading SDXL from {config.model_id}...")
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    # Move components to device
    pipe.vae.to(device)
    if pipe.text_encoder:
        pipe.text_encoder.to(device)
    if pipe.text_encoder_2:
        pipe.text_encoder_2.to(device)
    
    # Configure LoRA
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",  # Attention
            "ff.net.0.proj", "ff.net.2",          # FFN (if available)
        ],
        lora_dropout=config.lora_dropout,
    )
    
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    pipe.unet.to(device)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in pipe.unet.parameters())
    print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"  Total params:     {total_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, pipe.unet.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=config.learning_rate / 100,
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        start_epoch, _ = load_checkpoint(pipe, optimizer, lr_scheduler, resume_from, device)
        print(f"  Resuming from epoch {start_epoch + 1}")
    
    # Encode prompt once
    print("\nEncoding prompt...")
    with torch.no_grad():
        prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
            prompt=config.prompt,
            device=device,
        )
    
    # Time IDs for SDXL
    add_time_ids = torch.tensor(
        [[1024, 1024, 0, 0, 1024, 1024]],
        device=device,
        dtype=dtype
    )
    
    # Setup logging
    logger = TrainingLogger(config.output_dir)
    
    # Save initial config
    config.save(Path(config.output_dir) / "config.json")
    
    # Training loop
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    best_val_loss = float("inf")
    start_time = time.time()
    
    for epoch in range(start_epoch, config.num_epochs):
        epoch_start = time.time()
        pipe.unet.train()
        
        # Shuffle training data
        random.shuffle(train_pairs)
        
        # Training epoch
        epoch_loss = 0
        pbar = tqdm(train_pairs, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for pair in pbar:
            loss = compute_loss(
                pipe, pair["original"], pair["compressed"],
                prompt_embeds, pooled_prompt_embeds, add_time_ids,
                device, dtype
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), config.gradient_clip)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Compute average training loss
        avg_train_loss = epoch_loss / len(train_pairs)
        
        # Validation
        val_loss = None
        if val_pairs and (epoch + 1) % config.validate_every == 0:
            print(f"  Running validation...")
            val_loss = validate(
                pipe, val_pairs,
                prompt_embeds, pooled_prompt_embeds, add_time_ids,
                device, dtype
            )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save best model
                save_checkpoint(
                    pipe, optimizer, lr_scheduler,
                    epoch + 1, config,
                    TrainingMetrics(
                        epoch=epoch + 1,
                        train_loss=avg_train_loss,
                        val_loss=val_loss,
                        learning_rate=lr_scheduler.get_last_lr()[0],
                        epoch_time=time.time() - epoch_start,
                        total_time=time.time() - start_time,
                        best_val_loss=best_val_loss,
                    ),
                    config.output_dir,
                    name="best"
                )
        
        # Update learning rate
        lr_scheduler.step()
        
        # Compute metrics
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        current_lr = lr_scheduler.get_last_lr()[0]
        
        metrics = TrainingMetrics(
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            val_loss=val_loss,
            learning_rate=current_lr,
            epoch_time=epoch_time,
            total_time=total_time,
            best_val_loss=best_val_loss if best_val_loss != float("inf") else avg_train_loss,
        )
        
        # Log and print
        logger.log(metrics)
        logger.print_epoch(metrics)
        
        # Save checkpoint
        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(
                pipe, optimizer, lr_scheduler,
                epoch + 1, config, metrics,
                config.output_dir
            )
    
    # Save final model
    print("\n" + "=" * 70)
    print("SAVING FINAL MODEL")
    print("=" * 70)
    
    final_metrics = TrainingMetrics(
        epoch=config.num_epochs,
        train_loss=avg_train_loss,
        val_loss=val_loss,
        learning_rate=lr_scheduler.get_last_lr()[0],
        epoch_time=epoch_time,
        total_time=time.time() - start_time,
        best_val_loss=best_val_loss if best_val_loss != float("inf") else avg_train_loss,
    )
    
    save_checkpoint(
        pipe, optimizer, lr_scheduler,
        config.num_epochs, config, final_metrics,
        config.output_dir,
        name="final"
    )
    
    # Save training summary
    logger.save_summary(config)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nFinal Results:")
    print(f"  Total epochs:      {config.num_epochs}")
    print(f"  Final train loss:  {avg_train_loss:.6f}")
    print(f"  Best val loss:     {best_val_loss:.6f}" if best_val_loss != float("inf") else "  Best val loss:     N/A")
    print(f"  Total time:        {(time.time() - start_time)/60:.1f} minutes")
    print(f"\nOutput files:")
    print(f"  {config.output_dir}/")
    print(f"  ├── best/              (best validation checkpoint)")
    print(f"  ├── final/             (final checkpoint)")
    print(f"  ├── checkpoint-*/      (intermediate checkpoints)")
    print(f"  ├── config.json        (training configuration)")
    print(f"  ├── training_log.jsonl (per-epoch metrics)")
    print(f"  └── training_summary.json (final summary)")
    print("\nTo use the trained LoRA:")
    print(f"  pipe.load_lora_weights('{config.output_dir}/best/lora')")


def main():
    parser = argparse.ArgumentParser(
        description="Train LoRA for SDXL Img2Img compression reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    uv run python scripts/train_lora.py --data training_data

    # Custom epochs and learning rate
    uv run python scripts/train_lora.py --data training_data --epochs 50 --lr 5e-5

    # Resume from checkpoint
    uv run python scripts/train_lora.py --data training_data --resume lora_output/checkpoint-20

    # Higher LoRA rank for more capacity
    uv run python scripts/train_lora.py --data training_data --rank 64
        """
    )
    
    parser.add_argument("--data", type=str, default="training_data",
                        help="Training data directory (default: training_data)")
    parser.add_argument("--output", type=str, default="lora_output",
                        help="Output directory (default: lora_output)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--rank", type=int, default=32,
                        help="LoRA rank (default: 32)")
    parser.add_argument("--save-every", type=int, default=20,
                        help="Save checkpoint every N epochs (default: 20)")
    parser.add_argument("--val-every", type=int, default=10,
                        help="Validate every N epochs (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint directory")
    parser.add_argument("--prompt", type=str, default="a high quality photograph",
                        help="Training prompt")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        data_dir=args.data,
        output_dir=args.output,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        lora_rank=args.rank,
        lora_alpha=args.rank,
        save_every=args.save_every,
        validate_every=args.val_every,
        seed=args.seed,
        prompt=args.prompt,
    )
    
    train(config, resume_from=args.resume)


if __name__ == "__main__":
    main()

