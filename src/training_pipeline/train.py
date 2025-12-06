"""
LoRA Training for SDXL Img2Img Compression Reconstruction.

This script trains a LoRA adapter to improve image reconstruction
from VAE-compressed inputs.

Usage:
    uv run python src/training_pipeline/train.py --data training_data
    
    # With custom settings:
    uv run python src/training_pipeline/train.py --data training_data --epochs 50 --lr 1e-5
    
    # Resume from checkpoint:
    uv run python src/training_pipeline/train.py --data training_data --resume lora_output/checkpoint-20
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

# Disable cuDNN to avoid segfaults in certain environments (gVisor, some container runtimes)
torch.backends.cudnn.enabled = False
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
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    seed: int = 42
    warmup_epochs: int = 5
    gradient_accumulation_steps: int = 8
    
    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    
    # Checkpointing
    save_every: int = 20
    validate_every: int = 5
    
    # Model
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    prompt: str = "a high quality photograph"
    
    # Timestep sampling - focus on lower timesteps for img2img
    num_timestep_buckets: int = 8
    max_timestep: int = 600  # Lower max since img2img uses lower strengths
    
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
    nan_count: int = 0
    
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
        if metrics.nan_count > 0:
            print(f"  NaN Skipped:     {metrics.nan_count}")
        print("=" * 70)
    
    def save_summary(self, config: TrainingConfig):
        """Save final training summary."""
        valid_losses = [m["train_loss"] for m in self.metrics_history if not math.isnan(m["train_loss"])]
        valid_val_losses = [m["val_loss"] for m in self.metrics_history if m["val_loss"] is not None and not math.isnan(m["val_loss"])]
        
        summary = {
            "config": asdict(config),
            "metrics_history": self.metrics_history,
            "final_train_loss": valid_losses[-1] if valid_losses else None,
            "best_val_loss": min(valid_val_losses) if valid_val_losses else None,
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


def get_stratified_timestep(
    step: int,
    num_buckets: int,
    max_timestep: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Get a timestep using stratified sampling for lower variance.
    
    Divides the timestep range into buckets and cycles through them,
    sampling randomly within each bucket.
    """
    bucket = step % num_buckets
    bucket_size = max_timestep // num_buckets
    t_min = bucket * bucket_size
    t_max = min((bucket + 1) * bucket_size, max_timestep)
    
    return torch.randint(t_min, t_max, (1,), device=device).long()


def compute_loss(
    pipe,
    original_path: str,
    compressed_path: str,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    add_time_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    timesteps: torch.Tensor,
    use_amp: bool = True,
) -> Tuple[torch.Tensor, bool]:
    """
    Compute training loss for compression reconstruction.
    
    This version uses BOTH original and compressed images:
    - Compressed image is the noisy input (what the model sees)
    - Original image is the target (what we want to reconstruct)
    
    The model learns to predict noise that would denoise compressed -> original.
    """
    
    # Load BOTH images
    original = Image.open(original_path).convert("RGB")
    compressed = Image.open(compressed_path).convert("RGB")
    
    # Preprocess both - use float32 for stable encoding
    original_tensor = pipe.image_processor.preprocess(original).to(device, torch.float32)
    compressed_tensor = pipe.image_processor.preprocess(compressed).to(device, torch.float32)
    
    with torch.no_grad():
        # Encode BOTH images to latent space
        original_latents = pipe.vae.encode(original_tensor).latent_dist.sample()
        original_latents = original_latents * pipe.vae.config.scaling_factor
        
        compressed_latents = pipe.vae.encode(compressed_tensor).latent_dist.sample()
        compressed_latents = compressed_latents * pipe.vae.config.scaling_factor
        
        # Check for NaN in latents
        if torch.isnan(original_latents).any() or torch.isnan(compressed_latents).any():
            print(f"  Warning: NaN in latents for {original_path}")
            return torch.tensor(0.0, device=device), False
        
        original_latents = original_latents.to(dtype)
        compressed_latents = compressed_latents.to(dtype)
    
    # Sample random noise
    noise = torch.randn_like(original_latents)
    
    # === KEY CHANGE: Add noise to COMPRESSED latents ===
    # The model sees: compressed image + noise
    noisy_latents = pipe.scheduler.add_noise(compressed_latents, noise, timesteps)
    noisy_latents = torch.clamp(noisy_latents, -30, 30)
    
    # === KEY CHANGE: Target noise should denoise toward ORIGINAL ===
    # If we were denoising original, target would just be `noise`
    # But we want to denoise compressed -> original, so we adjust the target
    #
    # The diffusion formula: noisy = sqrt(alpha) * clean + sqrt(1-alpha) * noise
    # We want the model to predict noise that would give us original, not compressed
    #
    # Rearranging: noise_target = (noisy - sqrt(alpha) * original) / sqrt(1-alpha)
    
    # Get alpha values for this timestep
    alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device)
    alpha_t = alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1).to(dtype)
    sigma_t = (1 - alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1).to(dtype)
    
    # Target: the noise that, when removed, gives us the ORIGINAL image
    # noisy_latents = alpha_t * compressed_latents + sigma_t * noise
    # We want prediction that satisfies: noisy_latents = alpha_t * original_latents + sigma_t * target
    # Therefore: target = (noisy_latents - alpha_t * original_latents) / sigma_t
    target = (noisy_latents - alpha_t * original_latents) / sigma_t.clamp(min=1e-8)
    
    # Forward pass with optional AMP
    if use_amp and device.type == "cuda":
        with autocast(device_type="cuda"):
            noise_pred = pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids,
                },
            ).sample
            
            # MSE loss between predicted noise and target
            loss = F.mse_loss(noise_pred.float(), target.float())
    else:
        noise_pred = pipe.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids,
            },
        ).sample
        
        loss = F.mse_loss(noise_pred.float(), target.float())
    
    # Check for NaN loss
    if torch.isnan(loss) or torch.isinf(loss):
        return loss, False
    
    # Log high-loss images for debugging
    if loss.item() > 10.0:
        print(f"\n  HIGH LOSS: {loss.item():.2f} - {original_path}")
    
    return loss, True


def validate(
    pipe,
    val_pairs: List[Dict],
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    add_time_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    config: TrainingConfig,
) -> float:
    """Compute validation loss."""
    pipe.unet.eval()
    total_loss = 0
    valid_count = 0
    
    with torch.no_grad():
        for step, pair in enumerate(val_pairs):
            # Use stratified timesteps for consistent validation
            timesteps = get_stratified_timestep(
                step, config.num_timestep_buckets, config.max_timestep, device
            )
            
            loss, is_valid = compute_loss(
                pipe, pair["original"], pair["compressed"],
                prompt_embeds, pooled_prompt_embeds, add_time_ids,
                device, dtype, timesteps, use_amp=False
            )
            if is_valid:
                total_loss += loss.item()
                valid_count += 1
    
    pipe.unet.train()
    return total_loss / valid_count if valid_count > 0 else float("nan")


def save_checkpoint(
    pipe,
    optimizer,
    scheduler,
    scaler,
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
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "metrics": metrics.to_dict(),
    }, checkpoint_dir / "training_state.pt")
    
    # Save config
    config.save(checkpoint_dir / "config.json")
    
    print(f"  ✓ Checkpoint saved: {checkpoint_dir}")
    return checkpoint_dir


def get_lr_with_warmup(epoch: int, warmup_epochs: int, base_lr: float, total_epochs: int) -> float:
    """Calculate learning rate with warmup."""
    if epoch < warmup_epochs:
        # Linear warmup
        return (epoch + 1) / warmup_epochs
    else:
        # Cosine decay after warmup
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))


def train(config: TrainingConfig, resume_from: str = None):
    """Main training function."""
    
    # Setup
    print("\n" + "=" * 70)
    print("LORA TRAINING FOR SDXL IMG2IMG - RECONSTRUCTION OBJECTIVE")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Data directory:           {config.data_dir}")
    print(f"  Output directory:         {config.output_dir}")
    print(f"  Epochs:                   {config.num_epochs}")
    print(f"  Learning rate:            {config.learning_rate}")
    print(f"  Warmup epochs:            {config.warmup_epochs}")
    print(f"  Gradient accumulation:    {config.gradient_accumulation_steps}")
    print(f"  Effective batch size:     {config.gradient_accumulation_steps}")
    print(f"  LoRA rank:                {config.lora_rank}")
    print(f"  LoRA alpha:               {config.lora_alpha}")
    print(f"  Timestep buckets:         {config.num_timestep_buckets}")
    print(f"  Max timestep:             {config.max_timestep}")
    print(f"  Save every:               {config.save_every} epochs")
    print(f"  Validate every:           {config.validate_every} epochs")
    print(f"\n  ** Using RECONSTRUCTION objective (compressed -> original) **")
    
    # Device setup
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    use_amp = device.type == "cuda"
    
    print(f"\nDevice: {device}")
    print(f"Dtype: {dtype}")
    print(f"Mixed Precision: {use_amp}")
    
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
    
    # Move VAE to float32 for stable encoding
    pipe.vae.to(device, dtype=torch.float32)
    if pipe.text_encoder:
        pipe.text_encoder.to(device)
    if pipe.text_encoder_2:
        pipe.text_encoder_2.to(device)
    
    # Gradient scaler for mixed precision
    scaler = GradScaler() if use_amp else None
    
    # === FIXED: Handle resume vs fresh start separately ===
    start_epoch = 0
    
    if resume_from:
        # RESUMING: Load LoRA from checkpoint (don't wrap with get_peft_model)
        print(f"\nResuming from checkpoint: {resume_from}")
        
        lora_path = Path(resume_from) / "lora"
        if lora_path.exists():
            pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
            pipe.unet.to(device)
        else:
            raise FileNotFoundError(f"LoRA weights not found at {lora_path}")
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in pipe.unet.parameters())
        print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"  Total params:     {total_params:,}")
        
        # Create optimizer AFTER loading model
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, pipe.unet.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=1e-8,
        )
        
        # Create LR scheduler
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: get_lr_with_warmup(
                epoch, config.warmup_epochs, 1.0, config.num_epochs
            )
        )
        
        # Load training state
        state_path = Path(resume_from) / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=device)
            start_epoch = state["epoch"]
            optimizer.load_state_dict(state["optimizer_state_dict"])
            lr_scheduler.load_state_dict(state["scheduler_state_dict"])
            if scaler and state.get("scaler_state_dict"):
                scaler.load_state_dict(state["scaler_state_dict"])
            print(f"  Loaded training state from epoch {start_epoch}")
        
        print(f"  Resuming from epoch {start_epoch + 1}")
        
    else:
        # FRESH START: Configure LoRA from scratch
        print("\nConfiguring LoRA...")
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=[
                "to_q", "to_k", "to_v", "to_out.0",  # Attention
                "ff.net.0.proj", "ff.net.2",  # Feedforward layers
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
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, pipe.unet.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=1e-8,
        )
        
        # Create LR scheduler
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: get_lr_with_warmup(
                epoch, config.warmup_epochs, 1.0, config.num_epochs
            )
        )
    
    # Encode prompt once
    print("\nEncoding prompt...")
    with torch.no_grad():
        prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
            prompt=config.prompt,
            device=device,
        )
        prompt_embeds = prompt_embeds.to(dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype)
    
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
    consecutive_nan = 0
    max_consecutive_nan = 10
    
    for epoch in range(start_epoch, config.num_epochs):
        epoch_start = time.time()
        pipe.unet.train()
        
        # Shuffle training data
        random.shuffle(train_pairs)
        
        # Training epoch with gradient accumulation
        epoch_loss = 0
        valid_batches = 0
        nan_count = 0
        accumulated_loss = 0
        accumulation_step = 0
        
        # Zero gradients at start of epoch
        optimizer.zero_grad()
        
        pbar = tqdm(train_pairs, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for step, pair in enumerate(pbar):
            # Get stratified timestep for lower variance
            timesteps = get_stratified_timestep(
                step, config.num_timestep_buckets, config.max_timestep, device
            )
            
            # Compute loss
            loss, is_valid = compute_loss(
                pipe, pair["original"], pair["compressed"],
                prompt_embeds, pooled_prompt_embeds, add_time_ids,
                device, dtype, timesteps, use_amp=use_amp
            )
            
            if not is_valid:
                nan_count += 1
                consecutive_nan += 1
                pbar.set_postfix({"loss": "NaN", "skipped": nan_count})
                
                if consecutive_nan >= max_consecutive_nan:
                    print(f"\n\nERROR: {max_consecutive_nan} consecutive NaN losses!")
                    print("Training is unstable. Try:")
                    print("  1. Lower learning rate: --lr 5e-6")
                    print("  2. Check your training images for corruption")
                    return
                continue
            
            consecutive_nan = 0
            
            # Scale loss by accumulation steps
            scaled_loss = loss / config.gradient_accumulation_steps
            
            # Backward pass (accumulate gradients)
            if scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            
            accumulated_loss += loss.item()
            accumulation_step += 1
            
            # Step optimizer every gradient_accumulation_steps
            if accumulation_step >= config.gradient_accumulation_steps:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), config.gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), config.gradient_clip)
                    optimizer.step()
                
                optimizer.zero_grad()
                
                # Track loss for this accumulated batch
                avg_accumulated_loss = accumulated_loss / accumulation_step
                epoch_loss += avg_accumulated_loss
                valid_batches += 1
                
                pbar.set_postfix({
                    "loss": f"{avg_accumulated_loss:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # Reset accumulation
                accumulated_loss = 0
                accumulation_step = 0
        
        # Handle remaining accumulated gradients at end of epoch
        if accumulation_step > 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), config.gradient_clip)
                optimizer.step()
            
            optimizer.zero_grad()
            
            avg_accumulated_loss = accumulated_loss / accumulation_step
            epoch_loss += avg_accumulated_loss
            valid_batches += 1
        
        # Compute average training loss
        avg_train_loss = epoch_loss / valid_batches if valid_batches > 0 else float("nan")
        
        # Validation
        val_loss = None
        if val_pairs and (epoch + 1) % config.validate_every == 0:
            print(f"  Running validation...")
            val_loss = validate(
                pipe, val_pairs,
                prompt_embeds, pooled_prompt_embeds, add_time_ids,
                device, dtype, config
            )
            
            if not math.isnan(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save best model
                save_checkpoint(
                    pipe, optimizer, lr_scheduler, scaler,
                    epoch + 1, config,
                    TrainingMetrics(
                        epoch=epoch + 1,
                        train_loss=avg_train_loss,
                        val_loss=val_loss,
                        learning_rate=optimizer.param_groups[0]["lr"],
                        epoch_time=time.time() - epoch_start,
                        total_time=time.time() - start_time,
                        best_val_loss=best_val_loss,
                        nan_count=nan_count,
                    ),
                    config.output_dir,
                    name="best"
                )
        
        # Update learning rate
        lr_scheduler.step()
        
        # Compute metrics
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]
        
        metrics = TrainingMetrics(
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            val_loss=val_loss,
            learning_rate=current_lr,
            epoch_time=epoch_time,
            total_time=total_time,
            best_val_loss=best_val_loss if best_val_loss != float("inf") else avg_train_loss,
            nan_count=nan_count,
        )
        
        # Log and print
        logger.log(metrics)
        logger.print_epoch(metrics)
        
        # Save checkpoint
        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(
                pipe, optimizer, lr_scheduler, scaler,
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
        learning_rate=optimizer.param_groups[0]["lr"],
        epoch_time=epoch_time,
        total_time=time.time() - start_time,
        best_val_loss=best_val_loss if best_val_loss != float("inf") else avg_train_loss,
        nan_count=nan_count,
    )
    
    save_checkpoint(
        pipe, optimizer, lr_scheduler, scaler,
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
    if best_val_loss != float("inf"):
        print(f"  Best val loss:     {best_val_loss:.6f}")
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
    # Basic training (uses stable defaults)
    uv run python src/training_pipeline/train.py --data training_data

    # Custom epochs and learning rate
    uv run python src/training_pipeline/train.py --data training_data --epochs 100 --lr 1e-5

    # Resume from checkpoint
    uv run python src/training_pipeline/train.py --data training_data --resume lora_output/checkpoint-20

    # Higher LoRA rank for more capacity
    uv run python src/training_pipeline/train.py --data training_data --rank 64 --alpha 128
        """
    )
    
    parser.add_argument("--data", type=str, default="training_data",
                        help="Training data directory (default: training_data)")
    parser.add_argument("--output", type=str, default="lora_output",
                        help="Output directory (default: lora_output)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (default: 1e-5)")
    parser.add_argument("--rank", type=int, default=32,
                        help="LoRA rank (default: 32)")
    parser.add_argument("--alpha", type=int, default=64,
                        help="LoRA alpha (default: 64)")
    parser.add_argument("--accum", type=int, default=8,
                        help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--save-every", type=int, default=20,
                        help="Save checkpoint every N epochs (default: 20)")
    parser.add_argument("--val-every", type=int, default=5,
                        help="Validate every N epochs (default: 5)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup epochs (default: 5)")
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
        lora_alpha=args.alpha,
        gradient_accumulation_steps=args.accum,
        save_every=args.save_every,
        validate_every=args.val_every,
        warmup_epochs=args.warmup,
        seed=args.seed,
        prompt=args.prompt,
    )
    
    train(config, resume_from=args.resume)


if __name__ == "__main__":
    main()