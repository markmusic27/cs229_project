"""
ILVR for SDXL - v3 with RGB noise fixes

Fixes for the colorful noise artifacts:
1. More diffusion steps (gives model more time to denoise properly)
2. Stop ILVR earlier (let final steps clean up without interference)
3. Optional: Apply ILVR less frequently (every N steps instead of every step)
"""

import argparse
import os
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from diffusers import StableDiffusionXLPipeline, DDIMScheduler


def check_tensor(t: torch.Tensor, name: str) -> bool:
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    if has_nan or has_inf:
        print(f"  ⚠️  {name}: NaN={has_nan}, Inf={has_inf}")
        return False
    print(f"  ✓ {name}: min={t.min().item():.3f}, max={t.max().item():.3f}, mean={t.mean().item():.3f}")
    return True


def lowpass_phi(x: torch.Tensor, down_factor: int) -> torch.Tensor:
    """Low-pass filter via downsample/upsample."""
    if down_factor <= 1:
        return x.clone()

    _, _, h, w = x.shape
    h_ds = max(1, h // down_factor)
    w_ds = max(1, w // down_factor)

    x_down = F.interpolate(x, size=(h_ds, w_ds), mode="bicubic", align_corners=False)
    x_up = F.interpolate(x_down, size=(h, w), mode="bicubic", align_corners=False)
    return x_up


def encode_image_to_latent_safe(pipe, image: Image.Image, device, dtype) -> torch.Tensor:
    """Encode PIL image to VAE latent space in fp32 for stability."""
    image_tensor = pipe.image_processor.preprocess(image)
    image_tensor_fp32 = image_tensor.to(device=device, dtype=torch.float32)
    
    with torch.no_grad():
        vae_dtype = pipe.vae.dtype
        pipe.vae.to(dtype=torch.float32)
        posterior = pipe.vae.encode(image_tensor_fp32)
        latents = posterior.latent_dist.mean * pipe.vae.config.scaling_factor
        pipe.vae.to(dtype=vae_dtype)
        latents = latents.to(dtype=dtype)
    
    return latents


class ILVRCallbackV3:
    """
    ILVR callback with noise reduction strategies.
    
    Key changes:
    - end_ratio: Stop ILVR before the final steps to let diffusion clean up
    - apply_every: Only apply ILVR every N steps (reduces interference)
    """
    
    def __init__(
        self,
        y0_latent: torch.Tensor,
        noise: torch.Tensor,
        scheduler,
        down_factor: int = 4,
        strength: float = 1.0,
        start_ratio: float = 0.0,   # Start at this fraction of total steps
        end_ratio: float = 0.8,     # Stop at this fraction (let last 20% clean up)
        apply_every: int = 1,       # Apply every N steps
        debug: bool = False,
    ):
        self.y0_latent = y0_latent
        self.noise = noise
        self.scheduler = scheduler
        self.down_factor = down_factor
        self.strength = strength
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.apply_every = apply_every
        self.debug = debug
        
        if torch.isnan(y0_latent).any():
            raise ValueError("y0_latent contains NaN!")
        
    def __call__(self, pipe, step_index: int, timestep: torch.Tensor, callback_kwargs: dict) -> dict:
        latents = callback_kwargs["latents"]
        num_steps = self.scheduler.num_inference_steps
        
        # Calculate step ranges
        start_step = int(num_steps * self.start_ratio)
        end_step = int(num_steps * self.end_ratio)
        
        # Decide whether to apply ILVR
        in_range = start_step <= step_index < end_step
        on_cycle = (step_index % self.apply_every) == 0
        apply_ilvr = in_range and on_cycle
        
        if self.debug and step_index % 10 == 0:
            status = "ILVR" if apply_ilvr else "skip"
            print(f"Step {step_index}/{num_steps}, t={timestep.item():.0f} [{status}]")
        
        if apply_ilvr:
            t = timestep.reshape(-1).to(self.y0_latent.device)
            
            # Noised reference
            y_t = self.scheduler.add_noise(self.y0_latent, self.noise, t)
            
            # Handle CFG batch
            if y_t.shape[0] != latents.shape[0]:
                y_t = y_t.expand(latents.shape[0], -1, -1, -1)
            
            # Low-pass filter
            phi_y = lowpass_phi(y_t, self.down_factor)
            phi_x = lowpass_phi(latents, self.down_factor)
            
            # ILVR update
            if self.strength >= 1.0:
                latents_new = phi_y + (latents - phi_x)
            else:
                latents_new = latents + self.strength * (phi_y - phi_x)
            
            # Safety
            latents_new = torch.clamp(latents_new, -30, 30)
            latents_new = torch.nan_to_num(latents_new, nan=0.0)
            
            callback_kwargs["latents"] = latents_new
        
        return callback_kwargs


def run_ilvr_v3(
    prompt: str,
    ref_path: str,
    output_path: str = "ilvr_output.png",
    negative_prompt: str = "blurry, low quality, distorted, noise, artifacts",
    num_inference_steps: int = 100,  # More steps!
    guidance_scale: float = 3.0,
    down_factor: int = 4,
    strength: float = 1.0,
    end_ratio: float = 0.8,  # Stop ILVR at 80% to let final steps clean up
    apply_every: int = 1,
    height: int = 1024,
    width: int = 1024,
    seed: int = 42,
    debug: bool = True,
):
    """Run SDXL with improved ILVR."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    print(f"Device: {device}, dtype: {dtype}")
    print("Loading SDXL...")
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        use_safetensors=True,
    ).to(device)
    
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass
    
    # Load reference
    print(f"\nReference: {ref_path}")
    ref_image = Image.open(ref_path).convert("RGB")
    ref_image = ref_image.resize((width, height), Image.LANCZOS)
    
    # Encode reference
    print("Encoding reference...")
    y0_latent = encode_image_to_latent_safe(pipe, ref_image, device, dtype)
    print(f"  Latent range: [{y0_latent.min():.2f}, {y0_latent.max():.2f}]")
    
    # Setup scheduler
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    
    # Create noise
    generator = torch.Generator(device=device).manual_seed(seed)
    ref_noise = torch.randn(y0_latent.shape, device=device, dtype=dtype, generator=generator)
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Create callback
    callback = ILVRCallbackV3(
        y0_latent=y0_latent,
        noise=ref_noise,
        scheduler=pipe.scheduler,
        down_factor=down_factor,
        strength=strength,
        end_ratio=end_ratio,
        apply_every=apply_every,
        debug=debug,
    )
    
    print(f"\n{'='*50}")
    print(f"ILVR v3 Generation")
    print(f"  steps: {num_inference_steps}")
    print(f"  down_factor: {down_factor}")
    print(f"  guidance: {guidance_scale}")
    print(f"  end_ratio: {end_ratio} (ILVR stops at step {int(num_inference_steps * end_ratio)})")
    print(f"  apply_every: {apply_every}")
    print(f"{'='*50}\n")
    
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    
    output_image = result.images[0]
    
    # Save
    output_image.save(output_path)
    print(f"\nSaved: {output_path}")
    
    # Save comparison
    comparison = Image.new("RGB", (width * 2, height))
    comparison.paste(ref_image, (0, 0))
    comparison.paste(output_image, (width, 0))
    comp_path = output_path.replace(".png", "_comparison.png")
    comparison.save(comp_path)
    print(f"Saved: {comp_path}")
    
    return output_image


def run_ablation(ref_path: str, output_dir: str = "ilvr_v3_ablation", seed: int = 42):
    """Test different end_ratio values to find best setting."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    print("Loading SDXL...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        use_safetensors=True,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass
    
    # Load reference
    ref_image = Image.open(ref_path).convert("RGB")
    ref_image = ref_image.resize((1024, 1024), Image.LANCZOS)
    y0_latent = encode_image_to_latent_safe(pipe, ref_image, device, dtype)
    
    # Test configurations
    configs = [
        # (end_ratio, steps, down_factor, guidance)
        (1.0, 100, 4, 3.0),   # Original: ILVR all steps
        (0.9, 100, 4, 3.0),   # Stop at 90%
        (0.8, 100, 4, 3.0),   # Stop at 80%
        (0.7, 100, 4, 3.0),   # Stop at 70%
        (0.6, 100, 4, 3.0),   # Stop at 60%
        (0.8, 100, 2, 3.0),   # N=2, stop at 80%
    ]
    
    results = []
    
    for end_ratio, steps, N, cfg in configs:
        print(f"\n{'='*50}")
        print(f"end_ratio={end_ratio}, steps={steps}, N={N}, cfg={cfg}")
        print(f"{'='*50}")
        
        pipe.scheduler.set_timesteps(steps, device=device)
        
        generator = torch.Generator(device=device).manual_seed(seed)
        ref_noise = torch.randn(y0_latent.shape, device=device, dtype=dtype, generator=generator)
        generator = torch.Generator(device=device).manual_seed(seed)
        
        callback = ILVRCallbackV3(
            y0_latent=y0_latent,
            noise=ref_noise,
            scheduler=pipe.scheduler,
            down_factor=N,
            end_ratio=end_ratio,
            debug=False,
        )
        
        result = pipe(
            prompt="a high quality photograph",
            negative_prompt="blurry, low quality, distorted, noise",
            num_inference_steps=steps,
            guidance_scale=cfg,
            height=1024,
            width=1024,
            generator=generator,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )
        
        output = result.images[0]
        filename = f"end{end_ratio}_N{N}_steps{steps}_cfg{cfg}.png"
        output.save(os.path.join(output_dir, filename))
        results.append((f"end={end_ratio} N={N}", output))
        print(f"Saved: {filename}")
    
    # Create comparison grid
    n_cols = 3
    n_rows = (len(results) + 1 + n_cols - 1) // n_cols  # +1 for reference
    cell_size = 512
    label_h = 30
    
    grid = Image.new("RGB", (cell_size * n_cols, (cell_size + label_h) * n_rows), "white")
    
    from PIL import ImageDraw
    draw = ImageDraw.Draw(grid)
    
    # Add reference first
    all_items = [("Reference", ref_image)] + results
    
    for i, (label, img) in enumerate(all_items):
        row = i // n_cols
        col = i % n_cols
        x = col * cell_size
        y = row * (cell_size + label_h)
        
        img_small = img.resize((cell_size, cell_size), Image.LANCZOS)
        grid.paste(img_small, (x, y + label_h))
        draw.text((x + 5, y + 5), label, fill="black")
    
    grid.save(os.path.join(output_dir, "comparison_grid.png"))
    print(f"\nSaved comparison grid to {output_dir}/comparison_grid.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="a high quality photograph")
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, distorted, noise, artifacts")
    parser.add_argument("--output", type=str, default="ilvr_v3_output.png")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--down_factor", type=int, default=4)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--end_ratio", type=float, default=0.8, 
                        help="Stop ILVR at this fraction of steps (0.8 = stop at 80%%)")
    parser.add_argument("--apply_every", type=int, default=1,
                        help="Apply ILVR every N steps")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_debug", action="store_true")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--output_dir", type=str, default="ilvr_v3_ablation")
    
    args = parser.parse_args()
    
    if args.ablation:
        run_ablation(args.ref, args.output_dir, args.seed)
    else:
        run_ilvr_v3(
            prompt=args.prompt,
            ref_path=args.ref,
            output_path=args.output,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            down_factor=args.down_factor,
            strength=args.strength,
            end_ratio=args.end_ratio,
            apply_every=args.apply_every,
            height=args.height,
            width=args.width,
            seed=args.seed,
            debug=not args.no_debug,
        )


if __name__ == "__main__":
    main()