"""
ILVR Parameter Sweep - Find optimal settings for closest resemblance

Based on the ILVR paper:
- Lower N (down_factor) = more similar to reference
- N=4 or N=2 should give very close resemblance
- Reducing guidance_scale reduces prompt influence
"""

import argparse
import os
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from diffusers import StableDiffusionXLPipeline, DDIMScheduler


def lowpass_phi(x: torch.Tensor, down_factor: int) -> torch.Tensor:
    if down_factor <= 1:
        return x.clone()
    _, _, h, w = x.shape
    h_ds, w_ds = max(1, h // down_factor), max(1, w // down_factor)
    x_down = F.interpolate(x, size=(h_ds, w_ds), mode="bicubic", align_corners=False)
    x_up = F.interpolate(x_down, size=(h, w), mode="bicubic", align_corners=False)
    return x_up


def encode_image_to_latent_safe(pipe, image: Image.Image, device, dtype) -> torch.Tensor:
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


class ILVRCallback:
    def __init__(self, y0_latent, noise, scheduler, down_factor=8, strength=1.0):
        self.y0_latent = y0_latent
        self.noise = noise
        self.scheduler = scheduler
        self.down_factor = down_factor
        self.strength = strength
        
    def __call__(self, pipe, step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        
        t = timestep.reshape(-1).to(self.y0_latent.device)
        y_t = self.scheduler.add_noise(self.y0_latent, self.noise, t)
        
        batch_size = latents.shape[0]
        if y_t.shape[0] != batch_size:
            y_t = y_t.expand(batch_size, -1, -1, -1)
        
        phi_y = lowpass_phi(y_t, self.down_factor)
        phi_x = lowpass_phi(latents, self.down_factor)
        
        if self.strength >= 1.0:
            latents_new = phi_y + (latents - phi_x)
        else:
            latents_new = latents + self.strength * (phi_y - phi_x)
        
        latents_new = torch.clamp(latents_new, -30, 30)
        callback_kwargs["latents"] = latents_new
        return callback_kwargs


def run_single(
    pipe,
    prompt,
    negative_prompt,
    y0_latent,
    ref_noise,
    down_factor,
    guidance_scale,
    num_steps,
    height,
    width,
    seed,
    device,
    dtype,
):
    """Run a single generation with given parameters."""
    pipe.scheduler.set_timesteps(num_steps, device=device)
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    callback = ILVRCallback(
        y0_latent=y0_latent,
        noise=ref_noise,
        scheduler=pipe.scheduler,
        down_factor=down_factor,
        strength=1.0,
    )
    
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    
    return result.images[0]


def run_sweep(
    ref_path: str,
    prompt: str = "a high quality photograph, detailed face",
    negative_prompt: str = "blurry, low quality, distorted, deformed",
    output_dir: str = "ilvr_sweep",
    height: int = 1024,
    width: int = 1024,
    seed: int = 42,
    num_steps: int = 50,
):
    """Run parameter sweep to find best settings."""
    
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
    
    # Load and encode reference
    print(f"Loading reference: {ref_path}")
    ref_image = Image.open(ref_path).convert("RGB")
    ref_image = ref_image.resize((width, height), Image.LANCZOS)
    
    print("Encoding reference...")
    y0_latent = encode_image_to_latent_safe(pipe, ref_image, device, dtype)
    
    # Fixed noise for fair comparison
    generator = torch.Generator(device=device).manual_seed(seed)
    ref_noise = torch.randn(y0_latent.shape, device=device, dtype=dtype, generator=generator)
    
    # Parameter combinations to try
    # Lower N = more similar to reference
    down_factors = [2, 4, 8, 16]
    guidance_scales = [3.0, 5.0, 7.5]
    
    results = []
    
    for N in down_factors:
        for cfg in guidance_scales:
            print(f"\n{'='*50}")
            print(f"N={N}, guidance={cfg}")
            print(f"{'='*50}")
            
            output = run_single(
                pipe=pipe,
                prompt=prompt,
                negative_prompt=negative_prompt,
                y0_latent=y0_latent,
                ref_noise=ref_noise,
                down_factor=N,
                guidance_scale=cfg,
                num_steps=num_steps,
                height=height,
                width=width,
                seed=seed,
                device=device,
                dtype=dtype,
            )
            
            # Save individual result
            filename = f"N{N}_cfg{cfg}.png"
            output.save(os.path.join(output_dir, filename))
            
            results.append({
                "N": N,
                "cfg": cfg,
                "image": output,
                "filename": filename,
            })
            
            print(f"Saved: {filename}")
    
    # Create comparison grid
    print("\nCreating comparison grid...")
    
    n_cols = len(guidance_scales) + 1  # +1 for reference column
    n_rows = len(down_factors)
    
    cell_w, cell_h = 512, 512  # Smaller for grid
    label_h = 40
    
    grid_w = cell_w * n_cols
    grid_h = (cell_h + label_h) * n_rows + label_h  # Extra row for header
    
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)
    
    # Header row
    draw.text((10, 10), "Reference", fill="black")
    for i, cfg in enumerate(guidance_scales):
        x = (i + 1) * cell_w + 10
        draw.text((x, 10), f"CFG={cfg}", fill="black")
    
    # Fill grid
    ref_small = ref_image.resize((cell_w, cell_h), Image.LANCZOS)
    
    for row, N in enumerate(down_factors):
        y = label_h + row * (cell_h + label_h)
        
        # Reference in first column
        grid.paste(ref_small, (0, y + label_h))
        draw.text((10, y + 5), f"N={N}", fill="black")
        
        # Results
        for col, cfg in enumerate(guidance_scales):
            x = (col + 1) * cell_w
            
            # Find matching result
            for r in results:
                if r["N"] == N and r["cfg"] == cfg:
                    img_small = r["image"].resize((cell_w, cell_h), Image.LANCZOS)
                    grid.paste(img_small, (x, y + label_h))
                    break
    
    grid_path = os.path.join(output_dir, "comparison_grid.png")
    grid.save(grid_path)
    print(f"Saved grid: {grid_path}")
    
    # Also save just the "best" settings comparison (N=2, N=4)
    print("\nCreating best-settings comparison...")
    best_comparison = Image.new("RGB", (width * 3, height), "white")
    best_comparison.paste(ref_image, (0, 0))
    
    # Find N=4, cfg=5.0 result
    for r in results:
        if r["N"] == 4 and r["cfg"] == 5.0:
            best_comparison.paste(r["image"], (width, 0))
    
    # Find N=2, cfg=5.0 result  
    for r in results:
        if r["N"] == 2 and r["cfg"] == 5.0:
            best_comparison.paste(r["image"], (width * 2, 0))
    
    best_path = os.path.join(output_dir, "best_comparison.png")
    best_comparison.save(best_path)
    print(f"Saved best comparison: {best_path}")
    
    print(f"\n✓ All results saved to {output_dir}/")
    print("\nRecommendations for closest resemblance:")
    print("  - N=2 or N=4: Maximum detail preservation")
    print("  - CFG=3.0-5.0: Less prompt influence, more reference fidelity")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str, required=True, help="Reference image")
    parser.add_argument("--prompt", type=str, default="a high quality photograph, detailed face")
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, distorted, deformed")
    parser.add_argument("--output_dir", type=str, default="ilvr_sweep")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    
    args = parser.parse_args()
    
    run_sweep(
        ref_path=args.ref,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        output_dir=args.output_dir,
        height=args.height,
        width=args.width,
        seed=args.seed,
        num_steps=args.steps,
    )


if __name__ == "__main__":
    main()