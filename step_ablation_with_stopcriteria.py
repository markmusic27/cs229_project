"""
ILVR Sweep - Test different inference steps with fixed stop_step

Instead of end_ratio, uses an absolute stop_step value.
ILVR stops after this many steps regardless of total steps.
"""

import argparse
import os
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
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
    def __init__(self, y0_latent, noise, scheduler, down_factor=4, stop_step=100):
        """
        Args:
            stop_step: Absolute step number to stop ILVR (not a ratio).
                       ILVR applies on steps 0 to stop_step-1.
        """
        self.y0_latent = y0_latent
        self.noise = noise
        self.scheduler = scheduler
        self.down_factor = down_factor
        self.stop_step = stop_step
        
    def __call__(self, pipe, step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        
        # Apply ILVR only if we haven't reached stop_step
        if step_index < self.stop_step:
            t = timestep.reshape(-1).to(self.y0_latent.device)
            y_t = self.scheduler.add_noise(self.y0_latent, self.noise, t)
            
            if y_t.shape[0] != latents.shape[0]:
                y_t = y_t.expand(latents.shape[0], -1, -1, -1)
            
            phi_y = lowpass_phi(y_t, self.down_factor)
            phi_x = lowpass_phi(latents, self.down_factor)
            latents_new = phi_y + (latents - phi_x)
            latents_new = torch.clamp(latents_new, -30, 30)
            callback_kwargs["latents"] = latents_new
        
        return callback_kwargs


def run_sweep(
    ref_path: str,
    output_dir: str = "ilvr_steps_sweep",
    prompt: str = "a high quality photograph",
    negative_prompt: str = "blurry, low quality, distorted, noise, artifacts",
    down_factor: int = 4,
    guidance_scale: float = 3.0,
    height: int = 1024,
    width: int = 1024,
    seed: int = 42,
    steps_list: List[int] = None,
    stop_step: int = 100,
):
    """
    Run sweep over different step counts with fixed stop_step.
    
    Args:
        stop_step: ILVR stops after this many steps (absolute, not ratio)
    """
    
    if steps_list is None:
        steps_list = [100, 150, 200, 250, 300]
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    print(f"Device: {device}")
    print(f"Steps to test: {steps_list}")
    print(f"Stop step (fixed): {stop_step}")
    print(f"Total experiments: {len(steps_list)}")
    
    print("\nLoading SDXL...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        use_safetensors=True,
    ).to(device)
    
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("Enabled xformers")
    except:
        pass
    
    # Load and encode reference
    print(f"\nLoading reference: {ref_path}")
    ref_image = Image.open(ref_path).convert("RGB")
    ref_image = ref_image.resize((width, height), Image.LANCZOS)
    
    print("Encoding reference...")
    y0_latent = encode_image_to_latent_safe(pipe, ref_image, device, dtype)
    
    # Store results
    results = {}  # steps -> image
    
    for i, steps in enumerate(steps_list):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(steps_list)}] Steps={steps}, stop_step={stop_step}")
        print(f"  ILVR active: steps 0-{min(stop_step, steps)-1}")
        print(f"  Cleanup steps: {stop_step}-{steps-1}" if stop_step < steps else "  No cleanup (stop_step >= steps)")
        print(f"{'='*60}")
        
        # Setup scheduler
        pipe.scheduler.set_timesteps(steps, device=device)
        
        # Create noise (same seed for fair comparison)
        generator = torch.Generator(device=device).manual_seed(seed)
        ref_noise = torch.randn(y0_latent.shape, device=device, dtype=dtype, generator=generator)
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Create callback with absolute stop_step
        callback = ILVRCallback(
            y0_latent=y0_latent,
            noise=ref_noise,
            scheduler=pipe.scheduler,
            down_factor=down_factor,
            stop_step=stop_step,
        )
        
        # Generate
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )
        
        output = result.images[0]
        
        # Save individual result
        filename = f"steps{steps}_stop{stop_step}.png"
        output.save(os.path.join(output_dir, filename))
        print(f"Saved: {filename}")
        
        results[steps] = output
    
    # Create comparison grid
    print("\n" + "="*60)
    print("Creating comparison grid...")
    print("="*60)
    
    n_images = len(steps_list) + 1  # +1 for reference
    n_cols = min(4, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    cell_size = 450
    label_h = 40
    
    grid_w = cell_size * n_cols
    grid_h = (cell_size + label_h) * n_rows
    
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)
    
    # All images: reference + results
    all_images = [("Reference", ref_image)] + [(f"steps={s}, stop={stop_step}", results[s]) for s in steps_list]
    
    for idx, (label, img) in enumerate(all_images):
        row = idx // n_cols
        col = idx % n_cols
        x = col * cell_size
        y = row * (cell_size + label_h)
        
        img_small = img.resize((cell_size, cell_size), Image.LANCZOS)
        grid.paste(img_small, (x, y + label_h))
        draw.text((x + 10, y + 10), label, fill="black")
    
    grid_path = os.path.join(output_dir, f"comparison_stop{stop_step}.png")
    grid.save(grid_path)
    print(f"\nSaved comparison grid: {grid_path}")
    
    print(f"\n✓ All results saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep ILVR with fixed stop_step across different total steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test steps 100-300 with ILVR stopping at step 100
  python ilvr_stop_sweep.py --ref image.jpg --stop_step 100 --steps 100 150 200 250 300
  
  # Test with ILVR stopping at step 50
  python ilvr_stop_sweep.py --ref image.jpg --stop_step 50 --steps 75 100 150 200
  
  # Test with ILVR stopping at step 80
  python ilvr_stop_sweep.py --ref image.jpg --stop_step 80 --steps 100 150 200
        """
    )
    
    parser.add_argument("--ref", type=str, required=True, help="Reference image path")
    parser.add_argument("--output_dir", type=str, default="ilvr_stop_sweep", help="Output directory")
    parser.add_argument("--prompt", type=str, default="a high quality photograph")
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, distorted, noise, artifacts")
    parser.add_argument("--down_factor", type=int, default=4, help="ILVR downsampling factor")
    parser.add_argument("--guidance", type=float, default=3.0, help="CFG scale")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    
    # Key parameters
    parser.add_argument("--steps", type=int, nargs="+", default=[100, 150, 200, 250, 300],
                        help="List of total step counts to test")
    parser.add_argument("--stop_step", type=int, default=100,
                        help="Absolute step to stop ILVR (ILVR runs on steps 0 to stop_step-1)")
    
    args = parser.parse_args()
    
    run_sweep(
        ref_path=args.ref,
        output_dir=args.output_dir,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        down_factor=args.down_factor,
        guidance_scale=args.guidance,
        height=args.height,
        width=args.width,
        seed=args.seed,
        steps_list=args.steps,
        stop_step=args.stop_step,
    )


if __name__ == "__main__":
    main()