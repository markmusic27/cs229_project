"""
ILVR for SDXL - Fixed VAE Encoding

The NaN issue comes from VAE encoding in fp16.
Fix: encode in fp32, then convert back to fp16.
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
    """Check tensor for NaN/Inf and print stats."""
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
    """
    Encode PIL image to VAE latent space SAFELY.
    
    Key fix: Do VAE encoding in fp32 to avoid NaN issues,
    then convert result back to the target dtype.
    """
    # Preprocess image
    image_tensor = pipe.image_processor.preprocess(image)
    print(f"  Preprocessed image: shape={image_tensor.shape}, range=[{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
    
    # CRITICAL: Encode in fp32 to avoid precision issues
    image_tensor_fp32 = image_tensor.to(device=device, dtype=torch.float32)
    
    with torch.no_grad():
        # Temporarily convert VAE to fp32 for encoding
        vae_dtype = pipe.vae.dtype
        pipe.vae.to(dtype=torch.float32)
        
        posterior = pipe.vae.encode(image_tensor_fp32)
        
        if hasattr(posterior, "latent_dist"):
            latents = posterior.latent_dist.mean
        else:
            latents = posterior.mean
        
        # Scale by VAE factor
        latents = latents * pipe.vae.config.scaling_factor
        
        # Convert VAE back to original dtype
        pipe.vae.to(dtype=vae_dtype)
        
        # Convert latents to target dtype
        latents = latents.to(dtype=dtype)
    
    print(f"  Encoded latent: shape={latents.shape}, range=[{latents.min():.3f}, {latents.max():.3f}]")
    check_tensor(latents, "latent")
    
    return latents


class ILVRCallback:
    """ILVR callback with proper handling."""
    
    def __init__(
        self,
        y0_latent: torch.Tensor,
        noise: torch.Tensor,
        scheduler,
        down_factor: int = 8,
        strength: float = 1.0,
        start_step: int = 0,
        end_step: Optional[int] = None,
        debug: bool = True,
    ):
        self.y0_latent = y0_latent
        self.noise = noise
        self.scheduler = scheduler
        self.down_factor = down_factor
        self.strength = strength
        self.start_step = start_step
        self.end_step = end_step
        self.debug = debug
        
        # Verify y0_latent is valid
        if torch.isnan(y0_latent).any():
            raise ValueError("y0_latent contains NaN values!")
        
    def __call__(self, pipe, step_index: int, timestep: torch.Tensor, callback_kwargs: dict) -> dict:
        latents = callback_kwargs["latents"]
        num_steps = self.scheduler.num_inference_steps
        end_step = self.end_step if self.end_step is not None else num_steps
        
        apply_ilvr = self.start_step <= step_index < end_step
        
        if self.debug and step_index % 10 == 0:
            print(f"\n--- Step {step_index}/{num_steps}, t={timestep.item():.0f} ---")
            check_tensor(latents, "input_latents")
        
        if apply_ilvr:
            t = timestep.reshape(-1).to(self.y0_latent.device)
            
            # Compute noised reference
            y_t = self.scheduler.add_noise(self.y0_latent, self.noise, t)
            
            if self.debug and step_index % 10 == 0:
                check_tensor(y_t, "y_t")
            
            # Handle CFG batch size (latents may be [2, 4, H, W])
            batch_size = latents.shape[0]
            if y_t.shape[0] != batch_size:
                y_t = y_t.expand(batch_size, -1, -1, -1)
            
            # Low-pass filter
            phi_y = lowpass_phi(y_t, self.down_factor)
            phi_x = lowpass_phi(latents, self.down_factor)
            
            # ILVR update
            if self.strength >= 1.0:
                latents_new = phi_y + (latents - phi_x)
            else:
                delta = phi_y - phi_x
                latents_new = latents + self.strength * delta
            
            if self.debug and step_index % 10 == 0:
                check_tensor(latents_new, "output_latents")
            
            # Safety clamp
            latents_new = torch.clamp(latents_new, -30, 30)
            latents_new = torch.nan_to_num(latents_new, nan=0.0)
            
            callback_kwargs["latents"] = latents_new
        
        return callback_kwargs


def run_ilvr(
    prompt: str,
    ref_path: str,
    output_path: str = "ilvr_output.png",
    negative_prompt: str = "blurry, low quality, distorted",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    down_factor: int = 8,
    strength: float = 1.0,
    height: int = 1024,
    width: int = 1024,
    seed: int = 42,
    debug: bool = True,
):
    """Run SDXL with ILVR."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    print(f"Device: {device}, dtype: {dtype}")
    print(f"Loading SDXL...")
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        use_safetensors=True,
    ).to(device)
    
    # Use DDIM scheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    print(f"Scheduler: {type(pipe.scheduler).__name__}")
    
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("Enabled xformers")
    except:
        pass
    
    # Load reference
    print(f"\nLoading reference: {ref_path}")
    ref_image = Image.open(ref_path).convert("RGB")
    ref_array = np.array(ref_image)
    print(f"Reference: size={ref_image.size}, pixel range=[{ref_array.min()}, {ref_array.max()}]")
    
    ref_image = ref_image.resize((width, height), Image.LANCZOS)
    
    # Encode reference with SAFE method
    print("\nEncoding reference (fp32 for stability)...")
    y0_latent = encode_image_to_latent_safe(pipe, ref_image, device, dtype)
    
    # Verify encoding worked
    if torch.isnan(y0_latent).any():
        print("\n❌ ERROR: VAE encoding still produced NaN!")
        print("This may be a corrupted model or incompatible image.")
        return None
    
    print("✓ Reference encoded successfully!")
    
    # Set up scheduler
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    
    # Create noise
    generator = torch.Generator(device=device).manual_seed(seed)
    ref_noise = torch.randn(y0_latent.shape, device=device, dtype=dtype, generator=generator)
    
    # Reset generator
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Create callback
    callback = ILVRCallback(
        y0_latent=y0_latent,
        noise=ref_noise,
        scheduler=pipe.scheduler,
        down_factor=down_factor,
        strength=strength,
        debug=debug,
    )
    
    print(f"\n{'='*50}")
    print(f"Generating with ILVR")
    print(f"  down_factor: {down_factor}")
    print(f"  strength: {strength}")
    print(f"  steps: {num_inference_steps}")
    print(f"  guidance: {guidance_scale}")
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
    
    # Check output
    output_array = np.array(output_image)
    print(f"\nOutput: pixel range=[{output_array.min()}, {output_array.max()}]")
    
    # Save
    output_image.save(output_path)
    print(f"Saved: {output_path}")
    
    # Save comparison
    comparison = Image.new("RGB", (width * 2, height))
    comparison.paste(ref_image, (0, 0))
    comparison.paste(output_image, (width, 0))
    comp_path = output_path.replace(".png", "_comparison.png")
    comparison.save(comp_path)
    print(f"Saved: {comp_path}")
    
    return output_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="a high quality photograph")
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, distorted")
    parser.add_argument("--output", type=str, default="ilvr_output.png")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--down_factor", type=int, default=8)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_debug", action="store_true")
    
    args = parser.parse_args()
    
    run_ilvr(
        prompt=args.prompt,
        ref_path=args.ref,
        output_path=args.output,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        down_factor=args.down_factor,
        strength=args.strength,
        height=args.height,
        width=args.width,
        seed=args.seed,
        debug=not args.no_debug,
    )


if __name__ == "__main__":
    main()