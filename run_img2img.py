"""
Image-to-Image ILVR: Start diffusion from the reference image instead of pure noise.

Uses the native StableDiffusionXLImg2ImgPipeline for memory efficiency.

Strength parameter:
- strength=1.0: Start from pure noise (like regular txt2img)
- strength=0.5: Start from 50% noised reference (balance of original and new)
- strength=0.2: Start from 20% noised reference (mostly preserve original)
"""

import argparse
import os

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import numpy as np

from diffusers import StableDiffusionXLImg2ImgPipeline, DDIMScheduler


def prepare_image(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """
    Prepare image to target size.
    - If larger: center crop to target aspect ratio, then resize
    - If smaller: resize to fit
    """
    img_width, img_height = image.size
    target_ratio = target_width / target_height
    img_ratio = img_width / img_height
    
    if img_width >= target_width and img_height >= target_height:
        if img_ratio > target_ratio:
            new_width = int(img_height * target_ratio)
            left = (img_width - new_width) // 2
            image = image.crop((left, 0, left + new_width, img_height))
        elif img_ratio < target_ratio:
            new_height = int(img_width / target_ratio)
            top = (img_height - new_height) // 2
            image = image.crop((0, top, img_width, top + new_height))
    
    image = image.resize((target_width, target_height), Image.LANCZOS)
    return image


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
    """Optional ILVR guidance during denoising."""
    def __init__(self, y0_latent, noise, scheduler, down_factor=4, stop_step=100):
        self.y0_latent = y0_latent
        self.noise = noise
        self.scheduler = scheduler
        self.down_factor = down_factor
        self.stop_step = stop_step
        
    def __call__(self, pipe, step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        
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


def create_comparison_grid(images: list, labels: list, output_path: str, hyperparams: dict = None):
    """Create a comparison grid of all images with hyperparameters header."""
    n_images = len(images)
    n_cols = min(4, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    cell_size = 450
    label_h = 50
    header_h = 140 if hyperparams else 0
    
    grid_w = cell_size * n_cols
    grid_h = header_h + (cell_size + label_h) * n_rows
    
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)
    
    if hyperparams:
        draw.rectangle([(0, 0), (grid_w, header_h)], fill="#f0f0f0")
        draw.line([(0, header_h), (grid_w, header_h)], fill="#cccccc", width=2)
        
        lines = [
            f"Img2Img ILVR - Start from Reference",
            f"Strength: {hyperparams.get('strength', 'N/A')}  |  Steps: {hyperparams.get('steps', 'N/A')}  |  Guidance: {hyperparams.get('guidance_scale', 'N/A')}",
            f"ILVR: down_factor={hyperparams.get('down_factor', 'N/A')}, stop_step={hyperparams.get('stop_step', 'N/A')}, enabled={hyperparams.get('use_ilvr', 'N/A')}",
            f"Iterations: {hyperparams.get('iterations', 'N/A')}  |  Seed: {hyperparams.get('seed', 'N/A')}  |  Size: {hyperparams.get('width', 'N/A')}x{hyperparams.get('height', 'N/A')}",
            f"Reference: {hyperparams.get('ref_path', 'N/A')}",
        ]
        
        y_text = 10
        for i, line in enumerate(lines):
            if i == 0:
                draw.text((20, y_text), line, fill="#333333")
                y_text += 28
            else:
                draw.text((20, y_text), line, fill="#555555")
                y_text += 22
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // n_cols
        col = idx % n_cols
        x = col * cell_size
        y = header_h + row * (cell_size + label_h)
        
        img_small = img.resize((cell_size, cell_size), Image.LANCZOS)
        grid.paste(img_small, (x, y + label_h))
        draw.text((x + 10, y + 15), label, fill="black")
    
    grid.save(output_path)
    print(f"Saved comparison grid: {output_path}")


def run_multi_img2img(
    ref_path: str,
    output_dir: str = "img2img_results",
    prompt: str = "a high quality photograph",
    negative_prompt: str = "blurry, low quality, distorted, noise, artifacts",
    strength: float = 0.5,
    guidance_scale: float = 3.0,
    height: int = 1024,
    width: int = 1024,
    seed: int = 42,
    steps: int = 100,
    iterations: int = 5,
    use_ilvr: bool = False,
    down_factor: int = 4,
    stop_step: int = 50,
):
    """
    Run img2img iteratively, using each output as the next input.
    Uses native Img2ImgPipeline for memory efficiency.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    print(f"Device: {device}")
    print(f"Mode: Img2Img (start from reference)")
    print(f"Strength: {strength} (lower = more like original)")
    print(f"ILVR: {'enabled' if use_ilvr else 'disabled'}")
    print(f"Iterations: {iterations}")
    print(f"Steps per iteration: {steps}")
    print(f"Guidance scale: {guidance_scale}")
    
    print("\nLoading SDXL Img2Img Pipeline...")
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
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
    
    # Load initial reference
    print(f"\nLoading initial reference: {ref_path}")
    current_image = Image.open(ref_path).convert("RGB")
    orig_size = current_image.size
    current_image = prepare_image(current_image, width, height)
    print(f"  Original size: {orig_size[0]}x{orig_size[1]} -> Target: {width}x{height}")
    
    # Store all images for comparison
    all_images = [current_image.copy()]
    all_labels = ["Original"]
    
    # Save original
    original_path = os.path.join(output_dir, "iter_0_original.png")
    current_image.save(original_path)
    print(f"Saved: iter_0_original.png")
    
    # Run iterations
    for i in range(1, iterations + 1):
        print(f"\n{'='*60}")
        print(f"Iteration {i}/{iterations}")
        print(f"{'='*60}")
        
        # Use different seed for each iteration
        iter_seed = seed + i
        generator = torch.Generator(device=device).manual_seed(iter_seed)
        
        # Setup ILVR callback if enabled
        callback = None
        callback_inputs = None
        if use_ilvr and down_factor > 1:
            # Encode current image for ILVR
            y0_latent = encode_image_to_latent_safe(pipe, current_image, device, dtype)
            noise = torch.randn(y0_latent.shape, device=device, dtype=dtype, generator=generator)
            generator = torch.Generator(device=device).manual_seed(iter_seed)  # Reset generator
            
            callback = ILVRCallback(
                y0_latent=y0_latent,
                noise=noise,
                scheduler=pipe.scheduler,
                down_factor=down_factor,
                stop_step=stop_step,
            )
            callback_inputs = ["latents"]
        
        # Calculate actual steps
        actual_steps = int(steps * strength)
        print(f"  Strength {strength}: denoising ~{actual_steps} steps")
        
        # Run img2img
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=current_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=callback_inputs,
        )
        
        output_image = result.images[0]
        
        # Save this iteration's result
        iter_path = os.path.join(output_dir, f"iter_{i}.png")
        output_image.save(iter_path)
        print(f"Saved: iter_{i}.png")
        
        # Store for comparison
        all_images.append(output_image.copy())
        all_labels.append(f"Iteration {i}")
        
        # Use output as next input
        current_image = output_image
    
    # Create comparison grid
    print("\n" + "="*60)
    print("Creating comparison grid...")
    print("="*60)
    
    hyperparams = {
        "ref_path": os.path.basename(ref_path),
        "strength": strength,
        "steps": steps,
        "stop_step": stop_step,
        "down_factor": down_factor,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "iterations": iterations,
        "width": width,
        "height": height,
        "use_ilvr": use_ilvr,
    }
    
    comparison_path = os.path.join(output_dir, "comparison.png")
    create_comparison_grid(all_images, all_labels, comparison_path, hyperparams)
    
    print(f"\n✓ All {iterations + 1} images saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Img2Img ILVR: Start diffusion from reference image instead of noise",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic img2img with 50% strength (balance between original and new)
  python img2img_ilvr.py --ref images/test.png --strength 0.5 --iterations 5
  
  # Low strength = more like original
  python img2img_ilvr.py --ref images/city.png --strength 0.3 --iterations 5
  
  # High strength = more variation (closer to txt2img)
  python img2img_ilvr.py --ref images/test.png --strength 0.8 --iterations 5
  
  # With ILVR guidance for extra control
  python img2img_ilvr.py --ref images/test.png --strength 0.5 --use_ilvr --down_factor 4
  
  # Compare different strengths
  python img2img_ilvr.py --ref images/test.png --strength 0.2  # mostly original
  python img2img_ilvr.py --ref images/test.png --strength 0.5  # balanced
  python img2img_ilvr.py --ref images/test.png --strength 0.8  # mostly new
        """
    )
    
    parser.add_argument("--ref", type=str, required=True, help="Initial reference image path")
    parser.add_argument("--output_dir", type=str, default="img2img_results", help="Output directory")
    parser.add_argument("--prompt", type=str, default="a high quality photograph")
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, distorted, noise, artifacts")
    parser.add_argument("--strength", type=float, default=0.5, 
                        help="Denoising strength (0-1). Lower=more like original, higher=more variation")
    parser.add_argument("--guidance", type=float, default=3.0, help="CFG scale")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42, help="Base seed (incremented each iteration)")
    parser.add_argument("--steps", type=int, default=100, help="Total inference steps")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations")
    
    # ILVR options
    parser.add_argument("--use_ilvr", action="store_true", help="Enable ILVR guidance during denoising")
    parser.add_argument("--down_factor", type=int, default=4, help="ILVR downsampling factor")
    parser.add_argument("--stop_step", type=int, default=50, help="Step to stop ILVR")
    
    args = parser.parse_args()
    
    run_multi_img2img(
        ref_path=args.ref,
        output_dir=args.output_dir,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        strength=args.strength,
        guidance_scale=args.guidance,
        height=args.height,
        width=args.width,
        seed=args.seed,
        steps=args.steps,
        iterations=args.iterations,
        use_ilvr=args.use_ilvr,
        down_factor=args.down_factor,
        stop_step=args.stop_step,
    )


if __name__ == "__main__":
    main()
