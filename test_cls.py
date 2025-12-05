"""
Test: Img2Img with CLIP CLS token conditioning only.

Uses only the CLS token (~2.5 KB) instead of full patch features (~660 KB).
This is a 256x reduction in conditioning data size.

Original: dataset/budi_og.png (CLS token extracted for conditioning)
Compressed: dataset/budi_comp.png (img2img starting point)
"""

import os
import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionXLImg2ImgPipeline, DDIMScheduler
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor


def prepare_image(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Prepare image to target size with center crop if needed."""
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


def create_comparison_grid(images: list, labels: list, output_path: str, title: str = ""):
    """Create a comparison grid of all images."""
    n_images = len(images)
    n_cols = min(4, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    cell_size = 512
    label_h = 40
    header_h = 60 if title else 0
    
    grid_w = cell_size * n_cols
    grid_h = header_h + (cell_size + label_h) * n_rows
    
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)
    
    if title:
        draw.rectangle([(0, 0), (grid_w, header_h)], fill="#2a2a2a")
        draw.text((20, 18), title, fill="white")
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // n_cols
        col = idx % n_cols
        x = col * cell_size
        y = header_h + row * (cell_size + label_h)
        
        img_small = img.resize((cell_size, cell_size), Image.LANCZOS)
        grid.paste(img_small, (x, y + label_h))
        draw.text((x + 10, y + 10), label, fill="black")
    
    grid.save(output_path)
    print(f"Saved: {output_path}")


def extract_clip_cls_embedding(image: Image.Image, device, dtype):
    """
    Extract only the CLS token embedding from CLIP.
    
    Returns:
        CLS embedding: shape (1, 1280) for ViT-bigG
        Size: 1280 * 16 bits = 2.5 KB
    """
    # Load CLIP image encoder (same one used by SDXL IP-Adapter)
    clip_model = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="sdxl_models/image_encoder",
        torch_dtype=dtype,
    ).to(device)
    
    # Load processor from OpenCLIP (IP-Adapter doesn't include it)
    clip_processor = CLIPImageProcessor.from_pretrained(
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    )
    
    # Process image
    inputs = clip_processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device=device, dtype=dtype)
    
    # Get CLIP outputs
    with torch.no_grad():
        outputs = clip_model(pixel_values, output_hidden_states=True)
    
    # CLS token is the image_embeds (projected) - shape (1, 1280)
    cls_embedding = outputs.image_embeds
    
    # Clean up to save memory
    del clip_model
    torch.cuda.empty_cache()
    
    return cls_embedding


def run_cls_conditioned_img2img(
    original_path: str = "dataset/camel_og.png",
    compressed_path: str = "dataset/camel_comp.png",
    output_dir: str = "test_cls_results",
    prompt: str = "a high quality photograph",
    negative_prompt: str = "blurry, low quality, distorted, noise, artifacts",
    strength: float = 0.3,
    guidance_scale: float = 3.0,
    ip_adapter_scale: float = 0.6,
    height: int = 1024,
    width: int = 1024,
    seed: int = 42,
    steps: int = 100,
    iterations: int = 5,
):
    """
    Run img2img with CLS-only CLIP conditioning.
    
    Only sends 2.5 KB of CLIP data instead of 660 KB!
    """
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    print(f"Device: {device}")
    print(f"Original (CLS extraction): {original_path}")
    print(f"Compressed (starting point): {compressed_path}")
    print(f"IP-Adapter scale: {ip_adapter_scale}")
    print(f"Strength: {strength}")
    print(f"Iterations: {iterations}")
    
    # Load images first
    print(f"\nLoading original image: {original_path}")
    original_image = Image.open(original_path).convert("RGB")
    original_image = prepare_image(original_image, width, height)
    
    print(f"Loading compressed image: {compressed_path}")
    compressed_image = Image.open(compressed_path).convert("RGB")
    compressed_image = prepare_image(compressed_image, width, height)
    
    # Extract CLS token from original image
    print("\n" + "="*60)
    print("Extracting CLIP CLS token from original image...")
    print("="*60)
    cls_embedding = extract_clip_cls_embedding(original_image, device, dtype)
    
    cls_size_bits = cls_embedding.numel() * 16  # float16
    cls_size_kb = cls_size_bits / 8 / 1024
    print(f"CLS embedding shape: {cls_embedding.shape}")
    print(f"CLS embedding size: {cls_embedding.numel()} values = {cls_size_kb:.2f} KB")
    print(f"(vs ~660 KB for full patch features - {660/cls_size_kb:.0f}x reduction!)")
    
    # Load pipeline
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
    
    # Load IP-Adapter
    print("\nLoading IP-Adapter...")
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl.bin",
    )
    pipe.set_ip_adapter_scale(ip_adapter_scale)
    print(f"IP-Adapter loaded with scale={ip_adapter_scale}")
    
    # Prepare the CLS embedding for IP-Adapter
    # IP-Adapter expects [negative_embeds, positive_embeds] concatenated along batch dim
    # For negative, we use zeros (unconditioned)
    negative_cls = torch.zeros_like(cls_embedding)
    
    # Concatenate negative and positive: shape (2, 1280)
    cls_for_ip = torch.cat([negative_cls, cls_embedding], dim=0)
    # Add sequence dimension: shape (2, 1, 1280)  
    cls_for_ip = cls_for_ip.unsqueeze(1)
    print(f"CLS embedding prepared for IP-Adapter: {cls_for_ip.shape} (neg + pos)")
    
    # Store all images for comparison
    all_images = [original_image.copy(), compressed_image.copy()]
    all_labels = ["Original", "Compressed"]
    
    # Save inputs
    original_image.save(os.path.join(output_dir, "original.png"))
    compressed_image.save(os.path.join(output_dir, "compressed.png"))
    
    # Save CLS embedding for reference
    torch.save(cls_embedding.cpu(), os.path.join(output_dir, "cls_embedding.pt"))
    print(f"Saved CLS embedding to cls_embedding.pt")
    
    current_image = compressed_image
    
    # Run iterations
    for i in range(1, iterations + 1):
        print(f"\n{'='*60}")
        print(f"Iteration {i}/{iterations}")
        print(f"{'='*60}")
        
        iter_seed = seed + i
        generator = torch.Generator(device=device).manual_seed(iter_seed)
        
        actual_steps = int(steps * strength)
        print(f"  Strength {strength}: denoising ~{actual_steps} steps")
        print(f"  CLS-only conditioning ({cls_size_kb:.2f} KB)")
        
        # Run img2img with CLS embedding
        # Using ip_adapter_image_embeds to pass pre-computed embeddings
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=current_image,
            ip_adapter_image_embeds=[cls_for_ip],  # CLS-only!
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        output_image = result.images[0]
        
        # Save this iteration
        iter_path = os.path.join(output_dir, f"iter_{i}.png")
        output_image.save(iter_path)
        print(f"Saved: iter_{i}.png")
        
        all_images.append(output_image.copy())
        all_labels.append(f"Iter {i}")
        
        current_image = output_image
    
    # Create comparison grid
    print("\n" + "="*60)
    print("Creating comparison grid...")
    print("="*60)
    
    title = f"CLS-Only CLIP ({cls_size_kb:.1f}KB) | IP={ip_adapter_scale} | str={strength} | steps={steps}"
    comparison_path = os.path.join(output_dir, "comparison.png")
    create_comparison_grid(all_images, all_labels, comparison_path, title)
    
    print(f"\n✓ All images saved to {output_dir}/")
    print(f"  - CLIP CLS token: {cls_size_kb:.2f} KB (vs 660 KB full)")
    print(f"  - {iterations} iterations")


if __name__ == "__main__":
    run_cls_conditioned_img2img()

