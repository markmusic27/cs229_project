"""
Image encoding utilities for ILVR.
"""

import torch
from PIL import Image


def encode_image_safe(
    pipe,
    image: Image.Image,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Encode PIL image to VAE latent space safely.
    
    Uses fp32 for VAE encoding to avoid NaN issues with fp16,
    then converts the result back to the target dtype.
    
    Args:
        pipe: StableDiffusionXLPipeline instance
        image: PIL Image to encode
        device: Target device
        dtype: Target dtype for output
    
    Returns:
        Latent tensor of shape [1, 4, H//8, W//8]
    """
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