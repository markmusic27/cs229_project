"""
VAE encoding utilities.
"""

import torch
from PIL import Image


def encode_image(pipe, image: Image.Image, device, dtype) -> torch.Tensor:
    """
    Encode a PIL image to VAE latent space.
    
    Uses fp32 for VAE encoding to avoid numerical issues,
    then converts back to target dtype.
    
    Args:
        pipe: Diffusers pipeline with VAE.
        image: PIL Image to encode.
        device: Target device.
        dtype: Target dtype for output.
    
    Returns:
        Latent tensor of shape (1, 4, H/8, W/8).
    """
    # Preprocess image
    image_tensor = pipe.image_processor.preprocess(image)
    image_tensor = image_tensor.to(device=device, dtype=torch.float32)
    
    with torch.no_grad():
        # Temporarily use fp32 for VAE stability
        vae_dtype = pipe.vae.dtype
        pipe.vae.to(dtype=torch.float32)
        
        # Encode
        posterior = pipe.vae.encode(image_tensor)
        latents = posterior.latent_dist.mean * pipe.vae.config.scaling_factor
        
        # Restore VAE dtype
        pipe.vae.to(dtype=vae_dtype)
        latents = latents.to(dtype=dtype)
    
    return latents

