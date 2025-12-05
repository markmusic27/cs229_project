"""
VAE and CLIP encoding utilities.
"""

from dataclasses import dataclass
from typing import Tuple, Optional

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


# =============================================================================
# CLS Token Encoding (for CLIP conditioning)
# =============================================================================

@dataclass
class QuantizedCLS:
    """
    Quantized CLS embedding for transmission.
    
    This is what gets sent from sender to receiver.
    Total size: 1288 bytes (1280 int8 values + 8 bytes metadata)
    """
    data: bytes  # Quantized int8 values as bytes
    scale: float  # For dequantization
    zero_point: float  # For dequantization
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes for transmission."""
        import struct
        # Pack: scale (float32) + zero_point (float32) + data (1280 bytes)
        header = struct.pack('ff', self.scale, self.zero_point)
        return header + self.data
    
    @classmethod
    def from_bytes(cls, raw: bytes) -> "QuantizedCLS":
        """Deserialize from bytes."""
        import struct
        scale, zero_point = struct.unpack('ff', raw[:8])
        data = raw[8:]
        return cls(data=data, scale=scale, zero_point=zero_point)
    
    @property
    def size_bytes(self) -> int:
        """Total size in bytes."""
        return len(self.data) + 8  # data + scale + zero_point


def extract_and_quantize_cls(
    image: Image.Image,
    device: Optional[str] = None,
) -> QuantizedCLS:
    """
    SENDER FUNCTION: Extract and quantize CLS embedding from an image.
    
    This function should be called by the sender to create a compact
    representation of the original image for transmission.
    
    Args:
        image: Original PIL Image (before compression).
        device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
    
    Returns:
        QuantizedCLS: Compact representation (~1.3 KB) for transmission.
    
    Example:
        # Sender side
        quantized = extract_and_quantize_cls(original_image)
        send_to_receiver(quantized.to_bytes())  # ~1.3 KB
    """
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    # Load CLIP image encoder
    clip_model = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="sdxl_models/image_encoder",
        torch_dtype=dtype,
    ).to(device)
    
    clip_processor = CLIPImageProcessor.from_pretrained(
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    )
    
    # Process image and extract CLS
    inputs = clip_processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device=device, dtype=dtype)
    
    with torch.no_grad():
        outputs = clip_model(pixel_values, output_hidden_states=True)
        cls_embedding = outputs.image_embeds  # (1, 1280)
    
    # Clean up
    del clip_model
    torch.cuda.empty_cache()
    
    # Quantize to int8
    flat = cls_embedding.flatten().float().cpu()
    vmin, vmax = flat.min().item(), flat.max().item()
    scale = (vmax - vmin) / 254
    zero_point = vmin
    
    quantized = ((flat - zero_point) / scale).round().clamp(0, 255).byte()
    
    return QuantizedCLS(
        data=quantized.numpy().tobytes(),
        scale=scale,
        zero_point=zero_point,
    )


def dequantize_cls(
    quantized: QuantizedCLS,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    RECEIVER FUNCTION: Dequantize CLS embedding for use in generation.
    
    This function should be called by the receiver to convert the
    transmitted quantized CLS back to a usable tensor.
    
    Args:
        quantized: QuantizedCLS received from sender.
        device: Target device.
        dtype: Target dtype.
    
    Returns:
        Tensor of shape (1, 1280) ready for IP-Adapter.
    
    Example:
        # Receiver side
        quantized = QuantizedCLS.from_bytes(received_bytes)
        cls_embedding = dequantize_cls(quantized)
        # Use in pipeline.generate(..., cls_embedding=cls_embedding)
    """
    import numpy as np
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    # Convert bytes back to tensor
    quantized_array = np.frombuffer(quantized.data, dtype=np.uint8)
    quantized_tensor = torch.from_numpy(quantized_array.copy()).float()
    
    # Dequantize
    dequantized = quantized_tensor * quantized.scale + quantized.zero_point
    
    # Reshape to (1, 1280)
    cls_embedding = dequantized.reshape(1, 1280).to(device=device, dtype=dtype)
    
    return cls_embedding

