"""
Img2Img ILVR Pipeline.

Start diffusion from a reference image instead of pure noise,
with optional ILVR guidance for structure preservation.

CLS Conditioning:
    For sender/receiver scenarios, use the CLS encoding functions:
    
    # Sender side (extracts ~1.3 KB from original image):
    from img2img import extract_and_quantize_cls
    quantized = extract_and_quantize_cls(original_image)
    send(quantized.to_bytes())
    
    # Receiver side (uses CLS to guide generation):
    from img2img import QuantizedCLS, dequantize_cls, Img2ImgConfig, Img2ImgPipeline
    quantized = QuantizedCLS.from_bytes(received_bytes)
    cls_embedding = dequantize_cls(quantized)
    
    config = Img2ImgConfig(use_cls_conditioning=True)
    pipeline = Img2ImgPipeline(config=config)
    pipeline.load_reference("compressed.jpg")
    image = pipeline.generate(cls_embedding=cls_embedding)
"""

from .config import Img2ImgConfig
from .pipeline import Img2ImgPipeline
from .encoding import (
    extract_and_quantize_cls,
    dequantize_cls,
    QuantizedCLS,
)

__all__ = [
    "Img2ImgConfig",
    "Img2ImgPipeline",
    # CLS conditioning (sender/receiver)
    "extract_and_quantize_cls",
    "dequantize_cls", 
    "QuantizedCLS",
]

