"""
Img2Img ILVR Pipeline.

Start diffusion from a reference image instead of pure noise,
with optional ILVR guidance for structure preservation.
"""

from .config import Img2ImgConfig
from .pipeline import Img2ImgPipeline

__all__ = ["Img2ImgConfig", "Img2ImgPipeline"]

