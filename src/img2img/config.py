"""
Configuration for Img2Img ILVR pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Img2ImgConfig:
    """
    Configuration for Img2Img generation with optional ILVR guidance.
    
    Strength parameter controls how much the image changes:
        - strength=1.0: Start from pure noise (like txt2img)
        - strength=0.5: Start from 50% noised reference (balanced)
        - strength=0.2: Start from 20% noised reference (mostly preserve original)
    
    Attributes:
        prompt: Text prompt for generation.
        negative_prompt: Negative prompt to avoid.
        strength: Denoising strength (0-1). Lower = more like original.
        guidance_scale: Classifier-free guidance scale.
        num_inference_steps: Total diffusion steps.
        height: Output image height.
        width: Output image width.
        seed: Random seed for reproducibility.
        iterations: Number of iterative refinements.
        use_ilvr: Whether to apply ILVR guidance during denoising.
        down_factor: ILVR downsampling factor (higher = less constraint).
        stop_step: Step index to stop applying ILVR.
        use_cls_conditioning: Whether to use CLS token conditioning (requires cls_embedding).
        ip_adapter_scale: Scale for IP-Adapter when using CLS conditioning.
    """
    
    # Generation parameters
    prompt: str = "a high quality photograph"
    negative_prompt: str = "blurry, low quality, distorted, noise, artifacts"
    strength: float = 0.5
    guidance_scale: float = 3.0
    num_inference_steps: int = 100
    height: int = 1024
    width: int = 1024
    seed: int = 42
    iterations: int = 5
    
    # ILVR parameters
    use_ilvr: bool = False
    down_factor: int = 4
    stop_step: int = 50
    
    # CLS conditioning parameters (optional)
    use_cls_conditioning: bool = False
    ip_adapter_scale: float = 0.6
    
    def __post_init__(self):
        """Validate configuration values."""
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be in [0, 1], got {self.strength}")
        if self.down_factor < 1:
            raise ValueError(f"down_factor must be >= 1, got {self.down_factor}")
        if self.num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {self.num_inference_steps}")
        if self.iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")
    
    def copy(self, **overrides) -> "Img2ImgConfig":
        """Create a copy with optional overrides."""
        from dataclasses import asdict
        params = asdict(self)
        params.update(overrides)
        return Img2ImgConfig(**params)
    
    @property
    def actual_steps(self) -> int:
        """Number of denoising steps that will actually run."""
        return int(self.num_inference_steps * self.strength)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/visualization."""
        from dataclasses import asdict
        return asdict(self)

