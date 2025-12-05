"""
Configuration dataclass for ILVR hyperparameters.
"""

from dataclasses import dataclass


@dataclass
class ILVRConfig:
    """
    Configuration for ILVR generation.
    
    Attributes:
        down_factor: Low-pass filter downsampling factor (N in paper).
                     Lower = more similar to reference.
                     - 2: Very close (fine details preserved)
                     - 4: Close (facial features preserved)  
                     - 8: Moderate similarity
                     - 16+: Coarse features only
        
        stop_step: Absolute step to stop ILVR conditioning.
                   Remaining steps allow the model to "clean up" artifacts.
        
        num_inference_steps: Total diffusion steps.
        
        guidance_scale: Classifier-free guidance scale.
                        Lower = less prompt influence, more reference fidelity.
        
        prompt: Text prompt for generation.
        
        negative_prompt: Negative prompt to avoid unwanted features.
        
        height: Output image height.
        
        width: Output image width.
        
        seed: Random seed for reproducibility.
    """
    
    # ILVR parameters
    down_factor: int = 4
    stop_step: int = 50
    
    # Diffusion parameters
    num_inference_steps: int = 100
    guidance_scale: float = 3.0
    
    # Prompts
    prompt: str = "a high quality photograph"
    negative_prompt: str = "blurry, low quality, distorted, noise, artifacts"
    
    # Image dimensions
    height: int = 1024
    width: int = 1024
    
    # Reproducibility
    seed: int = 42
    
    def update(self, **kwargs) -> "ILVRConfig":
        """
        Update config with new values. Returns self for chaining.
        
        Example:
            config.update(down_factor=2, stop_step=30)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
        return self
    
    def copy(self, **kwargs) -> "ILVRConfig":
        """
        Create a copy with optional overrides.
        
        Example:
            new_config = config.copy(down_factor=2)
        """
        new_config = ILVRConfig(
            down_factor=self.down_factor,
            stop_step=self.stop_step,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            height=self.height,
            width=self.width,
            seed=self.seed,
        )
        if kwargs:
            new_config.update(**kwargs)
        return new_config
    
    @property
    def cleanup_steps(self) -> int:
        """Number of steps after ILVR stops (for denoising cleanup)."""
        return max(0, self.num_inference_steps - self.stop_step)
    
    def __repr__(self) -> str:
        return (
            f"ILVRConfig(\n"
            f"  down_factor={self.down_factor},\n"
            f"  stop_step={self.stop_step},\n"
            f"  num_inference_steps={self.num_inference_steps},\n"
            f"  cleanup_steps={self.cleanup_steps},\n"
            f"  guidance_scale={self.guidance_scale},\n"
            f"  seed={self.seed}\n"
            f")"
        )