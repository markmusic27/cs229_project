"""
ILVR callback for guided denoising.
"""

import torch
from .filters import lowpass


class ILVRCallback:
    """
    ILVR (Iterative Latent Variable Refinement) callback.
    
    Applies low-frequency guidance from a reference image during denoising.
    The callback replaces low-frequency components of the denoised latent
    with those from the noised reference at each step.
    
    Args:
        y0_latent: Reference image latent (clean).
        noise: Noise tensor used for reference.
        scheduler: Diffusion scheduler.
        down_factor: Low-pass filter factor. Higher = less constraint.
        stop_step: Step index to stop applying ILVR.
    """
    
    def __init__(
        self,
        y0_latent: torch.Tensor,
        noise: torch.Tensor,
        scheduler,
        down_factor: int = 4,
        stop_step: int = 50,
    ):
        self.y0_latent = y0_latent
        self.noise = noise
        self.scheduler = scheduler
        self.down_factor = down_factor
        self.stop_step = stop_step
    
    def __call__(self, pipe, step_index: int, timestep, callback_kwargs: dict) -> dict:
        """
        Apply ILVR guidance at each denoising step.
        
        Args:
            pipe: The diffusion pipeline.
            step_index: Current step index.
            timestep: Current timestep.
            callback_kwargs: Dict containing 'latents'.
        
        Returns:
            Updated callback_kwargs with modified latents.
        """
        latents = callback_kwargs["latents"]
        
        if step_index < self.stop_step:
            # Get noised reference at current timestep
            t = timestep.reshape(-1).to(self.y0_latent.device)
            y_t = self.scheduler.add_noise(self.y0_latent, self.noise, t)
            
            # Handle batch size mismatch (CFG doubles batch)
            if y_t.shape[0] != latents.shape[0]:
                y_t = y_t.expand(latents.shape[0], -1, -1, -1)
            
            # Replace low-frequency components
            phi_y = lowpass(y_t, self.down_factor)
            phi_x = lowpass(latents, self.down_factor)
            latents_new = phi_y + (latents - phi_x)
            
            # Clamp for stability
            latents_new = torch.clamp(latents_new, -30, 30)
            callback_kwargs["latents"] = latents_new
        
        return callback_kwargs

