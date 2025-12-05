"""
ILVR Callback for diffusion pipeline.
"""

import torch

from .filters import lowpass_phi


class ILVRCallback:
    """
    Callback that applies ILVR refinement during diffusion.
    
    At each step before stop_step, replaces the low-frequency
    components of the latent with those from the noised reference.
    
    This implements Equation 8 from the ILVR paper:
        x_{t-1} = φ(y_{t-1}) + (x'_{t-1} - φ(x'_{t-1}))
    
    Where:
        - x'_{t-1} is the unconditional proposal from the diffusion model
        - y_{t-1} is the noised reference at timestep t-1
        - φ is the low-pass filter (downsample then upsample)
    """
    
    def __init__(
        self,
        y0_latent: torch.Tensor,
        noise: torch.Tensor,
        scheduler,
        down_factor: int = 4,
        stop_step: int = 50,
    ):
        """
        Args:
            y0_latent: Clean reference latent [1, 4, H, W]
            noise: Fixed noise for reference [1, 4, H, W]
            scheduler: Diffusion scheduler
            down_factor: Low-pass filter factor (N in paper)
            stop_step: Absolute step to stop ILVR
        """
        self.y0_latent = y0_latent
        self.noise = noise
        self.scheduler = scheduler
        self.down_factor = down_factor
        self.stop_step = stop_step
    
    def __call__(
        self,
        pipe,
        step_index: int,
        timestep: torch.Tensor,
        callback_kwargs: dict,
    ) -> dict:
        """
        Apply ILVR at each diffusion step.
        
        Args:
            pipe: The diffusion pipeline
            step_index: Current step index (0-indexed)
            timestep: Current timestep value
            callback_kwargs: Dict containing 'latents'
        
        Returns:
            Updated callback_kwargs with modified latents
        """
        latents = callback_kwargs["latents"]
        
        # Only apply ILVR before stop_step
        if step_index < self.stop_step:
            # Compute noised reference at this timestep
            t = timestep.reshape(-1).to(self.y0_latent.device)
            y_t = self.scheduler.add_noise(self.y0_latent, self.noise, t)
            
            # Handle CFG batch size (latents may be [2, 4, H, W])
            if y_t.shape[0] != latents.shape[0]:
                y_t = y_t.expand(latents.shape[0], -1, -1, -1)
            
            # ILVR: replace low-frequency components
            # x_new = φ(y_t) + (x - φ(x))
            phi_y = lowpass_phi(y_t, self.down_factor)
            phi_x = lowpass_phi(latents, self.down_factor)
            latents_new = phi_y + (latents - phi_x)
            
            # Safety clamp to prevent explosion
            latents_new = torch.clamp(latents_new, -30, 30)
            
            callback_kwargs["latents"] = latents_new
        
        return callback_kwargs