"""
Frequency filtering operations for ILVR.
"""

import torch
import torch.nn.functional as F


def lowpass_phi(x: torch.Tensor, down_factor: int) -> torch.Tensor:
    """
    Low-pass filter via downsampling then upsampling.
    
    This is the φ_N operation from the ILVR paper.
    Approximates keeping only the low-frequency components of x.
    
    Args:
        x: Tensor of shape [B, C, H, W]
        down_factor: Downsampling factor N. Higher = coarser filtering.
                     - N=2: Keep fine details
                     - N=4: Keep medium details
                     - N=8+: Keep only coarse structure
    
    Returns:
        Filtered tensor with same shape as input.
    """
    if down_factor <= 1:
        return x.clone()
    
    _, _, h, w = x.shape
    h_ds = max(1, h // down_factor)
    w_ds = max(1, w // down_factor)
    
    x_down = F.interpolate(x, size=(h_ds, w_ds), mode="bicubic", align_corners=False)
    x_up = F.interpolate(x_down, size=(h, w), mode="bicubic", align_corners=False)
    
    return x_up


def highpass(x: torch.Tensor, down_factor: int) -> torch.Tensor:
    """
    High-pass filter: extracts high-frequency components.
    
    Args:
        x: Tensor of shape [B, C, H, W]
        down_factor: Downsampling factor for the low-pass component.
    
    Returns:
        High-frequency components: x - φ(x)
    """
    return x - lowpass_phi(x, down_factor)