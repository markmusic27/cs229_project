"""
Low-pass filtering operations for ILVR.
"""

import torch
import torch.nn.functional as F


def lowpass(x: torch.Tensor, down_factor: int) -> torch.Tensor:
    """
    Apply low-pass filter via downsampling and upsampling.
    
    This extracts the low-frequency structure of the tensor by:
    1. Downsampling by the given factor
    2. Upsampling back to original size
    
    Args:
        x: Input tensor of shape (B, C, H, W).
        down_factor: Downsampling factor. 1 = no filtering.
    
    Returns:
        Low-pass filtered tensor of same shape.
    """
    if down_factor <= 1:
        return x.clone()
    
    _, _, h, w = x.shape
    h_ds = max(1, h // down_factor)
    w_ds = max(1, w // down_factor)
    
    # Downsample then upsample
    x_down = F.interpolate(x, size=(h_ds, w_ds), mode="bicubic", align_corners=False)
    x_up = F.interpolate(x_down, size=(h, w), mode="bicubic", align_corners=False)
    
    return x_up

