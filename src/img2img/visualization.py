"""
Visualization utilities for comparison grids.
"""

from typing import List, Optional
from PIL import Image, ImageDraw


def create_comparison_grid(
    images: List[Image.Image],
    labels: List[str],
    output_path: str,
    hyperparams: Optional[dict] = None,
    cell_size: int = 450,
    max_cols: int = 4,
) -> Image.Image:
    """
    Create a comparison grid of images with labels and optional header.
    
    Args:
        images: List of PIL Images to display.
        labels: List of labels for each image.
        output_path: Path to save the grid.
        hyperparams: Optional dict of hyperparameters to show in header.
        cell_size: Size of each cell in pixels.
        max_cols: Maximum columns in grid.
    
    Returns:
        The created grid image.
    """
    n_images = len(images)
    n_cols = min(max_cols, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    label_h = 50
    header_h = 140 if hyperparams else 0
    
    grid_w = cell_size * n_cols
    grid_h = header_h + (cell_size + label_h) * n_rows
    
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)
    
    # Draw header with hyperparameters
    if hyperparams:
        draw.rectangle([(0, 0), (grid_w, header_h)], fill="#f0f0f0")
        draw.line([(0, header_h), (grid_w, header_h)], fill="#cccccc", width=2)
        
        lines = [
            "Img2Img ILVR - Start from Reference",
            f"Strength: {hyperparams.get('strength', 'N/A')}  |  "
            f"Steps: {hyperparams.get('num_inference_steps', 'N/A')}  |  "
            f"Guidance: {hyperparams.get('guidance_scale', 'N/A')}",
            f"ILVR: down_factor={hyperparams.get('down_factor', 'N/A')}, "
            f"stop_step={hyperparams.get('stop_step', 'N/A')}, "
            f"enabled={hyperparams.get('use_ilvr', 'N/A')}",
            f"Iterations: {hyperparams.get('iterations', 'N/A')}  |  "
            f"Seed: {hyperparams.get('seed', 'N/A')}  |  "
            f"Size: {hyperparams.get('width', 'N/A')}x{hyperparams.get('height', 'N/A')}",
            f"Reference: {hyperparams.get('ref_path', 'N/A')}",
        ]
        
        y_text = 10
        for i, line in enumerate(lines):
            color = "#333333" if i == 0 else "#555555"
            draw.text((20, y_text), line, fill=color)
            y_text += 28 if i == 0 else 22
    
    # Draw images with labels
    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // n_cols
        col = idx % n_cols
        x = col * cell_size
        y = header_h + row * (cell_size + label_h)
        
        img_resized = img.resize((cell_size, cell_size), Image.LANCZOS)
        grid.paste(img_resized, (x, y + label_h))
        draw.text((x + 10, y + 15), label, fill="black")
    
    grid.save(output_path)
    print(f"Saved comparison grid: {output_path}")
    
    return grid

