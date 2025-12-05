"""
Image preprocessing utilities.
"""

from PIL import Image


def prepare_image(
    image: Image.Image,
    target_width: int,
    target_height: int,
) -> Image.Image:
    """
    Prepare image to target size with smart cropping.
    
    - If larger: center crop to target aspect ratio, then resize
    - If smaller: resize directly to fit
    
    Args:
        image: Input PIL Image.
        target_width: Desired output width.
        target_height: Desired output height.
    
    Returns:
        Resized and cropped PIL Image.
    """
    img_width, img_height = image.size
    target_ratio = target_width / target_height
    img_ratio = img_width / img_height
    
    # Center crop if image is larger than target in both dimensions
    if img_width >= target_width and img_height >= target_height:
        if img_ratio > target_ratio:
            # Image is wider - crop width
            new_width = int(img_height * target_ratio)
            left = (img_width - new_width) // 2
            image = image.crop((left, 0, left + new_width, img_height))
        elif img_ratio < target_ratio:
            # Image is taller - crop height
            new_height = int(img_width / target_ratio)
            top = (img_height - new_height) // 2
            image = image.crop((0, top, img_width, top + new_height))
    
    # Resize to target dimensions
    image = image.resize((target_width, target_height), Image.LANCZOS)
    return image

