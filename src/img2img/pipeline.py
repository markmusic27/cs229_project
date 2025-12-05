"""
Main Img2Img ILVR Pipeline.
"""

import os
from typing import Optional, Union, List, Tuple
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline, DDIMScheduler

from .config import Img2ImgConfig
from .preprocessing import prepare_image
from .encoding import encode_image
from .callback import ILVRCallback
from .visualization import create_comparison_grid


class Img2ImgPipeline:
    """
    Img2Img pipeline with optional ILVR guidance.
    
    Starts diffusion from a reference image instead of pure noise,
    allowing iterative refinement while preserving structure.
    
    Example:
        pipeline = Img2ImgPipeline()
        pipeline.load_reference("photo.jpg")
        
        # Single generation
        image = pipeline.generate()
        
        # Iterative refinement
        images = pipeline.generate_iterations()
    """
    
    def __init__(
        self,
        config: Optional[Img2ImgConfig] = None,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        load_model: bool = True,
    ):
        """
        Initialize pipeline.
        
        Args:
            config: Generation configuration. Uses defaults if None.
            model_id: HuggingFace model ID.
            device: Device ('cuda' or 'cpu'). Auto-detects if None.
            dtype: Data type. Uses fp16 on CUDA, fp32 on CPU if None.
            load_model: Whether to load model immediately.
        """
        self.config = config or Img2ImgConfig()
        self.model_id = model_id
        
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        
        self.pipe: Optional[StableDiffusionXLImg2ImgPipeline] = None
        self.reference_image: Optional[Image.Image] = None
        
        if load_model:
            self.load_model()
    
    def load_model(self) -> "Img2ImgPipeline":
        """Load the SDXL Img2Img model."""
        print(f"Loading SDXL Img2Img on {self.device} ({self.dtype})...")
        
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            use_safetensors=True,
        ).to(self.device)
        
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory optimization")
        except Exception:
            pass
        
        print("Model loaded successfully")
        return self
    
    def load_reference(self, image: Union[str, Path, Image.Image]) -> "Img2ImgPipeline":
        """
        Load a reference image.
        
        Args:
            image: Path to image or PIL Image.
        
        Returns:
            Self for chaining.
        """
        if isinstance(image, (str, Path)):
            print(f"Loading reference: {image}")
            image = Image.open(image).convert("RGB")
        
        orig_size = image.size
        image = prepare_image(image, self.config.width, self.config.height)
        print(f"  {orig_size[0]}x{orig_size[1]} -> {self.config.width}x{self.config.height}")
        
        self.reference_image = image
        return self
    
    def generate(
        self,
        config: Optional[Img2ImgConfig] = None,
        image: Optional[Image.Image] = None,
        seed_offset: int = 0,
    ) -> Image.Image:
        """
        Generate a single image.
        
        Args:
            config: Override config for this generation.
            image: Input image. Uses reference if None.
            seed_offset: Offset to add to seed.
        
        Returns:
            Generated PIL Image.
        """
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        cfg = config or self.config
        input_image = image or self.reference_image
        
        if input_image is None:
            raise RuntimeError("No input image. Call load_reference() or provide image.")
        
        # Setup generator
        seed = cfg.seed + seed_offset
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Setup ILVR callback if enabled
        callback = None
        callback_inputs = None
        
        if cfg.use_ilvr and cfg.down_factor > 1:
            y0_latent = encode_image(self.pipe, input_image, self.device, self.dtype)
            noise = torch.randn(y0_latent.shape, device=self.device, dtype=self.dtype, generator=generator)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            callback = ILVRCallback(
                y0_latent=y0_latent,
                noise=noise,
                scheduler=self.pipe.scheduler,
                down_factor=cfg.down_factor,
                stop_step=cfg.stop_step,
            )
            callback_inputs = ["latents"]
        
        print(f"Generating: strength={cfg.strength}, steps={cfg.num_inference_steps}, "
              f"guidance={cfg.guidance_scale}, ilvr={cfg.use_ilvr}")
        
        result = self.pipe(
            prompt=cfg.prompt,
            negative_prompt=cfg.negative_prompt,
            image=input_image,
            strength=cfg.strength,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            generator=generator,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=callback_inputs,
        )
        
        return result.images[0]
    
    def generate_iterations(
        self,
        config: Optional[Img2ImgConfig] = None,
        output_dir: Optional[str] = None,
    ) -> List[Image.Image]:
        """
        Generate multiple iterations, using each output as next input.
        
        Args:
            config: Override config for this generation.
            output_dir: Directory to save intermediate results.
        
        Returns:
            List of all images (including original reference).
        """
        if self.reference_image is None:
            raise RuntimeError("No reference loaded. Call load_reference() first.")
        
        cfg = config or self.config
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Start with reference
        all_images = [self.reference_image.copy()]
        all_labels = ["Original"]
        current_image = self.reference_image
        
        if output_dir:
            path = os.path.join(output_dir, "iter_0_original.png")
            current_image.save(path)
            print(f"Saved: iter_0_original.png")
        
        # Run iterations
        for i in range(1, cfg.iterations + 1):
            print(f"\n{'='*60}")
            print(f"Iteration {i}/{cfg.iterations}")
            print(f"{'='*60}")
            
            output = self.generate(config=cfg, image=current_image, seed_offset=i)
            
            all_images.append(output.copy())
            all_labels.append(f"Iteration {i}")
            
            if output_dir:
                path = os.path.join(output_dir, f"iter_{i}.png")
                output.save(path)
                print(f"Saved: iter_{i}.png")
            
            current_image = output
        
        # Create comparison grid
        if output_dir:
            self._save_comparison(all_images, all_labels, output_dir, cfg)
        
        return all_images
    
    def _save_comparison(
        self,
        images: List[Image.Image],
        labels: List[str],
        output_dir: str,
        config: Img2ImgConfig,
    ) -> None:
        """Save comparison grid with hyperparameters."""
        print("\n" + "="*60)
        print("Creating comparison grid...")
        print("="*60)
        
        hyperparams = config.to_dict()
        hyperparams["ref_path"] = "reference"
        
        path = os.path.join(output_dir, "comparison.png")
        create_comparison_grid(images, labels, path, hyperparams)
        
        print(f"\n✓ All {len(images)} images saved to {output_dir}/")

