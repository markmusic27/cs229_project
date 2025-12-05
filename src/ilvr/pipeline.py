"""
Main ILVR Pipeline class.
"""

import os
from typing import Optional, List, Union, Dict, Tuple
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from diffusers import StableDiffusionXLPipeline, DDIMScheduler

from .config import ILVRConfig
from .callback import ILVRCallback
from .encoding import encode_image_safe


class ILVRPipeline:
    """
    ILVR Pipeline for SDXL.
    
    Provides a clean interface for generating images with ILVR conditioning.
    
    Example:
        # Basic usage
        pipeline = ILVRPipeline()
        pipeline.load_reference("photo.jpg")
        image = pipeline.generate()
        
        # With custom config
        config = ILVRConfig(down_factor=2, stop_step=30)
        pipeline = ILVRPipeline(config=config)
        pipeline.load_reference("photo.jpg")
        image = pipeline.generate()
        
        # Update config between generations
        pipeline.config.update(down_factor=4, stop_step=50)
        image = pipeline.generate()
        
        # Run parameter sweep
        results = pipeline.sweep(steps_list=[100, 150, 200])
    """
    
    def __init__(
        self,
        config: Optional[ILVRConfig] = None,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        load_model: bool = True,
    ):
        """
        Initialize ILVR Pipeline.
        
        Args:
            config: ILVR configuration. Uses defaults if None.
            model_id: HuggingFace model ID for SDXL.
            device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
            dtype: Data type. Uses fp16 on CUDA, fp32 on CPU if None.
            load_model: Whether to load the model immediately.
        """
        self.config = config or ILVRConfig()
        self.model_id = model_id
        
        # Setup device and dtype
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        
        # Pipeline and reference state
        self.pipe: Optional[StableDiffusionXLPipeline] = None
        self.reference_image: Optional[Image.Image] = None
        self.reference_latent: Optional[torch.Tensor] = None
        
        if load_model:
            self.load_model()
    
    def load_model(self) -> "ILVRPipeline":
        """
        Load the SDXL model.
        
        Returns:
            Self for chaining.
        """
        print(f"Loading SDXL on {self.device} ({self.dtype})...")
        
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            use_safetensors=True,
        ).to(self.device)
        
        # Use DDIM scheduler for stable ILVR
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        # Enable memory optimizations
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory optimization")
        except Exception:
            pass
        
        print("Model loaded successfully")
        return self
    
    def load_reference(self, image: Union[str, Path, Image.Image]) -> "ILVRPipeline":
        """
        Load and encode a reference image.
        
        Args:
            image: Path to image or PIL Image.
        
        Returns:
            Self for chaining.
        """
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Load image if path
        if isinstance(image, (str, Path)):
            print(f"Loading reference: {image}")
            image = Image.open(image).convert("RGB")
        
        # Resize to target dimensions
        image = image.resize((self.config.width, self.config.height), Image.LANCZOS)
        self.reference_image = image
        
        # Encode to latent
        print("Encoding reference to latent space...")
        self.reference_latent = encode_image_safe(self.pipe, image, self.device, self.dtype)
        
        print(f"Reference latent shape: {self.reference_latent.shape}")
        print(f"Reference latent range: [{self.reference_latent.min():.2f}, {self.reference_latent.max():.2f}]")
        
        return self
    
    def generate(
        self,
        config: Optional[ILVRConfig] = None,
        return_comparison: bool = False,
    ) -> Union[Image.Image, Tuple[Image.Image, Image.Image]]:
        """
        Generate an image with ILVR conditioning.
        
        Args:
            config: Override config for this generation. Uses self.config if None.
            return_comparison: If True, returns (output, comparison) tuple.
        
        Returns:
            Generated PIL Image, or (output, comparison) if return_comparison=True.
        """
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self.reference_latent is None:
            raise RuntimeError("No reference loaded. Call load_reference() first.")
        
        cfg = config or self.config
        
        # Setup scheduler
        self.pipe.scheduler.set_timesteps(cfg.num_inference_steps, device=self.device)
        
        # Create noise for reference
        generator = torch.Generator(device=self.device).manual_seed(cfg.seed)
        ref_noise = torch.randn(
            self.reference_latent.shape,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        
        # Reset generator for main generation
        generator = torch.Generator(device=self.device).manual_seed(cfg.seed)
        
        # Create callback
        callback = ILVRCallback(
            y0_latent=self.reference_latent,
            noise=ref_noise,
            scheduler=self.pipe.scheduler,
            down_factor=cfg.down_factor,
            stop_step=cfg.stop_step,
        )
        
        print(f"Generating: steps={cfg.num_inference_steps}, stop={cfg.stop_step}, "
              f"N={cfg.down_factor}, guidance={cfg.guidance_scale}")
        
        # Generate
        result = self.pipe(
            prompt=cfg.prompt,
            negative_prompt=cfg.negative_prompt,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            height=cfg.height,
            width=cfg.width,
            generator=generator,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )
        
        output = result.images[0]
        
        if return_comparison:
            comparison = Image.new("RGB", (cfg.width * 2, cfg.height))
            comparison.paste(self.reference_image, (0, 0))
            comparison.paste(output, (cfg.width, 0))
            return output, comparison
        
        return output
    
    def sweep(
        self,
        steps_list: Optional[List[int]] = None,
        stop_steps: Optional[List[int]] = None,
        down_factors: Optional[List[int]] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[Tuple[int, int, int], Image.Image]:
        """
        Run a parameter sweep.
        
        Args:
            steps_list: List of total steps to test.
            stop_steps: List of stop_step values to test.
            down_factors: List of down_factor values to test.
            output_dir: Directory to save results. If None, doesn't save.
        
        Returns:
            Dict mapping (steps, stop_step, down_factor) tuple to generated image.
        """
        steps_list = steps_list or [self.config.num_inference_steps]
        stop_steps = stop_steps or [self.config.stop_step]
        down_factors = down_factors or [self.config.down_factor]
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        total = len(steps_list) * len(stop_steps) * len(down_factors)
        count = 0
        
        for steps in steps_list:
            for stop in stop_steps:
                for N in down_factors:
                    count += 1
                    print(f"\n{'='*50}")
                    print(f"[{count}/{total}] steps={steps}, stop={stop}, N={N}")
                    print(f"{'='*50}")
                    
                    cfg = self.config.copy(
                        num_inference_steps=steps,
                        stop_step=stop,
                        down_factor=N,
                    )
                    
                    output = self.generate(config=cfg)
                    results[(steps, stop, N)] = output
                    
                    if output_dir:
                        filename = f"steps{steps}_stop{stop}_N{N}.png"
                        output.save(os.path.join(output_dir, filename))
                        print(f"Saved: {filename}")
        
        # Create comparison grid if saving
        if output_dir:
            grid = self._create_grid(results)
            grid.save(os.path.join(output_dir, "comparison_grid.png"))
            print(f"\nSaved comparison grid to {output_dir}/comparison_grid.png")
        
        return results
    
    def _create_grid(
        self,
        results: Dict[Tuple[int, int, int], Image.Image],
        cell_size: int = 450,
        label_height: int = 40,
    ) -> Image.Image:
        """Create a comparison grid from results."""
        images = [self.reference_image] + list(results.values())
        labels = ["Reference"] + [f"s={k[0]},stop={k[1]},N={k[2]}" for k in results.keys()]
        
        n_images = len(images)
        cols = min(4, n_images)
        rows = (n_images + cols - 1) // cols
        
        grid = Image.new("RGB", (cell_size * cols, (cell_size + label_height) * rows), "white")
        draw = ImageDraw.Draw(grid)
        
        for idx, (img, label) in enumerate(zip(images, labels)):
            row = idx // cols
            col = idx % cols
            x = col * cell_size
            y = row * (cell_size + label_height)
            
            img_resized = img.resize((cell_size, cell_size), Image.LANCZOS)
            grid.paste(img_resized, (x, y + label_height))
            draw.text((x + 10, y + 10), label, fill="black")
        
        return grid
    
    def save(
        self,
        image: Image.Image,
        path: str,
        save_comparison: bool = True,
    ) -> None:
        """
        Save an image and optionally a comparison with reference.
        
        Args:
            image: Image to save.
            path: Output path.
            save_comparison: Whether to also save a side-by-side comparison.
        """
        image.save(path)
        print(f"Saved: {path}")
        
        if save_comparison and self.reference_image:
            comparison = Image.new("RGB", (self.config.width * 2, self.config.height))
            comparison.paste(self.reference_image, (0, 0))
            comparison.paste(image, (self.config.width, 0))
            
            comp_path = path.replace(".png", "_comparison.png")
            comparison.save(comp_path)
            print(f"Saved: {comp_path}")