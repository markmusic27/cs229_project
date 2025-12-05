#!/usr/bin/env python3
"""
Img2Img ILVR example usage.
"""

from img2img import Img2ImgConfig, Img2ImgPipeline


def main():
    # Configure with your hyperparameters
    config = Img2ImgConfig(
        strength=0.3,
        guidance_scale=3.0,
        num_inference_steps=100,
        iterations=5,
        use_ilvr=False,  # Matches img2img.py default (no --use_ilvr flag)
        down_factor=4,
        stop_step=50,
    )

    # Create pipeline and load reference
    pipeline = Img2ImgPipeline(config=config)
    pipeline.load_reference("dataset/camel_comp.png")

    # Generate iterations (saves to output dir)
    images = pipeline.generate_iterations(output_dir="img2img_results")


if __name__ == "__main__":
    main()
