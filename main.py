#!/usr/bin/env python3
"""
Img2Img without CLS conditioning (basic usage).
"""

from img2img import Img2ImgConfig, Img2ImgPipeline


def main():
    # Configure (no CLS - use_cls_conditioning defaults to False)
    config = Img2ImgConfig(
        strength=0.3,
        guidance_scale=3.0,
        num_inference_steps=100,
        iterations=5,
    )

    # Create pipeline and load reference image
    pipeline = Img2ImgPipeline(config=config)
    pipeline.load_reference("dataset/camel_comp.png")

    # Generate iterations (no cls_embedding needed)
    images = pipeline.generate_iterations(output_dir="img2img_results")


if __name__ == "__main__":
    main()
