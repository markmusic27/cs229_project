from ilvr.config import ILVRConfig
from ilvr.pipeline import ILVRPipeline


def main():

    # Initialize with custom config
    config = ILVRConfig(
        down_factor=1,
        stop_step=50,
        num_inference_steps=100,
        guidance_scale=0,
    )
    pipeline = ILVRPipeline(config=config)

    # Load reference image
    pipeline.load_reference("dataset/universe.png")

    # Generate with comparison (returns tuple)
    image, comparison = pipeline.generate(return_comparison=True)


    # Save image
    pipeline.save(image, "output.png", save_comparison=True)


if __name__ == "__main__":
    main()
