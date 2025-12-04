import torch
from diffusers import StableDiffusionXLPipeline


def main():
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)

    def progress_callback(pipe, step, timestep, callback_kwargs):
        print(f"Step {step + 1}/25 (timestep: {timestep:.0f})")
        return callback_kwargs

    prompt = "generate a realistic picture of a woman showing her face. in sunshine"
    image = pipe(
        prompt,
        num_inference_steps=25,
        guidance_scale=7.5,
        callback_on_step_end=progress_callback,
    ).images[0]
    image.save("sdxl_test.png")
    print("Saved sdxl_test.png")


if __name__ == "__main__":
    main()
