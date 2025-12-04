import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from diffusers import StableDiffusionXLPipeline


# ------------------------------
# ILVR utilities
# ------------------------------

def lowpass_phi(x: torch.Tensor, down_factor: int) -> torch.Tensor:
    """
    Approximate low-pass filter by downsampling then upsampling.

    x: [B, C, H, W]
    down_factor: integer >= 1
    """
    if down_factor <= 1:
        return x

    b, c, h, w = x.shape
    h_ds = max(1, h // down_factor)
    w_ds = max(1, w // down_factor)

    x_ds = F.interpolate(
        x,
        size=(h_ds, w_ds),
        mode="bicubic",
        align_corners=False,
    )
    x_us = F.interpolate(
        x_ds,
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    )
    return x_us


def encode_reference_to_latent(pipe: StableDiffusionXLPipeline,
                               ref_image: Image.Image,
                               device: str) -> torch.Tensor:
    """
    Encode a reference image into SDXL's latent space, matching the pipeline's conventions.
    """
    # Preprocess to [-1, 1], add batch dim
    ref_tensor = pipe.image_processor.preprocess(ref_image).to(
        device=device,
        dtype=pipe.unet.dtype,
    )  # [1, 3, H, W]

    with torch.no_grad():
        posterior = pipe.vae.encode(ref_tensor)
        # diffusers VAE returns a DiagonalGaussianDistribution
        if hasattr(posterior, "latent_dist"):
            latents = posterior.latent_dist.sample()
        else:
            latents = posterior.sample()
        latents = latents * pipe.vae.config.scaling_factor  # important!

    return latents  # [1, 4, H/8, W/8]


def make_ilvr_callback(pipe: StableDiffusionXLPipeline,
                       y0_latent: torch.Tensor,
                       ref_noise: torch.Tensor,
                       down_factor: int,
                       ilvr_steps: int | None):
    """
    Build a callback that applies ILVR on SDXL latents at each step.

    This uses the scheduler's add_noise() as the forward process q(y_t | y_0),
    and replaces low-frequency content of latents with that of y_t.
    """

    scheduler = pipe.scheduler
    device = y0_latent.device

    if ilvr_steps is None:
        # apply ILVR for all steps
        ilvr_steps = scheduler.num_inference_steps

    def callback(pipe_inner, step_index: int, timestep: int, callback_kwargs: dict):
        latents = callback_kwargs["latents"]  # [B, 4, H/8, W/8]

        # Only apply ILVR for the first ilvr_steps steps (from noisy towards clean)
        if step_index < ilvr_steps:
            # timestep is an int or tensor; convert to tensor on correct device
            if torch.is_tensor(timestep):
                t = timestep.to(device=device, dtype=torch.long)
            else:
                t = torch.tensor([timestep], device=device, dtype=torch.long)

            # Forward-diffuse the reference latent y0 to y_t
            # Scheduler implements q(x_t | x_0) via add_noise
            y_t = scheduler.add_noise(y0_latent, ref_noise, t)  # [1, 4, H/8, W/8]

            # Match batch size if needed (typically B=1 anyway)
            if y_t.shape[0] != latents.shape[0]:
                y_t = y_t.expand(latents.shape[0], -1, -1, -1)

            # ILVR low-pass replacement:
            # x_{t-1} = φ_N(y_t) + (x_{t-1}' - φ_N(x_{t-1}'))
            phi_y = lowpass_phi(y_t, down_factor)
            phi_x = lowpass_phi(latents, down_factor)
            latents = phi_y + (latents - phi_x)

        callback_kwargs["latents"] = latents
        return callback_kwargs

    return callback


# ------------------------------
# Main ILVR sampling wrapper
# ------------------------------

def run_sdxl_ilvr(
    prompt: str,
    ref_path: str,
    negative_prompt: str = "",
    num_inference_steps: int = 30,
    guidance_scale: float = 5.0,
    down_factor: int = 16,
    ilvr_steps: int | None = None,
    height: int = 1024,
    width: int = 1024,
    seed: int = 42,
):
    """
    Run SDXL normally and with ILVR, saving both images.

    - down_factor: ILVR low-pass factor (larger = more coarse similarity).
    - ilvr_steps: number of denoising steps to apply ILVR.
                  If None, apply ILVR for all steps.
    """

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("Loading SDXL base...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)

    # Optional for memory/speed on A100
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    # ------------------------
    # 1) Plain SDXL sample
    # ------------------------
    print("Generating plain SDXL image...")
    gen_plain = torch.Generator(device=device).manual_seed(seed)

    plain_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=gen_plain,
    ).images[0]

    plain_image.save("sdxl_plain.png")
    print("Saved sdxl_plain.png")

    # ------------------------
    # 2) ILVR-conditioned SDXL
    # ------------------------

    # Load reference image and resize to target resolution
    ref_image = Image.open(ref_path).convert("RGB")
    ref_image = ref_image.resize((width, height), Image.BICUBIC)

    # Encode reference to latent
    print("Encoding reference image to latent...")
    y0_latent = encode_reference_to_latent(pipe, ref_image, device=device)

    # Use the same generator for ILVR so that ref noise and base noise are aligned
    gen_ilvr = torch.Generator(device=device).manual_seed(seed)
    ref_noise = torch.randn_like(y0_latent, generator=gen_ilvr)

    ilvr_callback = make_ilvr_callback(
        pipe=pipe,
        y0_latent=y0_latent,
        ref_noise=ref_noise,
        down_factor=down_factor,
        ilvr_steps=ilvr_steps,
    )

    print("Generating ILVR SDXL image...")
    # SDXL callback passes a dict with at least "latents" when we request it.
    ilvr_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=gen_ilvr,
        callback_on_step_end=ilvr_callback,
        callback_on_step_end_tensor_inputs=["latents"],
    ).images[0]

    ilvr_image.save("sdxl_ilvr.png")
    print("Saved sdxl_ilvr.png")


# ------------------------------
# CLI entrypoint
# ------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str,
                        default="a cinematic portrait, 35mm photo, shallow depth of field")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--ref", type=str, required=True, help="Path to reference image")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--down_factor", type=int, default=16)
    parser.add_argument("--ilvr_steps", type=int, default=None)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_sdxl_ilvr(
        prompt=args.prompt,
        ref_path=args.ref,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        down_factor=args.down_factor,
        ilvr_steps=args.ilvr_steps,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
