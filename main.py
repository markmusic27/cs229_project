#!/usr/bin/env python3
"""
Simple VAE compression cycle.
Usage: python main.py <input_image> <output_image>
"""

import sys
from vae import VAE


def compress_image(input_path: str, output_path: str):
    """Run a full VAE compression cycle: encode → quantize → pack → unpack → decode"""
    vae = VAE()
    image_tensor, original_size = vae.load_image(input_path)
    packed, latent_min, latent_max, original_shape = vae.encode_and_pack(image_tensor)
    vae.unpack_and_decode(packed, original_shape, latent_min, latent_max, original_size, output_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <input_image> <output_image>")
        print("Example: python main.py photo.jpg compressed_output.png")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    compress_image(input_path, output_path)
