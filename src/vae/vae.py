from PIL import Image
import torch
import numpy as np
from diffusers import AutoencoderKL
from typing import Tuple, Optional

# This is the class that contains functions related to the vae, including the encoder, the decoder, and the packing of bits for quantization

class VAE:
    def __init__(self, model = None, quant: int = 3, model_dims: Tuple[int, int] = (512, 512), device: str = None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.quant = quant
        self.model_dims = model_dims
        self.scaling_factor = 0.18215

        if model is None: 
            self.model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(self.device).eval()
        else: 
            self.model = model.to(self.device).eval()

    # loads image and returns loaded image_tensor and original size of image dimensions for future reference 
    def load_image(self, path_to_image: str):
        image = Image.open(path_to_image).convert("RGB")
        original_size = image.size
        image_resized = image.resize(self.model_dims, Image.LANCZOS) # Resize to VAE required size 

        image_tensor = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        image_tensor = (image_tensor * 2 - 1).to(self.device)  

        return image_tensor, original_size

    # Encode with quantization 
    def encode_with_quant(self, image_tensor):
        with torch.no_grad():
            
            latent_dist = self.model.encode(image_tensor)
            latents = latent_dist.latent_dist.sample() * self.scaling_factor
            
            latent_min = latents.min()
            latent_max = latents.max()
            
            
            max_val = (2 ** self.quant) - 1
            latents_quant = ((latents - latent_min) / (latent_max - latent_min) * max_val).round().to(torch.uint8)

            return latents_quant, latent_min, latent_max

    # Encode without quantization 
    def encode(self, image_tensor):
        with torch.no_grad():
       
            latent_dist = self.model.encode(image_tensor)
            latents = latent_dist.latent_dist.sample() * self.scaling_factor
        return latents

    def pack_3bit(self,data):
        flat = data.flatten().cpu().numpy()
        
        # Pad to multiple of 8
        pad_len = (8 - len(flat) % 8) % 8
        if pad_len:
            flat = np.pad(flat, (0, pad_len), mode='constant')
        
        # Pack 8 values into 3 bytes each
        packed = []
        for i in range(0, len(flat), 8):
            v = flat[i:i+8]
            # Pack 8 3-bit values into 3 bytes (24 bits)
            b0 = (v[0] << 5) | (v[1] << 2) | (v[2] >> 1)
            b1 = ((v[2] & 1) << 7) | (v[3] << 4) | (v[4] << 1) | (v[5] >> 2)
            b2 = ((v[5] & 3) << 6) | (v[6] << 3) | v[7]
            packed.extend([b0, b1, b2])
        
        return np.array(packed, dtype=np.uint8)

    def unpack_3bit(self, packed, original_shape):
        """Unpack 3-bit values: 3 bytes → 8 values"""
        unpacked = []
        
        for i in range(0, len(packed), 3):
            b0, b1, b2 = packed[i], packed[i+1], packed[i+2]
            # Unpack 3 bytes into 8 3-bit values
            v0 = (b0 >> 5) & 0x7
            v1 = (b0 >> 2) & 0x7
            v2 = ((b0 & 0x3) << 1) | ((b1 >> 7) & 0x1)
            v3 = (b1 >> 4) & 0x7
            v4 = (b1 >> 1) & 0x7
            v5 = ((b1 & 0x1) << 2) | ((b2 >> 6) & 0x3)
            v6 = (b2 >> 3) & 0x7
            v7 = b2 & 0x7
            unpacked.extend([v0, v1, v2, v3, v4, v5, v6, v7])
        
        # Trim to original size and reshape
        total_elements = np.prod(original_shape)
        unpacked = np.array(unpacked[:total_elements], dtype=np.uint8)
        return torch.from_numpy(unpacked).reshape(original_shape)

    def encode_and_pack(self, image_tensor):
        """Encode with quantization and pack to 3-bit"""
        latents_quant, latent_min, latent_max = self.encode_with_quant(image_tensor)
        original_shape = latents_quant.shape
        packed = self.pack_3bit(latents_quant)
        return packed, latent_min, latent_max, original_shape

    def unpack_and_decode(self, packed, original_shape, latent_min, latent_max, original_size, path_to_save):
        """Unpack and decode"""
        latents_quant = self.unpack_3bit(packed, original_shape).to(self.device)
        self.decode_and_save(latents_quant, quant=True, path_to_save=path_to_save, 
                            original_size=original_size, latent_max=latent_max, latent_min=latent_min)

    def decode_and_save(self, latents, quant: bool, path_to_save: str, original_size, latent_max: Optional[float] = None, latent_min: Optional[float] = None):
        # Dequantize if needed
        if quant:
            max_val = (2 ** self.quant) - 1
            latents = latents.float() / max_val * (latent_max - latent_min) + latent_min

        with torch.no_grad():
            recon = self.model.decode(latents / self.scaling_factor).sample
            recon = (recon + 1) / 2 
            recon = recon.clamp(0, 1)

            recon_np = recon.squeeze().permute(1, 2, 0).cpu().numpy()
            recon_np = (recon_np * 255).astype(np.uint8)
            output_image = Image.fromarray(recon_np)
            output_image = output_image.resize(original_size, Image.LANCZOS)
            output_image.save(path_to_save, quality=95)

            print(f"Saved: {path_to_save}")