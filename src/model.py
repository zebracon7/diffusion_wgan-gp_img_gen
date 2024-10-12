import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

def create_model(device):
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=3
    ).to(device)

    diffusion = GaussianDiffusion(
        model=model,
        image_size=128,
        timesteps=1000,
        sampling_timesteps=1000,
        objective='pred_noise'
    ).to(device)

    return diffusion