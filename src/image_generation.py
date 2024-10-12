import os
import torch
from torchvision.utils import save_image

def generate_base_image(diffusion, base_seed, generated_folder):
    torch.manual_seed(base_seed)
    base_sample = diffusion.sample(batch_size=1)
    base_image_path = os.path.join(generated_folder, 'base_palm.png')
    save_image(base_sample, base_image_path, nrow=1, normalize=True)
    print(f"Базовое изображение сохранено по пути: {base_image_path}")
    return base_image_path
