import torch
from models import Generator
from torchvision.utils import save_image
import os

def generate_variations():
    nz = 100
    ngf = 128
    nc = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG = Generator(nz, ngf, nc).to(device)
    netG.load_state_dict(torch.load('./models/generator_epoch_1000.pth', map_location=device))
    netG.eval()

    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)

    fixed_noise = torch.randn(1, nz, 1, 1, device=device)

    for i in range(10):
        variation = fixed_noise + 0.02 * torch.randn_like(fixed_noise)
        fake_image = netG(variation)
        save_image(fake_image.detach(), f'{output_dir}/individual_variation_{i}.png', normalize=True)

if __name__ == '__main__':
    generate_variations()
