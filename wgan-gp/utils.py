# utils.py
import torch.nn as nn
import torch
import os
from torchvision.utils import save_image
from torch.autograd import grad

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('LayerNorm') != -1:
        nn.init.constant_(m.bias.data, 0)
        nn.init.constant_(m.weight.data, 1.0)

def save_generated_images(images, epoch, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_image(images, f'{output_dir}/generated_images_epoch_{epoch}.png', normalize=True)

def save_model(model, epoch, model_dir, model_name):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), f'{model_dir}/{model_name}_epoch_{epoch}.pth')

def gradient_penalty(critic, real, fake, device):
    batch_size, c, h, w = real.size()
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device, requires_grad=True)
    epsilon = epsilon.expand_as(real)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated = interpolated.to(device)
    interpolated.requires_grad_(True)

    mixed_scores = critic(interpolated)

    gradient = grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.size(0), -1)
    gradient_norm = gradient.norm(2, dim=1)
    gp = ((gradient_norm - 1) ** 2).mean()
    return gp
