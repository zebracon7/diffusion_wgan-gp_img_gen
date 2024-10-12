# models.py
import torch
import torch.nn as nn

# Генератор для 256x256
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Входное: Z, выход: ngf*16 x 4 x 4
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # ngf*16 x 4 x 4 -> ngf*8 x 8 x 8
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # ngf*8 x 8 x 8 -> ngf*4 x 16 x 16
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # ngf*4 x 16 x 16 -> ngf*2 x 32 x 32
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # ngf*2 x 32 x 32 -> ngf x 64 x 64
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # ngf x 64 x 64 -> ngf//2 x 128 x 128
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            # ngf//2 x 128 x 128 -> nc x 256 x 256
            nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Дискриминатор для 256x256
class Discriminator(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Входное: nc x 256 x 256 -> ndf x 128 x 128
            nn.Conv2d(nc, ndf // 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf//2 x 128 x 128 -> ndf x 64 x 64
            nn.Conv2d(ndf // 2, ndf, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf x 64 x 64 -> ndf*2 x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf*2 x 32 x 32 -> ndf*4 x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf*4 x 16 x 16 -> ndf*8 x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf*8 x 8 x 8 -> ndf*16 x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf*16 x 4 x 4 -> 1 x 1 x 1
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            # Без функции активации, так как используется WGAN-GP
        )

    def forward(self, input):
        return self.main(input).view(-1)
