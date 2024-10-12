import torch
import torch.optim as optim
from tqdm import tqdm
from utils import save_generated_images, save_model, gradient_penalty

def train_model(dataloader, netG, netD, device, num_epochs, nz, lr_G, lr_D, output_dir, model_dir):
    # Оптимизаторы
    optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    critic_iterations = 1  
    lambda_gp = 10

    for epoch in range(1, num_epochs + 1):
        for i, data in enumerate(tqdm(dataloader), 0):
            real_images = data[0].to(device)
            b_size = real_images.size(0)

            # Обучение дискриминатора
            netD.zero_grad()
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_images = netG(noise).detach()
            real_scores = netD(real_images)
            fake_scores = netD(fake_images)
            gp = gradient_penalty(netD, real_images, fake_images, device)
            lossD = -(torch.mean(real_scores) - torch.mean(fake_scores)) + lambda_gp * gp
            lossD.backward()
            optimizerD.step()


            # Обучение генератора
            netG.zero_grad()
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_images = netG(noise)
            fake_scores = netD(fake_images)
            lossG = -torch.mean(fake_scores)
            lossG.backward()
            optimizerG.step()

            # Вывод статистики
            if i % 100 == 0:
                print(f'Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}]\t'
                      f'Loss_D: {lossD.item():.4f}\tLoss_G: {lossG.item():.4f}')

        # Сохранение сгенерированных изображений и моделей каждые 10 эпох
        if epoch % 10 == 0 or epoch == num_epochs:
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            save_generated_images(fake, epoch, output_dir)
            save_model(netG, epoch, model_dir, 'generator')
            save_model(netD, epoch, model_dir, 'discriminator')
            
