import torch
from models import Generator, Discriminator
from data_loader import get_dataloader
from utils import weights_init
from train import train_model
import os

def main():
    # Гиперпараметры
    nz = 100         # Размерность входного шума
    ngf = 128         # Размерность функций в генераторе
    ndf = 128         # Размерность функций в дискриминаторе
    nc = 3           # Количество каналов в изображении (RGB)
    image_size = 256 # Размер изображения
    batch_size = 16
    num_epochs = 1000
    lr_G = 0.00005
    lr_D = 0.00002
    beta1 = 0.5      # Бета1 для оптимизатора Adam (для RMSprop не требуется)

    # Пути
    data_root = '/content/drive/MyDrive/Biometric_hack/data'  # Измените на ваш путь
    output_dir = '/content/drive/MyDrive/Biometric_hack/output'
    model_dir = '/content/drive/MyDrive/Biometric_hack/models'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Создание директорий для выходных данных и моделей
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Загрузка данных
    dataloader = get_dataloader(data_root, image_size, batch_size, nc)

    # Инициализация моделей
    netG = Generator(nz, ngf, nc).to(device)
    netG.apply(weights_init)

    netD = Discriminator(ndf, nc).to(device)
    netD.apply(weights_init)
    
    # Инициализация моделей
    #netG = Generator(nz, ngf, nc).to(device)
    #netD = Discriminator(ndf, nc).to(device)

    # Загрузка сохраненных весов
    #netG.load_state_dict(torch.load('/content/drive/MyDrive/Biometric_hack/models/generator_epoch_170.pth'))
    #netD.load_state_dict(torch.load('/content/drive/MyDrive/Biometric_hack/models/discriminator_epoch_170.pth'))


    # Обучение модели
    train_model(dataloader, netG, netD, device, num_epochs, nz, lr_G, lr_D, output_dir, model_dir)

if __name__ == '__main__':
    main()
