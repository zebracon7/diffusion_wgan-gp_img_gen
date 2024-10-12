import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataloader(data_root, image_size, batch_size, nc):
    # Трансформации с аугментацией
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),          # Изменение размера
        #transforms.RandomHorizontalFlip(p=0.5),               # Случайное горизонтальное отражение
        transforms.RandomRotation(degrees=20),                # Случайное вращение на ±20 градусов
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Изменение яркости, контраста, насыщенности и оттенка
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Случайные масштабирование и сдвиг
        transforms.ToTensor(),                                # Преобразование в тензор
        transforms.Normalize([0.5] * nc, [0.5] * nc)          # Нормализация для каждого канала
    ])

    # Создание датасета и загрузчика данных
    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    return dataloader
