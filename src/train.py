import os
import torch
from model import create_model
from trainer import create_trainer

def main():
    # Устройство (GPU или CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Путь к проекту
    project_path = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_path, 'data')  # Папка с данными
    results_folder = os.path.join(project_path, 'results')  # Папка для результатов

    # Создание модели
    diffusion = create_model(device)

    # Создание тренера
    trainer = create_trainer(diffusion, data_dir, results_folder)

    # Запуск обучения
    trainer.train()

if __name__ == "__main__":
    main()