import os
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

def setup_diffusion_model():
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Определение корневой папки проекта
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Путь к папке results в корне проекта
    results_folder = os.path.join(project_root, 'results')
    
    # Путь к файлу модели
    model_path = os.path.join(results_folder, 'model-88.pt')

    # Проверка существования файла модели
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Файл модели не найден по пути: {model_path}")

    # Загрузка сохраненного чекпойнта
    checkpoint = torch.load(model_path, map_location=device)

    # Определение модели Unet
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=3
    ).to(device)

    # Определение диффузионной модели
    diffusion = GaussianDiffusion(
        model=model,
        image_size=128,
        timesteps=1000,
        sampling_timesteps=1000,
        objective='pred_noise'
    ).to(device)

    # Извлечение состояния модели из чекпойнта
    state_dict = checkpoint['model']

    # Получение ожидаемых ключей модели Unet
    expected_keys = set(diffusion.model.state_dict().keys())

    # Фильтрация state_dict и удаление префиксов 'model.' или 'module.'
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_key = k[len('model.'):]
        elif k.startswith('module.'):
            new_key = k[len('module.'):]
        else:
            new_key = k
        if new_key in expected_keys:
            new_state_dict[new_key] = v

    # Загрузка отфильтрованного состояния в модель Unet
    missing_keys, unexpected_keys = diffusion.model.load_state_dict(new_state_dict, strict=False)

    if missing_keys:
        print("Отсутствующие ключи в state_dict модели:")
        for key in missing_keys:
            print(f"  {key}")

    if unexpected_keys:
        print("Неожиданные ключи в state_dict модели:")
        for key in unexpected_keys:
            print(f"  {key}")

    diffusion.model.to(device)

    return device, diffusion
