import argparse
import os
import random

import torch

from src.keypoint_detection import detect_keypoints, perturb_keypoints
from src.transformations import apply_tps_transform
from src.image_generation import generate_base_image
from src.model_setup import setup_diffusion_model

def main():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Генерация вариаций ладони.')
    parser.add_argument('output_folder', type=str, help='Папка для сохранения вариаций')
    parser.add_argument('num_variations', type=int, help='Количество вариаций')
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # Настройка устройства и модели диффузии
    device, diffusion = setup_diffusion_model()

    # Создание папки для сгенерированных изображений
    generated_folder = args.output_folder

    # Генерация базового изображения из списка валидированных сидов
    seed_list = [5, 6, 7, 8, 10, 13, 16, 17, 20, 25, 28, 29, 32, 39, 40, 45, 47, 49, 51, 53, 55, 60, 68, 72, 73, 75, 76, 80, 82, 85, 87, 88, 90, 93, 94, 95, 96, 97, 99, 100]
    base_seed = random.choice(seed_list)
    base_image_path = generate_base_image(diffusion, base_seed, generated_folder)

    # Детекция ключевых точек
    base_keypoints, base_image = detect_keypoints(base_image_path)

    # Создание вариаций
    num_variations = args.num_variations
    perturbed_keypoints_list = [perturb_keypoints(base_keypoints, max_shift=1) for _ in range(num_variations)]

    for idx, perturbed_keypoints in enumerate(perturbed_keypoints_list, start=1):
        transformed_image = apply_tps_transform(base_image, base_keypoints, perturbed_keypoints)
        variation_path = os.path.join(generated_folder, f'variation_{idx}.png')

        import cv2
        cv2.imwrite(variation_path, transformed_image)
        print(f"Вариация {idx} сохранена по пути: {variation_path}")

if __name__ == "__main__":
    main()
