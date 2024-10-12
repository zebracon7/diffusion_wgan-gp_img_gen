import cv2
import numpy as np
from scipy.interpolate import Rbf

# Функция для применения Thin Plate Splines (TPS) преобразования
def apply_tps_transform(image, original_keypoints, perturbed_keypoints):
    original = np.array(original_keypoints, dtype=np.float32)
    perturbed = np.array(perturbed_keypoints, dtype=np.float32)

    # Создаём RBF (Radial Basis Function) для интерполяции деформаций
    rbf_x = Rbf(original[:,0], original[:,1], perturbed[:,0], function='thin_plate')
    rbf_y = Rbf(original[:,0], original[:,1], perturbed[:,1], function='thin_plate')

    height, width = image.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_x_flat = grid_x.flatten()
    grid_y_flat = grid_y.flatten()

    # Вычисляем деформации
    map_x = rbf_x(grid_x_flat, grid_y_flat).reshape((height, width)).astype(np.float32)
    map_y = rbf_y(grid_x_flat, grid_y_flat).reshape((height, width)).astype(np.float32)

    # Ограничиваем значения координат
    map_x = np.clip(map_x, 0, width-1)
    map_y = np.clip(map_y, 0, height-1)

    # Применяем деформацию
    transformed_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return transformed_image
