import cv2
import mediapipe as mp
import numpy as np
import random

# Инициализация MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Функция детекции ключевых точек
def detect_keypoints(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        raise ValueError("Не удалось обнаружить ладонь на изображении.")

    hand_landmarks = results.multi_hand_landmarks[0]
    keypoints = []
    for lm in hand_landmarks.landmark:
        x = int(lm.x * image.shape[1])
        y = int(lm.y * image.shape[0])
        keypoints.append((x, y))
    return keypoints, image

# Функция для модификации ключевых точек
def perturb_keypoints(keypoints, max_shift=5):
    perturbed = []
    for (x, y) in keypoints:
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        perturbed.append((x + shift_x, y + shift_y))
    return perturbed
