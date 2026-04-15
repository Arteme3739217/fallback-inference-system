"""Функции предобработки изображения для инференса."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from app.model_inference.config import IMAGE_SIZE, resolve_project_path


def load_image(image_path: str | Path) -> np.ndarray:
    """Загрузить изображение в оттенках серого."""
    path = resolve_project_path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image was not found: {image_path}")

    image_buffer = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(image_buffer, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image was not found: {image_path}")
    return image


def resize_image(image: np.ndarray, image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Изменить размер изображения до целевого."""
    return cv2.resize(image, image_size)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Нормализовать пиксели в диапазон [0, 1]."""
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.max() > 1.0:
        image_array = image_array / 255.0
    return image_array


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Преобразовать изображение в тензор батча с тремя каналами."""
    image_array = normalize_image(image)
    image_array = np.stack([image_array, image_array, image_array], axis=0)
    return torch.from_numpy(image_array).float().unsqueeze(0)


def prepare_image_for_main_model(image_path: str | Path) -> torch.Tensor:
    """Подготовить тензор изображения для основной модели."""
    image = load_image(image_path)
    image = resize_image(image, image_size=IMAGE_SIZE)
    return image_to_tensor(image)


def prepare_image_for_fallback_model(image_path: str | Path) -> torch.Tensor:
    """Подготовить тензор изображения для fallback-модели."""
    image = load_image(image_path)
    image = resize_image(image, image_size=IMAGE_SIZE)
    return image_to_tensor(image)


def prepare_image_for_monitoring(image_path: str | Path) -> np.ndarray:
    """Подготовить изображение для расчета monitoring-метрик."""
    image = load_image(image_path)
    return resize_image(image, image_size=IMAGE_SIZE)
