"""Функции предсказания для основной и fallback-моделей."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from skimage.feature import hog

from app.model_inference.config import DEFECT_CLASS_INDEX, DEVICE
from app.model_inference.monitoring import compute_confidence
from app.model_inference.preprocessing import (
    load_image,
    prepare_image_for_fallback_model,
    prepare_image_for_main_model,
    resize_image,
)


def _predict_model(model: torch.nn.Module, image_tensor: torch.Tensor, device: str = DEVICE) -> dict:
    """Запустить одну CNN-модель на подготовленном тензоре."""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        model = model.to(device)
        model.eval()
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        prediction = int(np.argmax(probabilities))
        return {
            "prediction": prediction,
            "probabilities": probabilities,
            "confidence": compute_confidence(probabilities),
            "defect_probability": float(probabilities[DEFECT_CLASS_INDEX]),
        }


def predict_main_model(
    model: torch.nn.Module,
    image_path: str | Path,
    device: str = DEVICE,
) -> dict:
    """Выполнить инференс основной моделью."""
    image_tensor = prepare_image_for_main_model(image_path)
    return _predict_model(model, image_tensor=image_tensor, device=device)


def extract_hog_features(image_path: str | Path) -> np.ndarray:
    """Извлечь HOG-признаки из одного изображения."""
    image = load_image(image_path)
    image = resize_image(image)
    return hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )


def predict_fallback_model(
    model: torch.nn.Module,
    image_path: str | Path,
    device: str = DEVICE,
) -> dict:
    """Выполнить инференс fallback-моделью."""
    image_tensor = prepare_image_for_fallback_model(image_path)
    return _predict_model(model, image_tensor=image_tensor, device=device)
