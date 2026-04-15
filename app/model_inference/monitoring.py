"""Monitoring-метрики для инференса."""

from __future__ import annotations

import time
from collections.abc import Callable

import cv2
import numpy as np
import torch


def compute_confidence(probabilities: np.ndarray | list[float]) -> float:
    """Вернуть максимальную предсказанную вероятность."""
    probabilities_array = np.asarray(probabilities, dtype=np.float32)
    return float(probabilities_array.max())


def compute_entropy(probabilities: np.ndarray | list[float], eps: float = 1e-8) -> float:
    """Вычислить энтропию предсказания."""
    probabilities_array = np.asarray(probabilities, dtype=np.float32)
    return float(-np.sum(probabilities_array * np.log(probabilities_array + eps)))


def compute_brightness(image: np.ndarray) -> float:
    """Вычислить нормализованную яркость изображения."""
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.max() > 1.0:
        image_array = image_array / 255.0
    return float(image_array.mean())


def compute_blur_score(image: np.ndarray) -> float:
    """Оценить степень размытия по дисперсии лапласиана."""
    image_uint8 = np.asarray(image, dtype=np.uint8)
    return float(cv2.Laplacian(image_uint8, cv2.CV_64F).var())


def measure_latency_ms(fn: Callable, *args, device: str | None = None, **kwargs) -> tuple[object, float]:
    """Измерить время выполнения функции в миллисекундах."""
    start_time = time.perf_counter()
    result = fn(*args, **kwargs)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    return result, float(latency_ms)


def build_monitoring_metrics(
    image: np.ndarray,
    probabilities: np.ndarray | list[float],
    latency_ms: float,
) -> dict:
    """Собрать все monitoring-метрики для одного вызова инференса."""
    return {
        "confidence": compute_confidence(probabilities),
        "entropy": compute_entropy(probabilities),
        "brightness": compute_brightness(image),
        "blur_score": compute_blur_score(image),
        "latency_ms": float(latency_ms),
    }
