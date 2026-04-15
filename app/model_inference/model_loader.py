"""Функции загрузки моделей."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import mobilenet_v2, resnet50

from app.model_inference.config import (
    DEVICE,
    FALLBACK_DROPOUT,
    FALLBACK_MODEL_PATH,
    MAIN_DROPOUT,
    MAIN_MODEL_PATH,
    NUM_CLASSES,
)


def _initialize_main_model(num_classes: int = NUM_CLASSES, dropout: float = MAIN_DROPOUT) -> nn.Module:
    """Создать архитектуру ResNet50, совпадающую с обучением."""
    model = resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def _initialize_fallback_model(num_classes: int = NUM_CLASSES, dropout: float = FALLBACK_DROPOUT) -> nn.Module:
    """Создать архитектуру MobileNetV2, совпадающую с обучением."""
    model = mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def _load_state_dict(model: nn.Module, model_path: str | Path, device: str = DEVICE) -> nn.Module:
    """Загрузить веса в модель и перевести ее в режим eval."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model weights were not found: {path}")

    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def load_main_model(model_path: str | Path = MAIN_MODEL_PATH, device: str = DEVICE) -> nn.Module:
    """Загрузить основную модель ResNet50."""
    model = _initialize_main_model()
    return _load_state_dict(model, model_path=model_path, device=device)


def load_fallback_model(model_path: str | Path = FALLBACK_MODEL_PATH, device: str = DEVICE) -> nn.Module:
    """Загрузить fallback-модель MobileNetV2."""
    model = _initialize_fallback_model()
    return _load_state_dict(model, model_path=model_path, device=device)


def load_models(device: str = DEVICE) -> tuple[nn.Module, nn.Module]:
    """Загрузить обе модели для инференса."""
    return load_main_model(device=device), load_fallback_model(device=device)
