"""Конфигурация переиспользуемой логики инференса."""

from __future__ import annotations

from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

MAIN_MODEL_PATH = MODELS_DIR / "resnet50_main.pth"
FALLBACK_MODEL_PATH = MODELS_DIR / "mobilenetv2_fallback.pth"

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 2
DEFECT_CLASS_INDEX = 1
CONFIDENCE_THRESHOLD = 0.75
TIMEOUT_THRESHOLD = 1.0
MAIN_MODEL_NAME = "resnet50"
FALLBACK_MODEL_NAME = "mobilenetv2"
MAIN_DROPOUT = 0.35
FALLBACK_DROPOUT = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_TO_NAME = {
    0: "normal",
    1: "defect",
}


def resolve_project_path(path: str | Path) -> Path:
    """Преобразовать путь относительно проекта в абсолютный путь."""
    target_path = Path(path)
    if target_path.is_absolute():
        return target_path
    return PROJECT_ROOT / target_path
