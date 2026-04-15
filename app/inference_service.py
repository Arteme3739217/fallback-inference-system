"""Сервис инференса для будущей интеграции с интерфейсом."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.model_inference.fallback_service import predict_with_fallback
from app.model_inference.model_loader import load_models


@lru_cache(maxsize=1)
def get_models():
    """Загрузить и закешировать модели для повторных вызовов."""
    return load_models()


def run_inference(image_path: str | Path, **kwargs) -> dict:
    """Запустить один вызов инференса через общий backend-сервис."""
    main_model, fallback_model = get_models()
    return predict_with_fallback(
        image_path=image_path,
        main_model=main_model,
        fallback_model=fallback_model,
        **kwargs,
    )
