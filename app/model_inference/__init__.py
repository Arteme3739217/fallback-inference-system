"""Переиспользуемая логика инференса."""

from app.model_inference.fallback_service import predict_with_fallback
from app.model_inference.model_loader import load_fallback_model, load_main_model, load_models

__all__ = [
    "load_fallback_model",
    "load_main_model",
    "load_models",
    "predict_with_fallback",
]
