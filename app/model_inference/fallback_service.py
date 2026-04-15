"""Оркестрация fallback-инференса."""

from __future__ import annotations

from pathlib import Path

from app.model_inference.config import (
    CONFIDENCE_THRESHOLD,
    DEVICE,
    LABEL_TO_NAME,
    TIMEOUT_THRESHOLD,
)
from app.model_inference.model_loader import load_models
from app.model_inference.monitoring import build_monitoring_metrics, measure_latency_ms
from app.model_inference.predictors import predict_fallback_model, predict_main_model
from app.model_inference.preprocessing import prepare_image_for_monitoring


def predict_with_fallback(
    image_path: str | Path,
    main_model=None,
    fallback_model=None,
    device: str = DEVICE,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    timeout_threshold: float = TIMEOUT_THRESHOLD,
    simulated_delay_seconds: float = 0.0,
    force_main_failure: bool = False,
) -> dict:
    """Сделать предсказание основной моделью и при необходимости переключиться на fallback."""
    if main_model is None or fallback_model is None:
        main_model, fallback_model = load_models(device=device)

    monitoring_image = prepare_image_for_monitoring(image_path)
    fallback_reason: str | None = None

    try:
        if force_main_failure:
            raise RuntimeError("Main model failure was simulated.")
        if simulated_delay_seconds > timeout_threshold:
            raise TimeoutError(
                f"Main model timed out after {simulated_delay_seconds:.3f}s "
                f"with timeout threshold {timeout_threshold:.3f}s."
            )

        main_result, latency_ms = measure_latency_ms(
            predict_main_model,
            main_model,
            image_path,
            device=device,
        )

        if latency_ms > timeout_threshold * 1000.0:
            fallback_reason = "timeout"
        elif main_result["confidence"] < confidence_threshold:
            fallback_reason = "low_confidence"
        else:
            monitoring_metrics = build_monitoring_metrics(
                image=monitoring_image,
                probabilities=main_result["probabilities"],
                latency_ms=latency_ms,
            )
            prediction = int(main_result["prediction"])
            return {
                "final_label": LABEL_TO_NAME[prediction],
                "prediction": prediction,
                "confidence": float(main_result["confidence"]),
                "model_used": "main_model",
                "fallback_reason": None,
                "latency_ms": float(latency_ms),
                "monitoring_metrics": monitoring_metrics,
            }
    except TimeoutError:
        fallback_reason = "timeout"
    except Exception:
        fallback_reason = "exception"

    fallback_result, latency_ms = measure_latency_ms(
        predict_fallback_model,
        fallback_model,
        image_path,
        device=device,
    )
    monitoring_metrics = build_monitoring_metrics(
        image=monitoring_image,
        probabilities=fallback_result["probabilities"],
        latency_ms=latency_ms,
    )
    prediction = int(fallback_result["prediction"])
    return {
        "final_label": LABEL_TO_NAME[prediction],
        "prediction": prediction,
        "confidence": float(fallback_result["confidence"]),
        "model_used": "fallback_model",
        "fallback_reason": fallback_reason,
        "latency_ms": float(latency_ms),
        "monitoring_metrics": monitoring_metrics,
    }
