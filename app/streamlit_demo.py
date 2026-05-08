from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from app.model_inference.config import (
    CONFIDENCE_THRESHOLD,
    LABEL_TO_NAME,
    PROJECT_ROOT,
)
from app.model_inference.fallback_service import predict_with_fallback
from app.model_inference.model_loader import load_models
from app.model_inference.monitoring import build_monitoring_metrics, measure_latency_ms
from app.model_inference.predictors import predict_fallback_model, predict_main_model
from app.model_inference.preprocessing import load_image, prepare_image_for_monitoring


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
RUNTIME_DIR = PROJECT_ROOT / ".streamlit_demo_runtime"
DEMO_TIMEOUT_SECONDS = 1.0
MAX_DISPLAY_IMAGE_WIDTH = 920
MAX_DISPLAY_IMAGE_HEIGHT = 560
MODEL_MODES = {
    "Auto: ResNet50 + fallback": "auto",
    "Main: ResNet50": "main",
    "Fallback: MobileNetV2": "fallback",
}


st.set_page_config(
    page_title="Fallback Inference Demo",
    page_icon="",
    layout="wide",
)


@st.cache_resource(show_spinner="Загружаю ResNet50 и MobileNetV2...")
def get_cached_models():
    return load_models()


@st.cache_data(show_spinner=False)
def collect_images(dataset_dir: str) -> list[str]:
    root = resolve_dataset_dir(dataset_dir)
    if not root.exists() or not root.is_dir():
        return []

    images = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem.endswith("_GT"):
            continue
        images.append(str(path))
    return sorted(images)


def resolve_dataset_dir(dataset_dir: str) -> Path:
    path = Path(dataset_dir).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_gray_image(image_path: str | Path) -> np.ndarray:
    return load_image(image_path)


def save_gray_image(image: np.ndarray, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", np.asarray(image, dtype=np.uint8))
    if not success:
        raise RuntimeError("Could not encode demo image.")
    encoded.tofile(output_path)
    return output_path


def apply_input_noise(image: np.ndarray, noise_enabled: bool, blur_enabled: bool, dark_enabled: bool) -> np.ndarray:
    output = np.asarray(image, dtype=np.float32)

    if noise_enabled:
        rng = np.random.default_rng(seed=42)
        output = output + rng.normal(loc=0.0, scale=14.0, size=output.shape)

    if blur_enabled:
        output = cv2.GaussianBlur(output, (5, 5), sigmaX=1.1)

    if dark_enabled:
        output = output * 0.48

    return np.clip(output, 0, 255).astype(np.uint8)


def make_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def fit_display_image(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(
        MAX_DISPLAY_IMAGE_WIDTH / width,
        MAX_DISPLAY_IMAGE_HEIGHT / height,
        1.0,
    )
    if scale >= 1.0:
        return image

    target_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def prepare_display_image(image: np.ndarray) -> np.ndarray:
    return fit_display_image(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))


def normalize_probabilities(probabilities: np.ndarray | list[float]) -> dict[str, float]:
    probabilities_array = np.asarray(probabilities, dtype=np.float32)
    return {
        LABEL_TO_NAME[index]: float(probability)
        for index, probability in enumerate(probabilities_array)
    }


def build_single_model_result(
    model_name: str,
    model,
    image_path: str | Path,
    predictor,
) -> dict:
    prediction_result, latency_ms = measure_latency_ms(predictor, model, image_path)
    monitoring_image = prepare_image_for_monitoring(image_path)
    metrics = build_monitoring_metrics(
        image=monitoring_image,
        probabilities=prediction_result["probabilities"],
        latency_ms=latency_ms,
    )
    prediction = int(prediction_result["prediction"])
    return {
        "final_label": LABEL_TO_NAME[prediction],
        "prediction": prediction,
        "probabilities": prediction_result["probabilities"],
        "confidence": float(prediction_result["confidence"]),
        "defect_probability": float(prediction_result["defect_probability"]),
        "model_used": model_name,
        "fallback_reason": None,
        "latency_ms": float(latency_ms),
        "monitoring_metrics": metrics,
    }


def run_demo_inference(
    image_path: str | Path,
    model_mode: str,
    force_timeout: bool,
    confidence_threshold: float,
) -> dict:
    main_model, fallback_model = get_cached_models()

    if model_mode == "main":
        return build_single_model_result("main_model", main_model, image_path, predict_main_model)

    if model_mode == "fallback":
        return build_single_model_result("fallback_model", fallback_model, image_path, predict_fallback_model)

    simulated_delay = DEMO_TIMEOUT_SECONDS if force_timeout else 0.0
    return predict_with_fallback(
        image_path=image_path,
        main_model=main_model,
        fallback_model=fallback_model,
        confidence_threshold=confidence_threshold,
        simulated_delay_seconds=simulated_delay,
    )


def format_reason(reason: str | None) -> str:
    if reason is None:
        return "нет"
    return {
        "timeout": "timeout основной модели",
        "low_confidence": "низкая уверенность",
        "exception": "ошибка основной модели",
    }.get(reason, reason)


if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "scenario" not in st.session_state:
    st.session_state.scenario = {
        "noise": False,
        "blur": False,
        "dark": False,
        "timeout": False,
    }


st.title("Fallback Inference Demo")

with st.sidebar:
    st.header("Источник")
    dataset_dir = st.text_input("Папка датасета", value="data/KolektorSDD2/test")
    images = collect_images(dataset_dir)

    st.header("Режим")
    model_mode_label = st.selectbox("Выбранная модель", list(MODEL_MODES.keys()))
    model_mode = MODEL_MODES[model_mode_label]
    delay_seconds = st.slider("Задержка между кадрами, сек", 1.0, 10.0, 3.0, 0.5)
    confidence_threshold = st.slider(
        "Порог fallback по confidence",
        0.50,
        0.99,
        float(CONFIDENCE_THRESHOLD),
        0.01,
    )
    live_enabled = st.toggle("Live", value=False)

    st.header("Симуляции")
    if st.button("Чистый вход", width="stretch"):
        st.session_state.scenario.update(
            noise=False,
            blur=False,
            dark=False,
            timeout=False,
        )
    preset_cols = st.columns(2)
    if preset_cols[1].button("Шум", width="stretch"):
        st.session_state.scenario["noise"] = not st.session_state.scenario["noise"]
    if preset_cols[0].button("Размытие", width="stretch"):
        st.session_state.scenario["blur"] = not st.session_state.scenario["blur"]
    if preset_cols[1].button("Темный кадр", width="stretch"):
        st.session_state.scenario["dark"] = not st.session_state.scenario["dark"]
    if st.button("Timeout ResNet50", width="stretch"):
        st.session_state.scenario["timeout"] = not st.session_state.scenario["timeout"]

    active_scenarios = [
        label
        for key, label in [
            ("noise", "шум"),
            ("blur", "размытие"),
            ("dark", "темный кадр"),
            ("timeout", "timeout ResNet50 1 сек"),
        ]
        if st.session_state.scenario[key]
    ]
    st.caption("Активно: " + (", ".join(active_scenarios) if active_scenarios else "нет"))


if not images:
    st.error("В выбранной папке не найдено изображений.")
    st.stop()

st.session_state.current_index %= len(images)
current_path = Path(images[st.session_state.current_index])
original_image = read_gray_image(current_path)
scenario = st.session_state.scenario
input_image = apply_input_noise(
    original_image,
    noise_enabled=scenario["noise"],
    blur_enabled=scenario["blur"],
    dark_enabled=scenario["dark"],
)
input_path = current_path
if scenario["noise"] or scenario["blur"] or scenario["dark"]:
    input_path = save_gray_image(input_image, RUNTIME_DIR / "current_input.png")

top_placeholder = st.empty()

control_cols = st.columns([0.8, 0.8, 3.0])
if control_cols[0].button("Назад", width="stretch"):
    st.session_state.current_index = (st.session_state.current_index - 1) % len(images)
    st.rerun()
if control_cols[1].button("Вперед", width="stretch"):
    st.session_state.current_index = (st.session_state.current_index + 1) % len(images)
    st.rerun()
control_cols[2].caption(str(current_path.relative_to(PROJECT_ROOT) if current_path.is_relative_to(PROJECT_ROOT) else current_path))

frame_started_at = time.perf_counter()
try:
    if scenario["timeout"] and model_mode == "auto":
        with st.status("ResNet50 не отвечает 1 сек...", expanded=True) as status:
            st.write("Ожидаем ответ основной модели.")
            inference_result = run_demo_inference(
                image_path=input_path,
                model_mode=model_mode,
                force_timeout=True,
                confidence_threshold=confidence_threshold,
            )
            status.update(
                label="Timeout ResNet50: включен fallback MobileNetV2",
                state="complete",
                expanded=True,
            )
    else:
        inference_result = run_demo_inference(
            image_path=input_path,
            model_mode=model_mode,
            force_timeout=scenario["timeout"],
            confidence_threshold=confidence_threshold,
        )
except Exception as exc:
    st.error(f"Инференс завершился ошибкой: {exc}")
    inference_result = None
frame_processing_ms = (time.perf_counter() - frame_started_at) * 1000.0

top_cols = top_placeholder.columns([1.1, 1.1, 1.1, 1.1])
top_cols[0].metric("Обработка кадра", f"{frame_processing_ms:.0f} ms")
top_cols[1].metric("Выбранная модель", model_mode_label)
top_cols[2].metric("Кадр", f"{st.session_state.current_index + 1} / {len(images)}")
top_cols[3].metric("Задержка live", f"{delay_seconds:.1f} сек")

if inference_result is not None:
    is_defect = inference_result["final_label"] == "defect"
    status_text = "ДЕФЕКТ ОБНАРУЖЕН" if is_defect else "NORMAL"
    status_class = "defect" if is_defect else "normal"

    st.markdown(
        f"""
        <style>
        .status-panel {{
            border: 1px solid rgba(49, 51, 63, 0.18);
            border-radius: 8px;
            padding: 14px 16px;
            margin: 8px 0 16px 0;
            background: #ffffff;
        }}
        .status-title {{
            font-size: 13px;
            color: #5c6470;
            margin-bottom: 4px;
        }}
        .status-value {{
            font-size: 28px;
            font-weight: 750;
            letter-spacing: 0;
        }}
        .status-value.defect {{ color: #c72532; }}
        .status-value.normal {{ color: #16794c; }}
        .timeout-panel {{
            border-left: 5px solid #d97706;
            border-radius: 8px;
            padding: 12px 16px;
            margin: 0 0 16px 0;
            background: #fff7ed;
            color: #7c2d12;
            font-weight: 650;
        }}
        </style>
        <div class="status-panel">
            <div class="status-title">Итог</div>
            <div class="status-value {status_class}">{status_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if inference_result["fallback_reason"] == "timeout":
        st.markdown(
            f"""
            <div class="timeout-panel">
                ResNet50 не ответила за {DEMO_TIMEOUT_SECONDS:.0f} сек. Запрос переведен на MobileNetV2 fallback.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.image(
        prepare_display_image(make_rgb(input_image)),
        caption="Вход модели",
        width="content",
    )

    metrics_cols = st.columns(5)
    metrics_cols[0].metric("Модель в работе", inference_result["model_used"])
    metrics_cols[1].metric("Fallback", format_reason(inference_result["fallback_reason"]))
    metrics_cols[2].metric("Latency", f"{inference_result['latency_ms']:.1f} ms")
    metrics_cols[3].metric("Confidence", f"{inference_result['confidence']:.3f}")
    metrics_cols[4].metric("P(defect)", f"{inference_result['defect_probability']:.3f}")

    probability_df = pd.DataFrame(
        {
            "class": list(normalize_probabilities(inference_result["probabilities"]).keys()),
            "probability": list(normalize_probabilities(inference_result["probabilities"]).values()),
        }
    )
    st.bar_chart(probability_df, x="class", y="probability", height=180)

    monitoring = inference_result["monitoring_metrics"]
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "entropy": round(monitoring["entropy"], 4),
                    "brightness": round(monitoring["brightness"], 4),
                    "blur_score": round(monitoring["blur_score"], 2),
                    "latency_ms": round(monitoring["latency_ms"], 2),
                }
            ]
        ),
        width="stretch",
        hide_index=True,
    )

if live_enabled:
    time.sleep(delay_seconds)
    st.session_state.current_index = (st.session_state.current_index + 1) % len(images)
    st.rerun()
