# Отказоустойчивый инференс для KolektorSDD2

Проект решает задачу бинарной классификации изображений `normal / defect`.

Текущая структура репозитория:
- `app/` — backend
- `data/` — локальный датасет
- `models/` — обученные веса моделей
- `notebooks/` — исследовательский ноутбук

## Модели

- основная модель: `ResNet50`
- fallback-модель: `MobileNetV2`

## Датасет

В проекте используется датасет `Kolektor Surface-Defect Dataset 2 (KolektorSDD2 / KSDD2)` для задачи обнаружения дефектов на промышленных поверхностях.

Официальная страница датасета:
- https://www.vicos.si/resources/kolektorsdd2/

На странице ViCoS Lab указано, что датасет содержит `356` изображений с дефектами и `2979` изображений без дефектов, а также фиксированные train/test-разделения.

Файлы весов:
- `models/resnet50_main.pth`
- `models/mobilenetv2_fallback.pth`

## app

- `app/model_inference/config.py` — конфиг инференса
- `app/model_inference/preprocessing.py` — загрузка и подготовка изображения
- `app/model_inference/monitoring.py` — метрики
- `app/model_inference/model_loader.py` — загрузка моделей
- `app/model_inference/predictors.py` — функции предсказания
- `app/model_inference/fallback_service.py` — оркестрация fallback-логики
- `app/inference_service.py` — сервис для будущей интеграции
- `app/streamlit_demo.py` — интерактивное Streamlit-демо

## Streamlit-демо

```bash
streamlit run app/streamlit_demo.py
```

В демо можно выбрать папку датасета, запустить live-режим с задержкой между кадрами, посмотреть итоговую метку `normal / defect`, время обработки последнего кадра, выбранный режим модели, latency, confidence и причину fallback. Кнопки симуляции включают шум/размытие/темный кадр на входе и timeout основной модели: ResNet50 не отвечает 1 секунду, после чего запрос переводится на MobileNetV2 fallback.

## Быстрый пример

```python
from app.inference_service import run_inference

result = run_inference("data/KolektorSDD2/test/20000.png")
print(result)
```

## Формат результата

`run_inference(...)` и `predict_with_fallback(...)` возвращают словарь:

- `final_label`
- `prediction`
- `confidence`
- `model_used`
- `fallback_reason`
- `latency_ms`
- `monitoring_metrics`
