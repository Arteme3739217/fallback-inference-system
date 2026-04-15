# Отказоустойчивый инференс для KolektorSDD2

Проект решает задачу бинарной классификации изображений `normal / defect`.

Текущая структура репозитория:
- `app/` — backend-слой инференса
- `data/` — локальный датасет
- `models/` — обученные веса моделей
- `notebooks/` — исследовательский ноутбук и демонстрация

## Модели

- основная модель: `ResNet50`
- fallback-модель: `MobileNetV2`

## Датасет

В проекте используется датасет `Kolektor Surface-Defect Dataset 2 (KolektorSDD2 / KSDD2)` для задачи обнаружения дефектов на промышленных поверхностях.

Официальная страница датасета:
- https://www.vicos.si/resources/kolektorsdd2/

На странице ViCoS Lab указано, что датасет содержит `356` изображений с дефектами и `2979` изображений без дефектов, а также фиксированные train/test-разделения. Источник: ViCoS Lab, KolektorSDD2 dataset page.

Файлы весов:
- `models/resnet50_main.pth`
- `models/mobilenetv2_fallback.pth`

## Что лежит в app

- `app/model_inference/config.py` — конфиг инференса
- `app/model_inference/preprocessing.py` — загрузка и подготовка изображения
- `app/model_inference/monitoring.py` — monitoring-метрики
- `app/model_inference/model_loader.py` — загрузка моделей
- `app/model_inference/predictors.py` — функции предсказания
- `app/model_inference/fallback_service.py` — orchestration fallback-логики
- `app/inference_service.py` — тонкий сервисный слой для будущей интеграции

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
