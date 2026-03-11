# DSPro VKR

Проект по физически согласованной приоритизации звёздных кандидатов для последующих наблюдений.

## Что делает проект

Текущая MVP-архитектура разбивает задачу на три слоя:

1. `router` определяет физический класс звезды по `teff_gspphot`, `logg_gspphot`, `radius_gspphot`.
2. `host-модель` внутри ветки `M/K/G/F dwarf` оценивает `host vs field`.
3. `decision layer` объединяет физический класс, host-score и коэффициенты качества в итоговый `final_score`.

## Актуальная архитектура

Каноническая логика живёт в пакетах:

- `src/router_model` — обучение, артефакты и scoring физического router;
- `src/host_model` — legacy и контрастивная host-модель;
- `src/priority_pipeline` — боевой pipeline приоритизации;
- `src/decision_calibration` — офлайн-калибровка decision layer;
- `analysis/host_eda` и `analysis/router_eda` — исследовательский контур EDA.

Верхнеуровневые файлы:

- `src/gaussian_router.py`
- `src/model_gaussian.py`
- `src/star_orchestrator.py`
- `src/eda.py`
- `src/router_eda.py`
- `src/decision_layer_calibrator.py`

сохранены как фасады совместимости для старых импортов и прежних CLI-точек входа.

## Текущий scoring-контракт

- `router`: выбор класса идёт по `router_log_posterior`, а не по одному расстоянию Mahalanobis;
- `host-модель`: основной ограниченный score — `host_posterior`, диагностический raw score — `host_log_lr`;
- `decision layer`: боевая формула имеет вид:

```text
final_score =
  host_posterior
  × class_prior
  × quality_factor
  × metallicity_factor
  × color_factor
  × validation_factor
```

Для офлайн-калибровки используется близкая формула, но с явным `distance_factor`.

## Структура репозитория

```text
src/
  input_layer.py                # валидация входного relation и registry
  gaussian_router.py            # фасад совместимости для router_model
  model_gaussian.py             # фасад совместимости для host_model
  star_orchestrator.py          # фасад совместимости для priority_pipeline
  decision_layer_calibrator.py  # фасад совместимости для decision_calibration
  eda.py                        # фасад совместимости для analysis.host_eda
  router_eda.py                 # фасад совместимости для analysis.router_eda
  infra/                        # общая инфраструктура БД и logbook
  logbooks/                     # генераторы markdown-журналов
  devtools/                     # служебные smoke-check сценарии
  router_model/                 # физический Gaussian router
  host_model/                   # host-модель
  priority_pipeline/            # боевой pipeline приоритизации
  decision_calibration/         # офлайн-калибровка

analysis/
  host_eda/                     # исследовательский host EDA
  router_eda/                   # исследовательский router EDA

notebooks/
  eda/

data/
  router_gaussian_params.json   # production artifact router-модели
  model_gaussian_params.json    # production artifact host-модели
  eda/                          # сохранённые EDA-артефакты

docs/
  documentation_style_ru.md     # принятый стандарт документации
  documentation_audit_tz_ru.md  # аудит и исходное ТЗ по унификации

experiments/
  Логи работы программы/
  Логи калибровки decision_layer/
  QA/
```

## Основные сценарии

Подготовка окружения:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Запуск тестов:

```bash
./venv/bin/python -m pytest -q
```

DB-backed интеграционный тест на временной схеме Postgres:

```bash
./venv/bin/python -m pytest -q -m db_integration
```

Этот тест:

- покрывает `run_pipeline(..., persist=True)` и `input_layer`;
- читает входной relation из временной схемы;
- проверяет запись в router-, priority- и registry-таблицы;
- требует доступный Postgres через `.env` или переменные `DATABASE_URL` / `PG*`;
- после завершения удаляет временную схему `test_it_*`.

Проверка подключения к БД:

```bash
python src/devtools/db_smoke.py
```

EDA router-слоя:

```bash
python src/router_eda.py
```

EDA host-модели:

```bash
python src/eda.py
```

Обучение contrastive host-модели:

```bash
python src/model_gaussian.py --mode contrastive
```

Preview боевого pipeline:

```bash
python src/star_orchestrator.py
```

Офлайн-калибровка:

```bash
python src/decision_layer_calibrator.py --relation public.gaia_dr3_training
```

Создание файла журнала прогона:

```bash
python src/logbooks/program_run.py
```

Создание файла журнала калибровки:

```bash
python src/logbooks/decision_layer.py
```

## Статические проверки

Lint:

```bash
./venv/bin/python -m ruff check src tests analysis
```

Проверка типов:

```bash
./venv/bin/python -m mypy src tests analysis
```

## Стандарт документации

В проекте принят единый русскоязычный стандарт документации:

- для Python-кода — module docstring, function docstring и описания структур данных;
- для markdown-документов — единый каркас README, QA-отчётов и logbook-файлов;
- для фасадов — обязательная явная пометка, что это фасад совместимости.

Канонический документ стандарта:

- `docs/documentation_style_ru.md`

Аудит и исходное ТЗ, по которому проводилась унификация:

- `docs/documentation_audit_tz_ru.md`

## Важные замечания

- Поля `similarity` и `d_mahal` сохранены только для обратной совместимости со старым legacy-контуром скоринга.
- Production host-ветка уже использует contrastive поля:
  `host_log_likelihood`, `field_log_likelihood`, `host_log_lr`, `host_posterior`.
- Если включать `persist=True`, целевые DB-таблицы должны содержать новые host-поля.

## Статус MVP и QA

Текущая версия проекта является первой научной MVP-итерацией. Её задача — проверить жизнеспособность связки:

- physical Gaussian router;
- contrastive `host-vs-field` model;
- decision layer с коэффициентами качества.

QA-прогон на 500 звёздах показал, что:

- pipeline стабильно выполняется end-to-end;
- численный контракт модели остаётся корректным;
- физические sanity-checks дают правдоподобную картину;
- кодовая база и тесты находятся в воспроизводимом состоянии.

Подробный QA-отчёт:

- `experiments/QA/qa_mvp_report_2026-03-11.md`

## Ближайшие этапы

- обновить persist-schema под новые host-поля;
- пересобрать боевые артефакты после retrain;
- выполнить новую итерацию калибровки;
- добавить OOD / `unknown` слой поверх router posterior.
