# DSPro VKR

Проект по физически согласованной приоритизации наблюдений за звёздами с целью оценки вероятности наличия экзопланет.

## Что делает проект

Текущая MVP-архитектура решает задачу в три шага:

1. `router` распознаёт звезду по `teff_gspphot`, `logg_gspphot`, `radius_gspphot` и относит её к физическому классу.
2. `host-модель` внутри ветки `M/K/G/F dwarf` сравнивает распознанную звезду со звёздами, у которых уже открыты экзопланеты, и оценивает `host vs field`.
3. `decision layer` объединяет физический класс, host-score и коэффициенты качества, чтобы оценить вероятность наличия экзопланеты и сформировать итоговый `final_score`.

Основные внешние архивы данных:

- [Gaia Archive](https://gea.esac.esa.int/archive/) — архив Gaia ESA;
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) — архив подтверждённых экзопланет и звёзд-хостов NASA.

## Текущий статус

- production-контур уже использует posterior-aware `router`, contrastive `host vs field` и decision layer с quality-факторами;
- comparison-layer доведён до формального ВКР-контракта: `test_size = 0.30`, `10-fold` CV и compact hyperparameter search;
- текущая версия проекта — рабочая научная MVP, а не финальная астрофизическая теория или окончательная публикационная модель;
- latest full QA wave зафиксирована в `experiments/QA/` и подтверждает, что у проекта сейчас нет красных флагов по базовой математике и архитектуре.

## Где смотреть сначала

- если нужен production-контур: `src/router_model`, `src/host_model`, `src/priority_pipeline`, `src/decision_calibration`;
- если нужен research-layer: `analysis/model_comparison`, `analysis/host_eda`, `analysis/router_eda`;
- если нужен current state policy: `docs/repository_state_policy_ru.md`;
- если нужны канонические comparison-артефакты: `experiments/model_comparison/README.md`;
- если нужны QA findings и backlog: `experiments/QA/README.md`.

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
  model_comparison/             # comparative benchmark основной модели и baseline

notebooks/
  eda/

data/
  README.md                    # policy по model artifacts, samples и EDA-data
  router_gaussian_params.json   # production artifact router-модели
  model_gaussian_params.json    # production artifact host-модели
  eda/                          # сохранённые EDA-артефакты

docs/
  documentation_style_ru.md     # принятый стандарт документации
  repository_state_policy_ru.md # policy текущего versioned state
  vkr_requirements_traceability_ru.md
  preprocessing_pipeline_ru.md
  model_comparison_protocol_ru.md
  model_comparison_findings_ru.md
  presentation/

experiments/
  Логи работы программы/
  Логи калибровки decision_layer/
  QA/
  model_comparison/

sql/
  adql/                         # ADQL-запросы Gaia Archive
  preprocessing/                # канонические SQL-артефакты preprocessing
```

Верхнеуровневые файлы `src/*.py` сохранены как фасады совместимости для
старых импортов и CLI-точек входа. Каноническая логика при этом живёт в
пакетах, а не в фасадах.

## Быстрый старт и основные сценарии

Подготовка окружения:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Целевой runtime для проекта: `Python 3.13.x`.

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

Сравнение основной модели и baseline:

```bash
python src/model_comparison.py
```

По умолчанию команда делает два артефактных контура:

- supervised benchmark на `host vs field`;
- live snapshot на `public.gaia_dr3_training` после общего `router + OOD`.

Canonical benchmark-режим теперь соответствует формальному протоколу ВКР:

- `test_size = 0.30`;
- `10-fold` stratified CV внутри train split;
- compact search для всех четырёх моделей;
- отдельный `search_summary.csv` с best params и best CV score.

В benchmark сейчас участвуют четыре model head:

- `main_contrastive_v1`
- `baseline_legacy_gaussian`
- `baseline_random_forest`
- `baseline_mlp_small`

Полезные overrides:

```bash
python src/model_comparison.py --snapshot-limit 5000 --snapshot-top-k 25
python src/model_comparison.py --skip-snapshot
python src/model_comparison.py --cv-folds 5 --search-refit-metric pr_auc
```

Каноническое поколение comparison-артефактов для ВКР:

- benchmark: `baseline_comparison_2026-03-13_vkr30_cv10`
- snapshot preview: `baseline_comparison_2026-03-13_vkr30_cv10_limit5000`

Исторические волны этого же дня сохранены в
`experiments/model_comparison/`, но текущим источником истины считаются
только файлы поколения `vkr30_cv10`. Это отдельно зафиксировано в:

- `experiments/model_comparison/README.md`

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

`pyrightconfig.json` в репозитории сохранён как editor-facing конфиг для
Pylance/Pyright-совместимых IDE. Канонический CLI-набор проверок проекта
сейчас: `ruff + mypy + pytest`. Отдельный `pyright` пока не считается
обязательной частью регулярного QA-прогона.

## Канонические документы и артефакты

Current state и policy:

- `docs/repository_state_policy_ru.md` — что считать versioned current state;
- `data/README.md` — policy по model artifacts, samples и EDA-data;
- `experiments/QA/README.md` — current QA wave и related artifacts;
- `experiments/model_comparison/README.md` — каноническая и historical policy для comparison-артефактов.

ВКР и narrative:

- `docs/vkr_requirements_traceability_ru.md`;
- `docs/preprocessing_pipeline_ru.md`;
- `docs/model_comparison_protocol_ru.md`;
- `docs/model_comparison_findings_ru.md`;
- `docs/presentation/vkr_slides_draft_ru.md`;
- `notebooks/eda/00_data_extraction_and_preprocessing.ipynb`;
- `notebooks/eda/04_model_comparison_summary.ipynb`.

Полный QA-аудит:

- `experiments/QA/qa_full_audit_log_2026-03-13_ru.md`;
- `experiments/QA/qa_backlog_and_decision_map_2026-03-13_ru.md`;
- `experiments/QA/qa_runbook_2026-03-13_ru.md`.

## Исторические planning docs

Эти документы сохранены как история решений и ТЗ, но не должны читаться
как главный current state:

- `docs/documentation_audit_tz_ru.md`;
- `docs/preprocessing_and_comparison_tz_ru.md`;
- `docs/ood_unknown_tz_ru.md`;
- `docs/ood_unknown_baselines_tz_ru.md`.

## Стандарт документации

В проекте принят единый русскоязычный стандарт документации:

- для Python-кода — module docstring, function docstring и описания структур данных;
- для markdown-документов — единый каркас README, QA-отчётов и logbook-файлов;
- для фасадов — обязательная явная пометка, что это фасад совместимости.

Канонический документ стандарта:

- `docs/documentation_style_ru.md`

## Важные замечания по совместимости

- Поля `similarity` и `d_mahal` сохранены только для обратной совместимости со старым legacy-контуром скоринга.
- Production host-ветка уже использует contrastive поля:
  `host_log_likelihood`, `field_log_likelihood`, `host_log_lr`, `host_posterior`.
- Если включать `persist=True`, целевые DB-таблицы должны содержать новые host-поля.

## Ближайшие прикладные хвосты

- консолидация tracked current state в git без потери канонических материалов;
- финальная упаковка пояснительной записки и презентации;
- точечный cleanup residue-файлов и generated noise;
- необязательные future refactor-пункты вроде `snapshot.py` только если они начнут реально мешать.
