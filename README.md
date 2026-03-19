# DSPro VKR

Проект по физически согласованной приоритизации наблюдений за звёздами с целью построения `host-prioritization score` для поиска кандидатов в звёзды-хосты экзопланет.

## Что делает проект

Текущая MVP-архитектура решает задачу в три шага:

1. `router` распознаёт звезду по `teff_gspphot`, `logg_gspphot`, `radius_gspphot` и относит её к физическому классу.
2. `host-модель` внутри ветки `M/K/G/F dwarf` сравнивает распознанную звезду со звёздами, у которых уже открыты экзопланеты, и оценивает `host vs field`.
3. `decision layer` объединяет физический класс, host-score и коэффициенты наблюдательной пригодности, чтобы сформировать итоговый `final_score` для follow-up ранжирования.

Основные внешние архивы данных:

- [Gaia Archive](https://gea.esac.esa.int/archive/) — архив Gaia ESA;
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) — архив подтверждённых экзопланет и звёзд-хостов NASA.

## Текущий статус

- production-контур уже использует posterior-aware `router`, contrastive `host vs field` и decision layer с quality-факторами;
- comparison-layer доведён до формального ВКР-контракта: `test_size = 0.30`, `10-fold` CV и compact hyperparameter search;
- поверх benchmark добавлен отдельный validation-layer: dataset preflight, generalization diagnostics и per-model audit;
- текущая версия проекта — рабочая научная MVP, а не финальная астрофизическая теория или окончательная публикационная модель;
- latest full QA wave зафиксирована в `experiments/QA/` и подтверждает, что у проекта сейчас нет красных флагов по базовой математике и архитектуре.

## Где смотреть сначала

- если нужен production-контур: `src/router_model`, `src/host_model`, `src/priority_pipeline`, `src/decision_calibration`;
- если нужен research-layer: `analysis/model_comparison`, `analysis/model_validation`, `analysis/host_eda`, `analysis/router_eda`;
- если нужен current state policy: `docs/repository_state_policy_ru.md`;
- если нужен канон orchestrator-а: `docs/orchestrator_host_prioritization_canon_ru.md`;
- если нужны канонические comparison-артефакты: `experiments/model_comparison/README.md`;
- если нужны QA findings и backlog: `experiments/QA/README.md`.

## Актуальная архитектура

Каноническая логика живёт в пакетах:

- `src/router_model` — обучение, артефакты и scoring физического router;
- `src/host_model` — legacy и контрастивная host-модель;
- `src/priority_pipeline` — боевой pipeline приоритизации;
- `src/decision_calibration` — офлайн-калибровка decision layer;
- `analysis/host_eda` и `analysis/router_eda` — исследовательский контур EDA.

Исторический top-level shim layer убран. Канонический запуск и импорт
теперь идут через installable пакеты и `python -m ...`, без промежуточных
фасадов в `src/*.py`.

## Текущий scoring-контракт

Канонический научный ориентир для следующей волны работ зафиксирован в:

- `docs/orchestrator_host_prioritization_canon_ru.md`

- `router`: выбор класса идёт по `router_log_posterior`, а не по одному расстоянию Mahalanobis;
- `host-модель`: основной ограниченный score — `host_posterior`, диагностический raw score — `host_log_lr`;
- `decision layer`: боевая формула имеет вид:

```text
host_score =
  host_posterior
  × class_prior
  × metallicity_factor

reliability_factor =
  quality_factor
  = avg(ruwe_factor, parallax_precision_factor)

followup_factor =
  distance_factor(parallax)

final_score =
  host_score
  × reliability_factor
  × followup_factor
  × color_factor
  × validation_factor
```

Текущая `V1` operational tier mapping:

```text
HIGH   : final_score >= 0.50
MEDIUM : final_score >= 0.30
LOW    : final_score <  0.30
```

Важно:

- в runtime `quality_factor` сейчас сохраняется как совместимый alias
  для `reliability_factor`;
- observability-логика в current production уже разделена на
  `reliability_factor` и `followup_factor`;
- `HIGH` в current `V1` трактуется как консервативный follow-up shortlist
  и практически состоит из `K` и верхушки `M`, тогда как `G` в основном
  остаются резервным `MEDIUM`-слоем;
- current `UNKNOWN` в production трактуется как router/OOD reject для
  уже scoreable строк; structurally incomplete объекты отфильтровываются
  input-layer раньше и не входят в `unknown_share`;
- офлайн-калибровка пока живёт на близкой, но не полностью идентичной
  формуле.

## Operational export

Comparison snapshot и production shortlist в текущем состоянии проекта
разведены.

- comparison-layer пишет benchmark и snapshot-preview в
  `experiments/model_comparison/`;
- production-like shortlist основной `V1` пишется отдельным export-слоем в
  `experiments/QA/production_runs/`.

Канонический operational export для текущей калиброванной версии:

- `experiments/QA/production_runs/production_priority_2026-03-19_v1_calibrated_limit5000.md`
- `experiments/QA/production_runs/production_priority_2026-03-19_v1_calibrated_limit5000_shortlist.csv`
- `experiments/QA/production_runs/production_priority_2026-03-19_v1_calibrated_limit5000_shortlist_summary.csv`

Команда пересборки:

```bash
python -m priority_pipeline.export \
  --run-name production_priority_2026-03-19_v1_calibrated_limit5000 \
  --limit 5000
```

## Какие входные поля реально используются

Канонический входной relation сейчас загружается с такими полями:

- идентификация: `source_id`, `ra`, `dec`
- router / host-model признаки: `teff_gspphot`, `logg_gspphot`, `radius_gspphot`
- decision-layer признаки: `mh_gspphot`, `parallax`, `parallax_over_error`, `ruwe`, `bp_rp`
- технический modifier: `validation_factor`

Важно:

- отдельной сырой колонки `distance` во входной таблице сейчас нет;
- удалённость выводится из `parallax` или используется через proxy-логику;
- `router` и `host-модель` не используют `metallicity`, `ruwe` или `parallax` напрямую.

| Поле | Берётся из relation | Кто использует | Роль |
| --- | --- | --- | --- |
| `source_id` | да | все слои | идентификатор объекта |
| `ra`, `dec` | да | output/persist/reporting | координаты и выдача результата |
| `teff_gspphot` | да | `router`, `host_model` | базовый физический признак |
| `logg_gspphot` | да | `router`, `host_model` | базовый физический признак |
| `radius_gspphot` | да | `router`, `host_model` | базовый физический признак |
| `mh_gspphot` | да | `decision layer` | мягкий astrophysical prior |
| `parallax` | да | `decision layer`, `decision_calibration` | proxy близости / basis для distance-like factor |
| `parallax_over_error` | да | `decision layer`, `decision_calibration` | надёжность расстояния |
| `ruwe` | да | `decision layer`, `decision_calibration` | астрометрическое качество |
| `bp_rp` | да | `decision layer` | мягкий color-based modifier |
| `validation_factor` | да | `decision layer` | технический modifier качества данных |
| `predicted_spec_class` | нет, derived | `router -> host_model -> decision layer` | результат router-классификации |
| `predicted_evolution_stage` | нет, derived | `router -> branching -> decision layer` | результат router-классификации |
| `host_posterior` | нет, derived | `host_model -> decision layer` | основной model score |
| `distance_pc` | нет, derived | `decision_calibration` | вычисляется из `parallax`, не хранится как raw input |

Если упростить архитектурно:

- `router` и `host_model` принимают решение по тройке `teff_gspphot / logg_gspphot / radius_gspphot`;
- `decision layer` уже поверх этого учитывает наблюдательную пригодность и мягкие astrophysical priors;
- `distance` сейчас является не отдельным входным столбцом, а derived-величиной поверх `parallax`.

## Структура репозитория

```text
src/
  input_layer/                  # валидация входного relation и registry
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
  model_validation/             # heavy validation repeated split и generalization layer

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

Каноническая логика и CLI теперь живут в пакетах. Top-level фасады больше
не являются частью текущей архитектуры проекта.

## Быстрый старт и основные сценарии

Подготовка окружения:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` включает канонический package CLI без `PYTHONPATH=src`.

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
python -m devtools.db_smoke
```

EDA router-слоя:

```bash
python -m analysis.router_eda
```

EDA host-модели:

```bash
python -m analysis.host_eda
```

Обучение contrastive host-модели:

```bash
python -m host_model --mode contrastive
```

Preview боевого pipeline:

```bash
python -m priority_pipeline
```

Офлайн-калибровка:

```bash
python -m decision_calibration --relation public.gaia_dr3_training
```

Сравнение основной модели и baseline:

```bash
python -m analysis.model_comparison
```

По умолчанию команда делает два артефактных контура:

- supervised benchmark на `host vs field`;
- live snapshot на `public.gaia_dr3_training` после общего `router + OOD`.

Canonical benchmark-режим теперь соответствует формальному протоколу ВКР:

- `test_size = 0.30`;
- `10-fold` stratified CV внутри train split;
- compact search для всех четырёх моделей;
- отдельный `search_summary.csv` с best params и best CV score;
- отдельный dataset validation report до model fitting;
- отдельный generalization audit report после benchmark.

В benchmark сейчас участвуют четыре model head:

- `main_contrastive_v1`
- `baseline_legacy_gaussian`
- `baseline_random_forest`
- `baseline_mlp_small`

Полезные overrides:

```bash
python -m analysis.model_comparison --snapshot-limit 5000 --snapshot-top-k 25
python -m analysis.model_comparison --skip-snapshot
python -m analysis.model_comparison --cv-folds 5 --search-refit-metric pr_auc
```

Каноническое поколение comparison-артефактов для ВКР:

- benchmark + snapshot preview: `baseline_comparison_2026-03-19_v1_calibrated_limit5000`

Исторические волны этого же дня сохранены в
`experiments/model_comparison/`, но текущим источником истины считаются
только файлы поколения `v1_calibrated_limit5000`. Это отдельно
зафиксировано в:

- `experiments/model_comparison/README.md`

## Model Validation / Generalization Checks

Comparison-layer теперь состоит из трёх разных слоёв:

- `benchmark` — supervised comparison на фиксированном `host vs field` split;
- `validation` — dataset gate и per-model generalization audit;
- `snapshot` — operational preview на живом batch после `router + OOD`.

Отдельный package под heavy validation wave живёт в
`analysis.model_validation`, а его артефакты — в
`experiments/model_validation/`.

Validation-layer сейчас делает:

- dataset preflight до model fitting;
- проверку `train/test` overlap и stratify integrity;
- fold-stability summary по CV;
- `Generalization Diagnostics` по `train/test` и `CV/test`;
- `Per-model Generalization Audit` с итоговым risk verdict.

Heavy validation run поверх benchmark запускается отдельно:

```bash
python -m analysis.model_validation --run-name validation_20260314 --mode fast
```

Этот контур требует тот же DB-backed benchmark dataset, что и
`analysis.model_comparison`, и пишет отдельные heavy validation артефакты:

- `*_validation_report.md`
- `*_repeated_splits.csv`
- `*_model_summary.csv`
- `*_generalization_summary.csv`
- `*_gap_diagnostics.csv`
- `*_risk_audit.csv`
- `*_plots/`

Основные validation-артефакты comparison-layer:

- `*_dataset_validation.md`
- `*_dataset_validation_summary.csv`
- `*_dataset_validation_stratify.csv`
- `*_dataset_validation_feature_drift.csv`
- `*_generalization.csv`
- `*_generalization_audit.csv`
- `*_generalization_audit.md`

Канонические внутренние документы:

- `docs/model_validation_protocol_ru.md`
- `docs/model_validation_dependency_spike_ru.md`
- `docs/model_comparison_protocol_ru.md`

Официальные внешние источники validation design:

- [scikit-learn: learning_curve](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html)
- [scikit-learn: validation curves](https://scikit-learn.org/stable/modules/learning_curve.html)
- [scikit-learn: nested vs non-nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
- [Deepchecks: installation](https://docs.deepchecks.com/stable/getting-started/installation.html)
- [Deepchecks: train-test validation suite](https://docs.deepchecks.com/stable/api/generated/deepchecks.tabular.suites.train_test_validation.html)
- [Deepchecks: train-test validation checks](https://docs.deepchecks.com/stable/api/generated/deepchecks.tabular.checks.train_test_validation.html)

Важная оговорка:

- основной validation-layer проекта строится на `scikit-learn` и typed contracts проекта;
- `Deepchecks` сейчас не входит в обязательный dependency stack из-за compatibility-рисков в текущем окружении Python `3.13` + `scikit-learn 1.8.0`;
- решение зафиксировано в `docs/model_validation_dependency_spike_ru.md`.

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
