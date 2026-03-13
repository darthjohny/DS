# File Ledger: Python Code

Дата фиксации: 13 марта 2026 года

Покрытие этого ledger:
- весь `src`
- весь `analysis`
- тесты будут разобраны отдельно на шаге аудита тестового покрытия

Статусы:
- `OK` — оставляем как есть
- `TOLERABLE` — неидеально, но пока жить можно
- `FIX` — кандидат на отдельную правку
- `REMOVE?` — кандидат на удаление/слияние/вынос

## `src`

- [src/decision_calibration/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/decision_calibration/__init__.py): `OK` — нормальный публичный API пакета, без явной лишней логики.
- [src/decision_calibration/cli.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/decision_calibration/cli.py): `OK` — CLI orchestration выглядит уместно для пакета калибровки.
- [src/decision_calibration/config.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/decision_calibration/config.py): `TOLERABLE` — крупный конфигурационный модуль, но по задаче целостный; следить, чтобы не разрастался дальше.
- [src/decision_calibration/constants.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/decision_calibration/constants.py): `OK` — простой и уместный constants-модуль.
- [src/decision_calibration/reporting.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/decision_calibration/reporting.py): `TOLERABLE` — большой, но ответственность пока единая: summary, markdown и CSV артефакты итерации.
- [src/decision_calibration/runtime.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/decision_calibration/runtime.py): `OK` — runtime-слой выглядит на своём месте.
- [src/decision_calibration/scoring.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/decision_calibration/scoring.py): `OK` — математический контур калибровки отделён от runtime/reporting.
- [src/decision_layer_calibrator.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/decision_layer_calibrator.py): `OK` — чистый фасад совместимости, оформлен честно.

- [src/devtools/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/devtools/__init__.py): `OK` — нейтральный пакетный файл.
- [src/devtools/db_smoke.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/devtools/db_smoke.py): `OK` — понятный локальный smoke-tool, не выглядит лишним.

- [src/eda.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/eda.py): `TOLERABLE` — фасад совместимости полезен, но использует `sys.path`-трюк и широкую re-export surface.
- [src/gaussian_router.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/gaussian_router.py): `OK` — аккуратный фасад совместимости без собственной логики.

- [src/host_model/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/host_model/__init__.py): `OK` — сильный публичный API пакета.
- [src/host_model/artifacts.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/host_model/artifacts.py): `OK` — contracts/validation/JSON IO лежат там, где должны.
- [src/host_model/cli.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/host_model/cli.py): `OK` — CLI не выглядит перегруженным доменной логикой.
- [src/host_model/constants.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/host_model/constants.py): `OK` — аккуратный constants-модуль.
- [src/host_model/contrastive_score.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/host_model/contrastive_score.py): `OK` — канонический production scoring-path, оформлен чисто.
- [src/host_model/db.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/host_model/db.py): `FIX` — DB-layer сейчас знает про training preparation из `fit`, зависимость неидеальная.
- [src/host_model/fit.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/host_model/fit.py): `TOLERABLE` — крупный training-модуль, но по ответственности пока цельный.
- [src/host_model/gaussian_math.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/host_model/gaussian_math.py): `OK` — хороший отдельный math-слой.
- [src/host_model/legacy_score.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/host_model/legacy_score.py): `OK` — baseline-роль определена чётко.

- [src/infra/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/infra/__init__.py): `OK` — нейтральный package stub.
- [src/infra/db.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/infra/db.py): `OK` — общая DB-инфраструктура лежит на правильном уровне.
- [src/infra/logbook.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/infra/logbook.py): `OK` — узкий utility-модуль, без перегруза.
- [src/infra/relations.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/infra/relations.py): `OK` — relation helpers изолированы корректно.

- [src/input_layer.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/input_layer.py): `TOLERABLE` — очень большой файл, но ответственность у него пока единая и понятная.

- [src/logbooks/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/logbooks/__init__.py): `OK` — нейтральный package stub.
- [src/logbooks/decision_layer.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/logbooks/decision_layer.py): `OK` — уместный template-generator для журнала.
- [src/logbooks/program_run.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/logbooks/program_run.py): `OK` — уместный template-generator для журнала запусков.

- [src/model_comparison.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/model_comparison.py): `TOLERABLE` — удобный фасад, но широкая re-export surface и `sys.path`-adjustment делают его не совсем лёгким.
- [src/model_gaussian.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/model_gaussian.py): `OK` — корректный legacy facade.

- [src/priority_pipeline/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/__init__.py): `OK` — хороший публичный API пакета.
- [src/priority_pipeline/branching.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/branching.py): `OK` — ветвление выделено правильно и не смешано со scoring.
- [src/priority_pipeline/constants.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/constants.py): `OK` — объёмный, но уместный constants-модуль.
- [src/priority_pipeline/contracts.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/contracts.py): `OK` — компактный и адекватный контрактный слой.
- [src/priority_pipeline/decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/decision.py): `FIX` — сильный модуль, но есть неидеальная зависимость от `input_data.ensure_decision_columns`.
- [src/priority_pipeline/input_data.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/input_data.py): `TOLERABLE` — в целом ок, но содержит helper, который уже используется глубже в decision-layer.
- [src/priority_pipeline/persist.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/persist.py): `OK` — чистый persist-layer, отделён удачно.
- [src/priority_pipeline/pipeline.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/pipeline.py): `OK` — orchestration читается и не выглядит перегруженным.
- [src/priority_pipeline/relations.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/relations.py): `OK` — узкий DB helper.

- [src/router_eda.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_eda.py): `TOLERABLE` — полезный compatibility facade, но с теми же рисками `sys.path` и широкой re-export surface.

- [src/router_model/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/__init__.py): `OK` — сильный публичный API пакета.
- [src/router_model/artifacts.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/artifacts.py): `OK` — contracts и normalization изолированы хорошо.
- [src/router_model/cli.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/cli.py): `TOLERABLE` — CLI хороший, но с необходимым import-path workaround для прямого запуска.
- [src/router_model/db.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/db.py): `OK` — DB bootstrap и loading отделены нормально.
- [src/router_model/fit.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/fit.py): `OK` — training-модуль выглядит на своём месте.
- [src/router_model/labels.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/labels.py): `OK` — нормализация labels вынесена удачно.
- [src/router_model/math.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/math.py): `OK` — чистый math-layer.
- [src/router_model/ood.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/ood.py): `OK` — open-set policy отделена удачно.
- [src/router_model/score.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/score.py): `TOLERABLE` — математически сильный модуль, но scoring-layer тянет shared `FEATURES` из DB-модуля.

- [src/star_orchestrator.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/star_orchestrator.py): `TOLERABLE` — полезный facade совместимости, но не полностью прозрачный из-за локального wrapper `run_host_similarity()`.

## `analysis`

- [analysis/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/__init__.py): `OK` — нейтральный package stub.

- [analysis/host_eda/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/host_eda/__init__.py): `FIX` — package import меняет `sys.path`, это рабочий, но хрупкий side effect.
- [analysis/host_eda/cli.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/host_eda/cli.py): `TOLERABLE` — очень script-like CLI, но для EDA пока допустимо.
- [analysis/host_eda/constants.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/host_eda/constants.py): `OK` — нормальный модуль соглашений и путей.
- [analysis/host_eda/contrastive.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/host_eda/contrastive.py): `TOLERABLE` — полезный аналитический модуль, но в нём уже есть длинный вычислительный сценарий.
- [analysis/host_eda/data.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/host_eda/data.py): `OK` — загрузка данных отделена корректно.
- [analysis/host_eda/exports.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/host_eda/exports.py): `OK` — узкий export-layer.
- [analysis/host_eda/plots.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/host_eda/plots.py): `TOLERABLE` — объёмный plotting-модуль, но по задаче цельный.
- [analysis/host_eda/stats.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/host_eda/stats.py): `OK` — компактный stats-layer.

- [analysis/model_comparison/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/__init__.py): `TOLERABLE` — package API слишком широк, но пока управляем.
- [analysis/model_comparison/cli.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/cli.py): `TOLERABLE` — orchestrator насыщенный, но для канонического benchmark entrypoint это приемлемо.
- [analysis/model_comparison/contracts.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/contracts.py): `OK` — сильный контрактный центр comparison-layer.
- [analysis/model_comparison/contrastive.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/contrastive.py): `OK` — wrapper основной модели выглядит уместно.
- [analysis/model_comparison/data.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/data.py): `TOLERABLE` — довольно насыщенный data-layer, но пока без явного расползания.
- [analysis/model_comparison/legacy_gaussian.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/legacy_gaussian.py): `OK` — baseline wrapper оформлен корректно.
- [analysis/model_comparison/manual_search.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/manual_search.py): `OK` — полезный узкий helper для non-sklearn search.
- [analysis/model_comparison/metrics.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/metrics.py): `OK` — метрики отделены удачно.
- [analysis/model_comparison/mlp_baseline.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/mlp_baseline.py): `TOLERABLE` — длинный, но ответственность единая и логика блоками.
- [analysis/model_comparison/presentation_assets.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/presentation_assets.py): `TOLERABLE` — большой util-script, но изолирован и не лезет в доменную логику.
- [analysis/model_comparison/random_forest.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/random_forest.py): `OK` — baseline wrapper выглядит нормально.
- [analysis/model_comparison/reporting.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/reporting.py): `OK` — reporting-layer отделён и читаем.
- [analysis/model_comparison/snapshot.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/snapshot.py): `FIX` — главный кандидат на future refactor: внутри смешаны training reuse, scoring heads, reporting и orchestration.
- [analysis/model_comparison/tuning.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/tuning.py): `OK` — хороший узкий infrastructure helper.

- [analysis/router_eda/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/router_eda/__init__.py): `FIX` — package import меняет `sys.path`, это рабочий, но хрупкий side effect.
- [analysis/router_eda/cli.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/router_eda/cli.py): `TOLERABLE` — script-like EDA entrypoint, пока терпимо.
- [analysis/router_eda/constants.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/router_eda/constants.py): `OK` — компактный constants-layer.
- [analysis/router_eda/data.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/router_eda/data.py): `OK` — data-layer выделен корректно.
- [analysis/router_eda/exports.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/router_eda/exports.py): `OK` — узкий export-layer.
- [analysis/router_eda/plots.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/router_eda/plots.py): `TOLERABLE` — большой plotting-модуль, но ещё в пределах уместного.
- [analysis/router_eda/readiness.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/router_eda/readiness.py): `TOLERABLE` — ценный модуль, но с длинным расчётным сценарием.
- [analysis/router_eda/stats.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/router_eda/stats.py): `OK` — компактный stats-layer.

- [analysis/visual_theme.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/visual_theme.py): `OK` — уместный shared visual helper.

## Краткий итог ledger

- Production-код в целом сильнее и стабильнее по структуре, чем исследовательский.
- Основные `FIX` по Python-коду на текущем этапе:
  - `src/host_model/db.py`
  - `src/priority_pipeline/decision.py`
  - `analysis/host_eda/__init__.py`
  - `analysis/router_eda/__init__.py`
  - `analysis/model_comparison/snapshot.py`
- Основные `TOLERABLE` зоны:
  - большие, но ещё управляемые модули;
  - фасады с широкой re-export surface;
  - script-like EDA CLI;
  - compatibility wrappers с небольшим поведенческим долгом.
