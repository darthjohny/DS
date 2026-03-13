# Test Coverage Matrix

Дата фиксации: 13 марта 2026 года

Цель этого документа:
- не считать формальный coverage percentage
- а понять, какие зоны проекта реально проверяются, а какие пока только частично прикрыты

Статусы:
- `OK` — зона покрыта достаточно разумно для текущего этапа
- `TOLERABLE` — покрытие есть, но оно в основном smoke/contract-level
- `FIX` — заметный пробел, который позже стоит закрыть отдельными тестами

## Карта покрытия по зонам

| Зона | Текущие тесты | Статус | Комментарий |
|---|---|---|---|
| Router model | `test_gaussian_router.py`, `test_runtime_artifacts.py` | `OK` | Сильный набор на label contract, posterior scoring, OOD policy, CLI args и artifact roundtrip. |
| Host Gaussian model | `test_model_gaussian.py`, `test_runtime_artifacts.py` | `OK` | Основной runtime contract contrastive/legacy прикрыт хорошо. |
| Input layer | `test_input_layer_db_integration.py` | `FIX` | Есть ценные DB-backed тесты, но почти нет быстрых unit-like проверок богатой логики валидации. |
| Priority pipeline branching | `test_priority_pipeline.py`, `test_star_orchestrator.py` | `OK` | Базовая логика ветвления и low/unknown handling читаемо закреплена. |
| Priority pipeline orchestration | `test_priority_pipeline.py`, `test_priority_pipeline_db_integration.py` | `TOLERABLE` | Есть smoke и e2e с Postgres, но внутренние слои decision/persist/input_data покрыты слабо. |
| Decision layer math | `test_star_orchestrator.py`, `test_decision_layer_calibrator.py` | `TOLERABLE` | Есть позитивные сценарии и проверка на host_posterior, но крайние численные случаи почти не тестируются. |
| Decision calibration runtime/reporting | `test_decision_layer_calibrator.py`, `test_decision_calibration_reporting.py` | `TOLERABLE` | Публичный фасад прикрыт, но пакетные внутренности не тестируются глубоко. |
| Comparison-layer contracts | `test_model_comparison_data.py`, `test_model_comparison_tuning.py`, `test_model_comparison_reporting.py`, `test_model_comparison_cli.py` | `OK` | Split/CV/refit/reporting/CLI контракт зафиксирован хорошо. |
| Comparison model wrappers | `test_model_comparison_contrastive.py`, `test_model_comparison_legacy_gaussian.py`, `test_model_comparison_random_forest.py`, `test_model_comparison_mlp.py` | `TOLERABLE` | Есть smoke-проверки общего score contract, но глубоких regression tests по search outcomes немного. |
| Comparison snapshot | `test_model_comparison_snapshot.py`, часть `test_model_comparison_cli.py` | `TOLERABLE` | Artifact/save слой проверен, но full runtime path snapshot остаётся тяжёлым и глубже почти не стрессуется. |
| Infra / relations / logbooks | прямых тестов почти нет | `FIX` | Вспомогательный слой практически без прямого покрытия. |
| CLI compatibility facades | `test_gaussian_router.py`, `test_model_comparison_cli.py` | `TOLERABLE` | Ключевые фасады прикрыты, но не весь фасадный слой. |
| Production artifacts in `data/` | `test_runtime_artifacts.py` | `OK` | Ценный регрессионный барьер на реальные JSON artifacts. |

## Покрытие по типам тестов

| Тип | Что есть сейчас | Комментарий |
|---|---|---|
| Unit-like | Да | Особенно в router/model/comparison contracts. |
| Contract tests | Да | Сильная сторона comparison-layer и artifacts. |
| Smoke tests | Да | Есть для pipeline, comparison wrappers, snapshot reporting. |
| DB integration | Да | Три сценария на временной схеме Postgres. |
| CLI tests | Да | Router CLI и comparison CLI покрыты. |
| Artifact regression | Да | Production JSON artifacts и comparison CSV/markdown сохраняются под тестами. |
| Failure-path tests | Частично | Есть `raises` на config/data contracts, но неравномерно по production-слоям. |
| Numerical edge-case tests | Слабо | Именно здесь остаётся заметный пробел. |

## Основные пробелы по рискам

1. `input_layer.py`
Причина: модуль большой и критичный, а прямых быстрых тестов на локальные ветки валидации мало.

2. `priority_pipeline.decision` / `persist` / `input_data`
Причина: production core прикрыт orchestration smoke-тестами, но не очень локально и не очень диагностично.

3. `infra.*` и `logbooks.*`
Причина: вспомогательные модули почти не тестируются напрямую.

4. Численные edge cases в decision/calibration
Причина: много позитивных сценариев, но мало стресс-тестов на границах значений.

5. Глубина regression для `analysis/model_comparison/snapshot.py`
Причина: reporting/save слой проверен, но сам runtime snapshot путь остаётся сложным и в основном smoke-level.

## Краткий вывод

- Тесты в проекте есть и они полезны.
- Самые сильные зоны: router, host-model runtime contract, comparison contracts, DB-backed e2e, artifact regression.
- Самые слабые зоны: `input_layer`, внутренние production helpers pipeline, infra/logbooks, численная устойчивость decision/calibration.
- Зелёный `pytest` здесь означает “база хорошая”, но не означает “рисков почти не осталось”.
