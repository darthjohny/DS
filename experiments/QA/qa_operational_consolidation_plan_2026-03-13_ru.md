# Operational Consolidation Plan

Дата фиксации: 13 марта 2026 года

## Зачем нужен этот план

После QA-аудита и серии точечных исправлений проект пришёл в состояние,
где:

- четыре promotion wave уже отдельно проверены и признаны
  `promotion-ready`;
- в рабочем дереве одновременно есть:
  - важные tracked-изменения в production/runtime-слое;
  - большой untracked current-state блок;
  - historical материалы, которые не должны случайно смешаться
    с каноническим состоянием.

Этот документ раскладывает дальнейшую консолидацию не по принципу
"собрать всё сразу", а по практическим батчам.

## Что уже считается подготовленным

На момент составления плана уже подтверждены:

- [qa_promotion_wave1_2026-03-13_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/QA/qa_promotion_wave1_2026-03-13_ru.md)
- [qa_promotion_wave2_2026-03-13_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/QA/qa_promotion_wave2_2026-03-13_ru.md)
- [qa_promotion_wave3_2026-03-13_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/QA/qa_promotion_wave3_2026-03-13_ru.md)
- [qa_promotion_wave4_2026-03-13_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/QA/qa_promotion_wave4_2026-03-13_ru.md)

Это означает, что quality-check на уровне блоков уже есть. Следующий
вопрос не в том, "достаточно ли это проверено", а в том, как
последовательно собрать из этого осмысленный versioned current state.

## Общие правила консолидации

1. Не смешивать runtime foundation и generated artifacts в один батч.
2. Не тянуть historical planning docs в первые технические батчи.
3. Не продвигать локальные EDA-выгрузки, если они не являются
   каноническими путями проекта.
4. Каждый батч должен иметь собственный acceptance-check.
5. Порядок важен: сначала runtime и code/test слои, потом docs/notebooks,
   и только после этого generated artifacts.

## Батч 1. Tracked Runtime Foundation

### Что входит

Tracked-изменения, которые уже являются частью реального runtime
контракта и не должны оставаться "висящими":

- `src/router_model/__init__.py`
- `src/router_model/artifacts.py`
- `src/router_model/cli.py`
- `src/router_model/fit.py`
- `src/router_model/labels.py`
- `src/router_model/score.py`
- `src/router_model/ood.py`
- `src/priority_pipeline/__init__.py`
- `src/priority_pipeline/constants.py`
- `src/priority_pipeline/decision.py`
- `src/priority_pipeline/input_data.py`
- `src/priority_pipeline/pipeline.py`
- `src/priority_pipeline/branching.py`
- `src/priority_pipeline/frame_contract.py`
- `src/decision_calibration/__init__.py`
- `src/decision_calibration/cli.py`
- `src/decision_calibration/reporting.py`
- `src/decision_calibration/runtime.py`
- `src/decision_calibration/scoring.py`
- `src/decision_layer_calibrator.py`
- `src/gaussian_router.py`
- `src/star_orchestrator.py`
- `src/host_model/__init__.py`
- `src/host_model/db.py`
- `src/host_model/fit.py`
- `src/host_model/training_data.py`
- `data/router_gaussian_params.json`

### Зачем первым

- это реальный production/runtime слой;
- часть из этих изменений уже завязана на новые tests и docs;
- держать его отдельно от comparison-wave безопаснее, чем собирать всё
  одним большим куском.

### Acceptance-check

- `ruff` по `src/` и связанным test-модулям;
- `mypy` по `src/`;
- целевой `pytest` по runtime-пакетам:
  - `tests/test_gaussian_router.py`
  - `tests/test_router_ood.py`
  - `tests/test_priority_pipeline.py`
  - `tests/test_priority_pipeline_branching.py`
  - `tests/test_priority_pipeline_persist.py`
  - `tests/test_priority_pipeline_relations.py`
  - `tests/test_priority_pipeline_facade.py`
  - `tests/test_decision_layer_calibrator.py`
  - `tests/test_decision_calibration_reporting.py`
  - `tests/test_decision_layer_calibrator_facade.py`
  - `tests/test_star_orchestrator.py`

## Батч 2. Comparison Code/Test Wave

### Что входит

- весь пакет `analysis/model_comparison/`
- `src/model_comparison.py`
- comparison-focused tests:
  - `tests/test_model_comparison_cli.py`
  - `tests/test_model_comparison_contrastive.py`
  - `tests/test_model_comparison_data.py`
  - `tests/test_model_comparison_facade.py`
  - `tests/test_model_comparison_legacy_gaussian.py`
  - `tests/test_model_comparison_mlp.py`
  - `tests/test_model_comparison_random_forest.py`
  - `tests/test_model_comparison_reporting.py`
  - `tests/test_model_comparison_snapshot.py`
  - `tests/test_model_comparison_tuning.py`

### Почему отдельным батчем

- это уже цельный research-layer;
- он методически проверен как Wave 1;
- его удобно продвигать независимо от production runtime foundation.

### Acceptance-check

- `ruff` по `analysis/model_comparison`, `src/model_comparison.py`,
  `tests/test_model_comparison_*.py`;
- `mypy` по `analysis/model_comparison` и `src/model_comparison.py`;
- `pytest -q tests/test_model_comparison_*.py`.

## Батч 3. Repo Policy и Shared Hygiene

### Что входит

- `.gitignore`
- `README.md`
- `requirements.txt`
- `analysis/host_eda/__init__.py`
- `analysis/router_eda/__init__.py`
- `data/README.md`
- `docs/repository_state_policy_ru.md`
- `experiments/Логи работы программы/README.md`
- `experiments/Логи калибровки decision_layer/README.md`

### Почему отдельно

- это клей между runtime, docs и local-noise policy;
- сюда же относится фиксация того, что historical comparison waves и
  local log iterations не считаются каноническим current state.

### Acceptance-check

- локальные markdown-ссылки целы;
- `git check-ignore` подтверждает новые ignore-правила;
- README не ссылается на несуществующие файлы.

## Батч 4. Current-State Docs и Presentation

### Что входит

- `docs/model_comparison_protocol_ru.md`
- `docs/model_comparison_findings_ru.md`
- `docs/preprocessing_pipeline_ru.md`
- `docs/notebook_review_2026-03-13_ru.md`
- `docs/vkr_requirements_traceability_ru.md`
- `docs/assets/README.md`
- `docs/assets/gaia_archive_crossmatch_ui.png`
- `docs/assets/gaia_archive_validation_ui.png`
- `docs/presentation/vkr_slides_draft_ru.md`
- `docs/presentation/assets/baseline_comparison_2026-03-13_vkr30_cv10/*`

### Почему не раньше

- эти документы уже ссылаются на comparison artifacts и notebook-state;
- безопаснее продвигать их после code/runtime слоёв.

### Acceptance-check

- локальные markdown-ссылки внутри docs и draft-slides целы;
- current-state run-name везде один и тот же:
  `baseline_comparison_2026-03-13_vkr30_cv10` и
  `baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot`;
- все referenced presentation assets существуют.

## Батч 5. Canonical Notebooks и SQL/ADQL

### Что входит

- `notebooks/eda/00_data_extraction_and_preprocessing.ipynb`
- `notebooks/eda/01_host_eda_overview.ipynb`
- `notebooks/eda/02_router_readiness.ipynb`
- `notebooks/eda/03_host_vs_field_contrastive.ipynb`
- `notebooks/eda/04_model_comparison_summary.ipynb`
- `sql/2026-03-13_gaia_results_unknown_constraints.sql`
- `sql/2026-03-13_gaia_results_unknown_constraints.rollback.sql`
- `sql/adql/*`
- `sql/preprocessing/*`

### Специальная оговорка

Каталог `notebooks/eda/data/` не включается в этот батч.

Причина:

- EDA-модули и ноутбуки используют канонический путь `data/eda/...`,
  а не `notebooks/eda/data/...`;
- текущий `notebooks/eda/data/` выглядит как локальная duplicate-copy
  EDA-выгрузок, а не как обязательный versioned state.

### Acceptance-check

- `00` и `04` успешно исполняются через `nbconvert --execute --inplace`;
- `01`, `02`, `03` имеют корректный bootstrap с явным добавлением `src/`;
- SQL/ADQL-файлы имеют осмысленные имена и соответствуют README/docs.

## Батч 6. Canonical Generated Artifacts и QA Wave

### Что входит

- `experiments/model_comparison/README.md`
- всё canonical family `baseline_comparison_2026-03-13_vkr30_cv10*`
- `experiments/QA/README.md`
- `experiments/QA/qa_full_audit_log_2026-03-13_ru.md`
- `experiments/QA/qa_backlog_and_decision_map_2026-03-13_ru.md`
- `experiments/QA/qa_tracking_manifest_2026-03-13_ru.md`
- `experiments/QA/qa_promotion_wave1_2026-03-13_ru.md`
- `experiments/QA/qa_promotion_wave2_2026-03-13_ru.md`
- `experiments/QA/qa_promotion_wave3_2026-03-13_ru.md`
- `experiments/QA/qa_promotion_wave4_2026-03-13_ru.md`
- `experiments/QA/qa_file_ledger_python_2026-03-13_ru.md`
- `experiments/QA/qa_test_coverage_matrix_2026-03-13_ru.md`
- `experiments/QA/qa_runbook_2026-03-13_ru.md`
- `experiments/QA/qa_docs_ledger_2026-03-13_ru.md`
- `experiments/QA/qa_notebooks_sql_ledger_2026-03-13_ru.md`
- `experiments/QA/qa_artifacts_cleanup_2026-03-13_ru.md`
- `experiments/QA/qa_operational_consolidation_plan_2026-03-13_ru.md`

### Почему последним

- это generated и narrative-heavy слой;
- он зависит от того, что code/docs/notebooks уже определены как
  canonical current state.

### Acceptance-check

- family `vkr30_cv10` полная и внутренно согласованная;
- QA docs имеют целые ссылки;
- canonical generated artifacts явно отделены от historical волн.

## Батч 7. Historical Reference Layer

### Что входит

Исторические planning docs и исторические QA-материалы, которые уже
связаны с README/docs, но не должны маскироваться под current state:

- `docs/ood_unknown_tz_ru.md`
- `docs/ood_unknown_baselines_tz_ru.md`
- `docs/preprocessing_and_comparison_tz_ru.md`
- `docs/documentation_audit_tz_ru.md`
- `experiments/QA/qa_mvp_report_2026-03-11.md`

### Решение по этому батчу

Допустимы только два сценария:

1. файлы осознанно продвигаются как historical reference layer;
2. ссылки на них убираются из README/current-state docs.

Не допускается промежуточное состояние, в котором README ссылается на
historical документы, которых нет в versioned состоянии.

### Acceptance-check

- либо файлы явно присутствуют и помечены как historical;
- либо ссылки на них полностью убраны из current-state narrative.

## Что не входит в operational promotion по умолчанию

- `notebooks/eda/data/` как duplicate local EDA-copy;
- local log iterations `experiments/Логи калибровки decision_layer/iteration_*`;
- historical comparison waves вне `vkr30_cv10`;
- `.DS_Store`, `__pycache__`, `.pytest_cache/`, `.mypy_cache/`,
  `.ruff_cache/`, `.pyright/`, `.ipynb_checkpoints/`;
- generic screenshots вида `docs/assets/Снимок экрана *.png`.

## Практический порядок выполнения

1. Батч 1
2. Батч 2
3. Батч 3
4. Батч 4
5. Батч 5
6. Батч 6
7. Батч 7

## Главный вывод

Operational consolidation теперь должен идти не как "чистка всего
подряд", а как последовательная сборка:

- сначала runtime foundation;
- потом comparison-layer;
- потом policy/docs/notebooks;
- и только после этого generated artifacts и historical layer.
