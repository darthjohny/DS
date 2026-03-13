# QA Tracking Manifest

Дата фиксации: 13 марта 2026 года

## Зачем нужен этот manifest

После полной QA-волны и серии точечных исправлений в рабочем дереве
остался большой `untracked` слой. Важно: это уже не в основном мусор, а
живой current state проекта.

Цель этого документа:

- отделить канонические материалы от generated history;
- зафиксировать, что должно войти в versioned current state;
- не пытаться уменьшать `git status` ценой потери полезных артефактов.

## 1. Must Track Now

Эти пути относятся к текущему состоянию проекта и не должны считаться
временными.

### Статус promotion waves

- первая осмысленная code/test wave уже отдельно собрана и подтверждена в:
  [qa_promotion_wave1_2026-03-13_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/QA/qa_promotion_wave1_2026-03-13_ru.md);
- current-state docs/presentation wave отдельно подтверждена в:
  [qa_promotion_wave2_2026-03-13_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/QA/qa_promotion_wave2_2026-03-13_ru.md);
- notebooks и SQL/ADQL wave отдельно подтверждена в:
  [qa_promotion_wave3_2026-03-13_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/QA/qa_promotion_wave3_2026-03-13_ru.md);
- canonical comparison artifacts wave отдельно подтверждена в:
  [qa_promotion_wave4_2026-03-13_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/QA/qa_promotion_wave4_2026-03-13_ru.md);
- это не означает автоматический `git add`, но означает, что блок уже
  валидирован как внутренне согласованный current-state candidate.

### 1.1 Research and comparison code

- `analysis/model_comparison/__init__.py`
- `analysis/model_comparison/cli.py`
- `analysis/model_comparison/contracts.py`
- `analysis/model_comparison/contrastive.py`
- `analysis/model_comparison/data.py`
- `analysis/model_comparison/legacy_gaussian.py`
- `analysis/model_comparison/manual_search.py`
- `analysis/model_comparison/metrics.py`
- `analysis/model_comparison/mlp_baseline.py`
- `analysis/model_comparison/presentation_assets.py`
- `analysis/model_comparison/random_forest.py`
- `analysis/model_comparison/reporting.py`
- `analysis/model_comparison/snapshot.py`
- `analysis/model_comparison/tuning.py`

### 1.2 Production/research support code

- `src/host_model/training_data.py`
- `src/model_comparison.py`
- `src/priority_pipeline/branching.py`
- `src/priority_pipeline/frame_contract.py`
- `src/router_model/ood.py`

### 1.3 Tests

- `tests/test_input_layer.py`
- `tests/test_decision_calibration_reporting.py`
- `tests/test_model_comparison_cli.py`
- `tests/test_model_comparison_contrastive.py`
- `tests/test_model_comparison_data.py`
- `tests/test_model_comparison_legacy_gaussian.py`
- `tests/test_model_comparison_mlp.py`
- `tests/test_model_comparison_random_forest.py`
- `tests/test_model_comparison_reporting.py`
- `tests/test_model_comparison_snapshot.py`
- `tests/test_model_comparison_tuning.py`

### 1.4 Current-state docs

- `docs/model_comparison_protocol_ru.md`
- `docs/model_comparison_findings_ru.md`
- `docs/preprocessing_pipeline_ru.md`
- `docs/notebook_review_2026-03-13_ru.md`
- `docs/vkr_requirements_traceability_ru.md`
- `docs/repository_state_policy_ru.md`
- `docs/assets/README.md`
- `docs/assets/gaia_archive_crossmatch_ui.png`
- `docs/assets/gaia_archive_validation_ui.png`
- `docs/presentation/vkr_slides_draft_ru.md`
- `docs/presentation/assets/baseline_comparison_2026-03-13_vkr30_cv10/*`
- `data/README.md`

### 1.5 Notebooks and SQL

- `notebooks/eda/00_data_extraction_and_preprocessing.ipynb`
- `notebooks/eda/04_model_comparison_summary.ipynb`
- `sql/2026-03-13_gaia_results_unknown_constraints.sql`
- `sql/2026-03-13_gaia_results_unknown_constraints.rollback.sql`
- `sql/adql/*`
- `sql/preprocessing/*`

### 1.6 QA artifacts

- `experiments/QA/README.md`
- `experiments/QA/qa_artifacts_cleanup_2026-03-13_ru.md`
- `experiments/QA/qa_backlog_and_decision_map_2026-03-13_ru.md`
- `experiments/QA/qa_docs_ledger_2026-03-13_ru.md`
- `experiments/QA/qa_file_ledger_python_2026-03-13_ru.md`
- `experiments/QA/qa_full_audit_log_2026-03-13_ru.md`
- `experiments/QA/qa_notebooks_sql_ledger_2026-03-13_ru.md`
- `experiments/QA/qa_operational_consolidation_plan_2026-03-13_ru.md`
- `experiments/QA/qa_runbook_2026-03-13_ru.md`
- `experiments/QA/qa_test_coverage_matrix_2026-03-13_ru.md`

### 1.7 Canonical comparison artifacts

Обязательно как current state:

- `experiments/model_comparison/README.md`
- `experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10.md`
- `experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10_summary.csv`
- `experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10_classwise.csv`
- `experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10_search_summary.csv`
- `experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10_*_train_scores.csv`
- `experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10_*_test_scores.csv`
- `experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot.md`
- `experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot_summary.csv`
- `experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot_router.csv`
- `experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot_*_priority.csv`
- `experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot_*_top.csv`

## 2. Historical, Optional To Keep

Эти файлы не являются главным current state. Их можно хранить как
исследовательскую историю, но они не обязаны входить в tracked current
state по умолчанию.

### 2.1 Older comparison waves

- `experiments/model_comparison/baseline_comparison_2026-03-13.md`
- `experiments/model_comparison/baseline_comparison_2026-03-13_*`
- `experiments/model_comparison/baseline_comparison_2026-03-13_mlp*`
- `experiments/model_comparison/baseline_comparison_2026-03-13_snapshot*`

Практическая policy:

- по умолчанию эти generated waves могут оставаться локальными;
- canonical tracked current state — это только поколение `vkr30_cv10`;
- если historical wave нужна для narrative или appendix, её можно
  осознанно продвинуть в git вручную.

### 2.2 Earlier QA slice

- `experiments/QA/qa_mvp_report_2026-03-11.md`

## 3. Local Working Outputs, Not Promoted Yet

Сейчас это generated артефакты, которые не используются как канонический
current state и пока не имеют прямых ссылок из README/docs.

- `experiments/Логи калибровки decision_layer/iteration_014*`
- `experiments/Логи калибровки decision_layer/iteration_015*`

Если какая-то из этих итераций станет важной для narrative ВКР или QA,
её можно отдельно продвинуть в tracked history.

## 4. Local Noise

Это не нужно продвигать в tracked state:

- `.DS_Store`
- `__pycache__/`
- `.pytest_cache/`
- `.mypy_cache/`
- `.ruff_cache/`
- `.pyright/`
- `.ipynb_checkpoints/`
- `docs/assets/Снимок экрана *.png`
- `notebooks/eda/data/`

## 5. Practical Interpretation

Самый важный вывод из этого manifest:

- проблема рабочего дерева сейчас не в том, что в нём много мусора;
- проблема в том, что в нём много ценного current state, который ещё не
  консолидирован как versioned набор.

Поэтому следующий технический шаг должен быть не “чистить всё подряд”, а
аккуратно продвигать `Must Track Now` в каноническое tracked состояние.
