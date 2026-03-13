# Promotion Wave 1: comparison-layer core

Дата фиксации: 13 марта 2026 года

## Назначение

Этот документ фиксирует первую осмысленную promotion wave для
versioned current state проекта.

Смысл wave:

- не пытаться сразу продвинуть в tracked state весь большой `untracked`
  слой;
- сначала выделить один цельный технический блок;
- подтвердить, что он внутренне согласован, типизирован и покрыт тестами;
- только после этого использовать его как основу для дальнейшей
  консолидации current state.

## Состав wave

### Research and comparison code

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

### Support code

- `src/model_comparison.py`
- `src/router_model/ood.py`
- `src/priority_pipeline/branching.py`
- `src/priority_pipeline/frame_contract.py`

### Tests

- `tests/test_input_layer.py`
- `tests/test_decision_calibration_reporting.py`
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
- `tests/test_priority_pipeline_branching.py`
- `tests/test_router_ood.py`

## Проверки

Для этой wave выполнены следующие проверки:

```bash
./venv/bin/python -m ruff check \
  analysis/model_comparison \
  src/model_comparison.py \
  src/router_model/ood.py \
  src/priority_pipeline/branching.py \
  src/priority_pipeline/frame_contract.py \
  tests/test_model_comparison_*.py \
  tests/test_input_layer.py \
  tests/test_decision_calibration_reporting.py

./venv/bin/python -m mypy \
  analysis/model_comparison \
  src/model_comparison.py \
  src/router_model/ood.py \
  src/priority_pipeline/branching.py \
  src/priority_pipeline/frame_contract.py \
  tests/test_model_comparison_cli.py \
  tests/test_model_comparison_contrastive.py \
  tests/test_model_comparison_data.py \
  tests/test_model_comparison_facade.py \
  tests/test_model_comparison_legacy_gaussian.py \
  tests/test_model_comparison_mlp.py \
  tests/test_model_comparison_random_forest.py \
  tests/test_model_comparison_reporting.py \
  tests/test_model_comparison_snapshot.py \
  tests/test_model_comparison_tuning.py \
  tests/test_input_layer.py \
  tests/test_decision_calibration_reporting.py

./venv/bin/python -m pytest -q \
  tests/test_model_comparison_*.py \
  tests/test_model_comparison_facade.py \
  tests/test_router_ood.py \
  tests/test_priority_pipeline_branching.py \
  tests/test_input_layer.py \
  tests/test_decision_calibration_reporting.py
```

Результат:

- `ruff`: green
- `mypy`: green, `30 source files`
- `pytest`: green, `45 passed`

## Почему этот блок выбран первым

- это уже канонический comparison-layer под ВКР;
- он изолирован от production pipeline и не тащит за собой DB migration;
- он включает не только модели, но и support-слой:
  `OOD`, `branching`, `frame contract`, façade и targeted tests;
- именно этот блок чаще всего используется в findings, notebook,
  presentation assets и docs.

## Статус

Статус wave: `promotion-ready`.

Это означает:

- блок технически согласован;
- по нему уже есть стабильный QA-контур;
- он может рассматриваться как первый кандидат на консолидацию в
  versioned current state без дополнительной большой волны исправлений.

## Что не входит в wave

В эту wave сознательно не включены:

- `docs/*`
- notebooks
- SQL/ADQL
- canonical experiment artifacts
- полный QA-архив

Для них лучше делать отдельные promotion waves, чтобы не смешивать
research code, documentation, data lineage и generated outputs в один
большой шаг.

## Следующий кандидат на wave

Следующий естественный блок:

- current-state docs и presentation materials;
- затем notebooks/SQL;
- затем canonical experiment artifacts.
