# Promotion Wave 4: canonical comparison artifacts

Дата фиксации: 13 марта 2026 года

## Назначение

Этот документ фиксирует четвёртую promotion wave для versioned current
state проекта.

Wave 4 закрывает последний крупный слой current-state материалов:

- канонические generated artifacts benchmark-контура;
- operational snapshot preview для `vkr30_cv10`;
- README/policy этого каталога как источника истины для findings,
  notebooks и presentation.

## Состав wave

### Canonical benchmark artifacts

- `experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10.md`
- `experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10_summary.csv`
- `experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10_classwise.csv`
- `experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10_search_summary.csv`
- `experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10_*_train_scores.csv`
- `experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10_*_test_scores.csv`

### Canonical snapshot preview artifacts

- `experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot.md`
- `experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot_summary.csv`
- `experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot_router.csv`
- `experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot_*_priority.csv`
- `experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot_*_top.csv`

### Policy/index file

- `experiments/model_comparison/README.md`

## Что проверялось

Для этой wave проверялись:

1. наличие полного file family для benchmark и snapshot;
2. согласованность model set между:
   - `summary.csv`
   - `classwise.csv`
   - `search_summary.csv`
   - `snapshot_summary.csv`
3. наличие всех per-model train/test score files;
4. наличие всех per-model snapshot `priority/top` files;
5. согласованность README/policy и ссылок из docs/presentation в этот слой.

## Проверки

Выполнены следующие проверки:

```bash
./venv/bin/python - <<'PY'
# validate canonical vkr30_cv10 file family and model sets
PY

sed -n '1,240p' \
  experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10.md

sed -n '1,240p' \
  experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot.md

./venv/bin/python - <<'PY'
# link-check for README/findings/protocol/slides references into model_comparison artifacts
PY
```

Результат:

- canonical `vkr30_cv10` file family: `present`
- summary/search/classwise/snapshot model sets: `consistent`
- per-model train/test files: `present`
- per-model snapshot priority/top files: `present`
- local markdown links into canonical artifacts: `green`

## Подтверждённые свойства canonical wave

- каноническая family содержит `23` файлов с префиксом
  `baseline_comparison_2026-03-13_vkr30_cv10*`;
- benchmark summary содержит обе split-ветки: `train` и `test`;
- во всех ключевых CSV фигурируют одни и те же четыре модели:
  - `main_contrastive_v1`
  - `baseline_legacy_gaussian`
  - `baseline_random_forest`
  - `baseline_mlp_small`
- snapshot summary и markdown согласованы по `input_rows`, `router_rows`,
  `host_candidates`, `low_known_rows`, `unknown_rows`.

## Статус

Статус wave: `promotion-ready`.

Это означает:

- canonical generated artifacts больше не выглядят как неоформленный
  local output;
- этот слой можно рассматривать как полноценную часть versioned current
  state проекта;
- все четыре главные promotion waves теперь сформированы отдельно и
  готовы к общей operational consolidation.

## Что не входит в wave

В эту wave сознательно не включены:

- historical comparison waves:
  `baseline_comparison_2026-03-13*`,
  `baseline_comparison_2026-03-13_mlp*`,
  `baseline_comparison_2026-03-13_snapshot*`;
- ранние промежуточные артефакты того же дня;
- presentation asset copies в `docs/presentation/assets/`.

Эти слои либо уже покрыты другими waves, либо относятся к historical
research history, а не к canonical generated state.

## Следующий шаг

После Wave 4 можно переходить уже не к ещё одной promotion wave, а к
общей operational current-state consolidation:

- собрать все `promotion-ready` waves в единый practical plan;
- решить, что именно продвигается в tracked state первым;
- только после этого обсуждать staging/commit strategy.
