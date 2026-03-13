# Model Comparison Artifacts

Дата актуализации: 13 марта 2026 года

## Назначение

Этот каталог хранит артефакты сравнительного benchmark-контура и
operational snapshot preview для ВКР.

Важно:

- в каталоге лежат несколько исторических волн прогонов;
- текущим каноническим поколением для ВКР считается только волна
  `baseline_comparison_2026-03-13_vkr30_cv10`;
- более ранние прогоны того же дня сохранены как исторические артефакты
  разработки и не должны считаться текущим источником истины.
- в текущей repo-policy historical comparison-waves считаются
  local-by-default и не обязаны входить в tracked current state;
- если конкретная historical wave нужна как versioned исследовательский
  артефакт, её нужно осознанно продвигать в git вручную.

## Канонические артефакты текущей волны

### Supervised benchmark

- `baseline_comparison_2026-03-13_vkr30_cv10.md`
- `baseline_comparison_2026-03-13_vkr30_cv10_summary.csv`
- `baseline_comparison_2026-03-13_vkr30_cv10_classwise.csv`
- `baseline_comparison_2026-03-13_vkr30_cv10_search_summary.csv`
- `baseline_comparison_2026-03-13_vkr30_cv10_*_train_scores.csv`
- `baseline_comparison_2026-03-13_vkr30_cv10_*_test_scores.csv`

### Operational snapshot preview

- `baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot.md`
- `baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot_summary.csv`
- `baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot_router.csv`
- `baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot_*_priority.csv`
- `baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot_*_top.csv`

## Как интерпретировать остальные файлы

Остальные файлы в каталоге относятся к предыдущим промежуточным волнам:

- первая baseline-волна без полного ВКР-контракта;
- отдельная `mlp`-волна;
- промежуточная snapshot-волна до фиксации канонического `vkr30_cv10`.

Эти файлы полезны как исследовательская история и не считаются мусором,
но при ссылках из README, docs, notebook и презентации нужно опираться
только на каноническое поколение `vkr30_cv10`.
