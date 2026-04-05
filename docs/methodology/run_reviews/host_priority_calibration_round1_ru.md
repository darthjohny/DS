# Host Priority Calibration Round 1

## Run

- model artifact:
  - `artifacts/models/host_field_classification__hist_gradient_boosting__2026_03_29_071601_090632`
- review notebook:
  - `analysis/notebooks/technical/host_priority_calibration_review.ipynb`

## Holdout Summary

- `n_rows_full = 7406`
- `n_rows_train = 5184`
- `n_rows_test = 2222`
- `train_positive_rate = 0.5`
- `test_positive_rate = 0.5`

## Core Metrics

- `brier_score = 0.035357`
- `log_loss = 0.134080`
- `roc_auc = 0.990521`
- `mean_predicted_probability = 0.503894`

## Primary Findings

1. Разделение `host` vs `field` сильное.
   - `roc_auc` высокий, значит сигнал `host_similarity_score` не выглядит случайным.

2. Распределение вероятностей почти бимодальное.
   - `(0.0, 0.1]`: `1032` строк (`46.44%`)
   - `(0.9, 1.0]`: `1046` строк (`47.07%`)
   - средние probability bins заполнены слабо

3. Верхний хвост выглядит слегка переуверенным, но не сломанным.
   - high bin:
     - `mean_probability = 0.991102`
     - `positive_rate = 0.969407`
   - low bin:
     - `mean_probability = 0.002514`
     - `positive_rate = 0.012597`

4. Проблема `priority` saturation пока выглядит не как bug в host-model.
   - host-model хорошо разделяет классы
   - saturation выше по pipeline, вероятно, усиливается через:
     - `class_priority_score`
     - границы `priority_label`
     - policy интеграции `host_similarity_score` в final ranking

## Group Context

По holdout `spec_class`:

- `G`: `1168` строк, `median_host_similarity_score = 0.758095`
- `K`: `636`, `0.220426`
- `F`: `282`, `0.285249`
- `M`: `136`, `0.489989`

По `evolution_stage`:

- `dwarf`: `2032`, `median_host_similarity_score = 0.686660`
- `evolved`: `190`, `0.417684`

## Decision After Round 1

- post-hoc calibration пока не включаем автоматически
- сначала сохраняем этот review как baseline
- следующий шаг:
  - проверить, достаточно ли скорректировать `priority` thresholds / scaling
  - и только после этого решать, нужен ли отдельный calibrated host layer

## Related

- [host_priority_calibration_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/host_priority_calibration_tz_ru.md)
- [priority_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/priority_review_round1_ru.md)
- [stabilization_issue_ledger_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/stabilization_issue_ledger_ru.md)
