# Первый обзор калибровки host-приоритета

## Прогон

- артефакт модели:
  - `artifacts/models/host_field_classification__hist_gradient_boosting__2026_03_29_071601_090632`
- обзорный ноутбук:
  - `analysis/notebooks/technical/host_priority_calibration_review.ipynb`

## Сводка по holdout-выборке

- `n_rows_full = 7406`
- `n_rows_train = 5184`
- `n_rows_test = 2222`
- `train_positive_rate = 0.5`
- `test_positive_rate = 0.5`

## Основные метрики

- `brier_score = 0.035357`
- `log_loss = 0.134080`
- `roc_auc = 0.990521`
- `mean_predicted_probability = 0.503894`

## Основные выводы

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

4. Проблема насыщения `priority` пока не выглядит как ошибка в host-модели.
   - host-модель хорошо разделяет классы
   - насыщение выше по контуру, вероятно, усиливается через:
     - `class_priority_score`
     - границы `priority_label`
     - политику интеграции `host_similarity_score` в итоговое ранжирование

## Контекст по группам

По holdout `spec_class`:

- `G`: `1168` строк, `median_host_similarity_score = 0.758095`
- `K`: `636`, `0.220426`
- `F`: `282`, `0.285249`
- `M`: `136`, `0.489989`

По `evolution_stage`:

- `dwarf`: `2032`, `median_host_similarity_score = 0.686660`
- `evolved`: `190`, `0.417684`

## Решение после первого обзора

- дополнительную калибровку пока не включаем автоматически
- сначала сохраняем этот обзор как базовый
- следующий шаг:
  - проверить, достаточно ли скорректировать пороги и масштабирование `priority`
  - и только после этого решать, нужен ли отдельный откалиброванный слой host

## Связанные документы

- [priority_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/priority_review_round1_ru.md)
- [stabilization_issue_ledger_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/stabilization_issue_ledger_ru.md)
