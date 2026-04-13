# Диагностический прогон перед донастройкой

Дата фиксации: `2026-04-05`

Прогон:

- [hierarchical_final_decision_2026_04_05_090717_885503](/Users/evgeniikuznetsov/Desktop/dspro-vkr/artifacts/decisions/hierarchical_final_decision_2026_04_05_090717_885503)

## Зачем нужен этот прогон

Этот прогон не считается новым рабочим базовым прогоном.

Он нужен как диагностическая точка отсчета перед пакетом донастройки:

- `quality_gate`;
- `priority`.

## Конфигурация

Источник данных:

- `lab.gaia_mk_quality_gated`

Артефакты:

- модель `id_ood`:
  - `artifacts/models/gaia_id_ood_classification__hist_gradient_boosting_calibrated_sigmoid__2026_03_28_215240_816364`
- порог `id_ood`:
  - `artifacts/thresholds/gaia_id_ood_classification__hist_gradient_boosting_calibrated_sigmoid__threshold__2026_03_28_215240_850166`
- модель `coarse`:
  - `artifacts/models/gaia_id_coarse_classification__hist_gradient_boosting__2026_03_28_215003_509969`
- семейство моделей `refinement`:
  - `a`
  - `b`
  - `f`
  - `g`
  - `k`
  - `m`
- модель `host`:
  - `artifacts/models/host_field_classification__hist_gradient_boosting__2026_03_29_071601_090632`

Политика:

- `priority_high_min = 0.75`
- `priority_medium_min = 0.45`
- остальные пороги взяты из текущей боевой конфигурации кода без ручного переопределения

## Итог по контуру

- `n_rows_input = 402226`
- `n_rows_final_decision = 402226`
- `n_rows_priority_input = 177674`
- `n_rows_priority_ranking = 177674`
- `n_unique_source_id = 402226`

## Финальная маршрутизация

- `id = 177674` (`44.17%`)
- `unknown = 223787` (`55.64%`)
- `ood = 765` (`0.19%`)

## Состояние `quality`

- `pass = 178439` (`44.36%`)
- `reject = 159964` (`39.77%`)
- `unknown = 63823` (`15.87%`)

## Причины Финального Решения

- `refinement_accepted = 177674`
- `quality_reject = 159964`
- `quality_unknown = 63823`
- `hard_ood = 765`

## Группы разбора

Основные группы разбора:

- `pass = 166847`
- `reject_missing_core_features = 159873`
- `review_high_ruwe = 28388`
- `review_missing_radius_flame = 16762`
- `review_non_single_star = 14402`
- `review_low_single_star_probability = 10474`
- `review_low_parallax_snr = 5292`

## Priority

Распределение:

- `high = 100173` (`56.38%`)
- `medium = 18373` (`10.34%`)
- `low = 59128` (`33.28%`)

Квантильный профиль:

- `priority_score p50 = 0.806156`
- `host_similarity_score p50 = 0.956772`
- `observability_score p50 = 0.614001`

Классовый профиль:

- `K`: `mean_priority_score = 0.797013`
- `G`: `0.770004`
- `F`: `0.760516`
- `M`: `0.474911`
- `A`: `0.297214`
- `B`: `0.277904`

## Метрики Моделей

- `id_ood`:
  - `accuracy = 0.995734`
  - `balanced_accuracy = 0.926215`
  - `macro_f1 = 0.944521`
  - `roc_auc_ovr = 0.996188`
- `coarse`:
  - `accuracy = 0.992926`
  - `balanced_accuracy = 0.992379`
  - `macro_f1 = 0.992573`
  - `roc_auc_ovr = 0.999977`
- `refinement_flat`:
  - `accuracy = 0.320336`
  - `balanced_accuracy = 0.187861`
  - `macro_f1 = 0.189683`
  - `roc_auc_ovr = 0.713111`
- `host_field`:
  - `accuracy = 0.955446`
  - `balanced_accuracy = 0.955446`
  - `macro_f1 = 0.955442`
  - `roc_auc = 0.990521`
  - `brier_score = 0.035357`
  - `log_loss = 0.134080`

## Текущая Трактовка

Этот прогон подтверждает:

- главное практическое ограничение сейчас связано не с падением базовых моделей;
- основное давление на итоговую маршрутизацию дает `quality_gate`;
- `priority` все еще заметно насыщен в верхней зоне при дефолтных порогах;
- поэтому следующий шаг должен идти не в интерпретацию объектов, а в
  спокойную донастройку политики двух слоев:
  - `quality_gate`
  - `priority`

## Следующий шаг

- переход к сравнению и донастройке `quality_gate` и `priority`.
