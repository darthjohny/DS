# Реестр базовых прогонов

## Цель

Этот документ фиксирует базовые артефакты, на которые опирается текущая
фаза стабилизации.

Правило:

- пока отдельное исправление или шаг калибровки не доказал необходимость смены
  базового прогона, именно эти прогоны считаются опорным состоянием проекта.

## Стабильные benchmark-прогоны

### Hierarchical Core

- `id_ood`:
  - `artifacts/benchmarks/gaia_id_ood_classification_2026_03_28_172217_384006`
- `coarse`:
  - `artifacts/benchmarks/gaia_id_coarse_classification_2026_03_28_171258_103400`
- `refinement_flat`:
  - `artifacts/benchmarks/gaia_mk_refinement_classification_2026_03_28_172145_781713`

### Host

- `host_field`:
  - `artifacts/benchmarks/host_field_classification_2026_03_29_071615_364867`

## Стабильные модельные артефакты

### OOD / Coarse

- `artifacts/models/gaia_id_ood_classification__hist_gradient_boosting_calibrated_sigmoid__2026_03_28_215240_816364`
- `artifacts/models/gaia_id_coarse_classification__hist_gradient_boosting__2026_03_28_215003_509969`

### Refinement Families

- `artifacts/models/gaia_mk_refinement_a_classification__hist_gradient_boosting__2026_03_28_215244_571356`
- `artifacts/models/gaia_mk_refinement_b_classification__hist_gradient_boosting__2026_03_28_215256_137842`
- `artifacts/models/gaia_mk_refinement_f_classification__hist_gradient_boosting__2026_03_28_215300_779017`
- `artifacts/models/gaia_mk_refinement_g_classification__hist_gradient_boosting__2026_03_28_215304_080764`
- `artifacts/models/gaia_mk_refinement_k_classification__hist_gradient_boosting__2026_03_28_215308_148605`
- `artifacts/models/gaia_mk_refinement_m_classification__hist_gradient_boosting__2026_03_28_215316_508533`

### Host

- `artifacts/models/host_field_classification__hist_gradient_boosting__2026_03_29_071601_090632`

## Стабильные артефакты порогов

- `artifacts/thresholds/gaia_id_ood_classification__hist_gradient_boosting_calibrated_sigmoid__threshold__2026_03_28_215240_850166`

## Стабильные прогоны итогового решения

### Первый сквозной исторический прогон

- `artifacts/decisions/hierarchical_final_decision_2026_03_28_220214_410649`

### Исторический стабильный базовый прогон

- `artifacts/decisions/hierarchical_final_decision_2026_03_29_075935_878508`

Это прежний стабильный базовый прогон, на который опирался основной слой
стабилизации до пакета донастройки `quality_gate + priority`.

### Текущий основной базовый прогон

- `artifacts/decisions/hierarchical_final_decision_2026_04_05_123111_055017`

Это текущий основной базовый прогон, потому что он:

- прошел после отдельной донастройки `quality_gate` и `priority`;
- сохраняет осторожные пороги `RUWE = 1.4` и `parallax_over_error = 5.0`;
- снимает только требование `radius_flame` для `pass`;
- использует более читаемую label-policy `priority`:
  - `high_min = 0.85`
  - `medium_min = 0.55`
- уже используется как основной базовый прогон для технического обзорного слоя.

Связанные разборы:

- [pre_battle_policy_candidate_run_2026_04_05_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/pre_battle_policy_candidate_run_2026_04_05_ru.md)
- [high_priority_cohort_review_2026_04_05_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/high_priority_cohort_review_2026_04_05_ru.md)
- [regression_validation_run_2026_04_06_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/regression_validation_run_2026_04_06_ru.md)

### Кандидатный прогон для новых порогов

- `artifacts/decisions/hierarchical_final_decision_2026_03_29_111132_270743`

Это отдельный рабочий прогон для проверки более строгих порогов `priority`:

- `priority_high_min = 0.85`
- `priority_medium_min = 0.55`

Этот прогон пока не переопределяет базовый автоматически и используется как
сравнительный кандидат в рамках стабилизации.

### Диагностический базовый прогон для донастройки

- `artifacts/decisions/hierarchical_final_decision_2026_04_05_090717_885503`

Это не новый основной рабочий прогон, а диагностический запуск для пакета
донастройки перед следующим боевым прогоном.

Он нужен как единая точка отсчета для:

- донастройки `quality_gate`;
- донастройки `priority`;
- сравнения вариантов политики без смешения со старым стабильным базовым
  прогоном.

Ключевые показатели этого прогона:

- `n_rows_input = 402226`
- `id = 177674` (`44.17%`)
- `unknown = 223787` (`55.64%`)
- `ood = 765` (`0.19%`)
- `quality_reject = 159964`
- `quality_unknown = 63823`
- `priority high = 100173` (`56.38%` от пула приоритета)

Связанный разбор:

- [pre_battle_diagnostic_run_2026_04_05_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/pre_battle_diagnostic_run_2026_04_05_ru.md)

## Основные обзорные ноутбуки

- обзор контура обработки:
  - [model_pipeline_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/model_pipeline_review.ipynb)
- обзор итогового решения:
  - [final_decision_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/final_decision_review.ipynb)
- разбор калибровки `quality_gate`:
  - [quality_gate_calibration.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/research/quality_gate_calibration.ipynb)

## Правило смены базового прогона

Базовый прогон можно менять только если новый прогон:

- воспроизводимо проходит;
- не ломает контракты;
- закрывает подтвержденную ошибку или цель калибровки;
- зафиксирован в документации и обзорных ноутбуках.
