# Baseline Run Registry

## Цель

Этот документ фиксирует baseline artifacts, на которые опирается текущая
stabilization-фаза.

Правило:

- пока отдельный bugfix или calibration-step не доказал необходимость смены
  baseline, именно эти runs считаются reference state проекта.

## Stable Benchmark Runs

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

## Stable Model Artifacts

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

## Stable Threshold Artifacts

- `artifacts/thresholds/gaia_id_ood_classification__hist_gradient_boosting_calibrated_sigmoid__threshold__2026_03_28_215240_850166`

## Stable Final Decision Runs

### Historical First End-To-End Run

- `artifacts/decisions/hierarchical_final_decision_2026_03_28_220214_410649`

### Historical Stable Baseline

- `artifacts/decisions/hierarchical_final_decision_2026_03_29_075935_878508`

Это прежний стабильный baseline run, на который опирался основной слой
stabilization до пакета донастройки `quality_gate + priority`.

### Current Active Baseline

- `artifacts/decisions/hierarchical_final_decision_2026_04_05_123111_055017`

Это текущий основной baseline run, потому что он:

- прошел после отдельной донастройки `quality_gate` и `priority`;
- сохраняет осторожные пороги `RUWE = 1.4` и `parallax_over_error = 5.0`;
- снимает только требование `radius_flame` для `pass`;
- использует более читаемую label-policy `priority`:
  - `high_min = 0.85`
  - `medium_min = 0.55`
- уже используется как active baseline для технического review-слоя.

Связанные разборы:

- [pre_battle_policy_candidate_run_2026_04_05_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/pre_battle_policy_candidate_run_2026_04_05_ru.md)
- [high_priority_cohort_review_2026_04_05_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/high_priority_cohort_review_2026_04_05_ru.md)
- [regression_validation_run_2026_04_06_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/regression_validation_run_2026_04_06_ru.md)

### Current Threshold Candidate Run

- `artifacts/decisions/hierarchical_final_decision_2026_03_29_111132_270743`

Это отдельный live run для проверки более строгих `priority` thresholds:

- `priority_high_min = 0.85`
- `priority_medium_min = 0.55`

Этот run пока не переопределяет baseline автоматически и используется как
comparison/candidate run для stabilization-review.

### Current Diagnostic Tuning Baseline

- `artifacts/decisions/hierarchical_final_decision_2026_04_05_090717_885503`

Это не новый production baseline, а диагностический run для пакета
донастройки перед следующим боевым прогоном.

Он нужен как единая точка отсчета для:

- `quality_gate` tuning;
- `priority` tuning;
- сравнения candidate policy без смешения со старым stable baseline.

Ключевые live-показатели этого run:

- `n_rows_input = 402226`
- `id = 177674` (`44.17%`)
- `unknown = 223787` (`55.64%`)
- `ood = 765` (`0.19%`)
- `quality_reject = 159964`
- `quality_unknown = 63823`
- `priority high = 100173` (`56.38%` от priority-пула)

Связанный разбор:

- [pre_battle_diagnostic_run_2026_04_05_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/pre_battle_diagnostic_run_2026_04_05_ru.md)

## Active Review Notebooks

- pipeline review:
  - [model_pipeline_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/model_pipeline_review.ipynb)
- final decision review:
  - [final_decision_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/final_decision_review.ipynb)
- quality-gate calibration:
  - [quality_gate_calibration.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/research/quality_gate_calibration.ipynb)

## Правило Смены Baseline

Baseline можно менять только если новый run:

- воспроизводимо проходит;
- не ломает contracts;
- закрывает подтвержденный defect или calibration goal;
- зафиксирован в docs и notebook review.
