# Priority Threshold Review Round 1

## Run

- decision run:
  - `artifacts/decisions/hierarchical_final_decision_2026_03_29_075935_878508`
- review notebook:
  - `analysis/notebooks/technical/priority_threshold_review.ipynb`

## Reviewed Variants

- baseline:
  - `high_min = 0.75`
  - `medium_min = 0.45`
- `strict_high_080`:
  - `high_min = 0.80`
  - `medium_min = 0.45`
- `strict_high_medium_085_055`:
  - `high_min = 0.85`
  - `medium_min = 0.55`

## Live Summary

`n_rows = 177674`

### Baseline

- `high = 100173` (`56.38%`)
- `medium = 18373` (`10.34%`)
- `low = 59128` (`33.28%`)

### `strict_high_080`

- `high = 90293` (`50.82%`)
- `medium = 28253` (`15.90%`)
- `low = 59128` (`33.28%`)
- changed from baseline:
  - `9880` (`5.56%`)

Transition:

- only one meaningful move:
  - `high -> medium`

### `strict_high_medium_085_055`

- `high = 72048` (`40.55%`)
- `medium = 41851` (`23.55%`)
- `low = 63775` (`35.89%`)
- changed from baseline:
  - `32772` (`18.45%`)

Transitions:

- `high -> medium = 28125`
- `medium -> low = 4647`

## Class-Level Impact For `strict_high_medium_085_055`

- `K`
  - `high = 69.51%`
  - `medium = 19.36%`
  - `low = 11.13%`
- `F`
  - `high = 49.56%`
  - `medium = 33.01%`
  - `low = 17.43%`
- `G`
  - `high = 46.86%`
  - `medium = 38.89%`
  - `low = 14.25%`
- `M`
  - `high = 1.64%`
  - `medium = 33.59%`
  - `low = 64.77%`
- `A/B`
  - остаются целиком в `low`

## Primary Findings

1. Threshold tightening уже дает заметный эффект без изменения ranking formula.
   - simple cutoff review действительно сужает `high` zone
   - значит saturation не обязательно требует немедленного scaling layer

2. `strict_high_080` слишком мягкий.
   - меняет только `5.56%` строк
   - operationally полезен, но не решает ширину high-zone достаточно сильно

3. `strict_high_medium_085_055` выглядит как рабочий кандидат для следующего live run.
   - high-zone сжимается с `56.38%` до `40.55%`
   - medium становится информативнее
   - low почти не ломается

4. `K` все еще слишком насыщен наверху.
   - даже на stricter thresholds у `K` high share остается `69.51%`
   - это уже аргумент в пользу следующего scaling review,
     если после нового run top-zone останется слишком плоской

## Decision After Round 1

- ranking formula пока не меняем
- post-hoc calibration для host-model пока не включаем
- следующий разумный шаг:
  - сделать one-step live run с stricter thresholds
  - и только потом решать, нужен ли отдельный scaling package

## Related

- [priority_threshold_calibration_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/priority_threshold_calibration_tz_ru.md)
- [priority_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/priority_review_round1_ru.md)
- [host_priority_calibration_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/host_priority_calibration_round1_ru.md)
