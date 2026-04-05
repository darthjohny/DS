# Priority Threshold Review Round 2

## Run

- baseline decision run:
  - `artifacts/decisions/hierarchical_final_decision_2026_03_29_075935_878508`
- threshold candidate run:
  - `artifacts/decisions/hierarchical_final_decision_2026_03_29_111132_270743`

## Applied Thresholds

- `high_min = 0.85`
- `medium_min = 0.55`

Это тот же вариант, который в round 1 был проверен как
`strict_high_medium_085_055`, но теперь уже на реальном live `decide` run.

## Live Comparison

`n_rows = 177674`

### Baseline

- `high = 100173` (`56.38%`)
- `medium = 18373` (`10.34%`)
- `low = 59128` (`33.28%`)

### Threshold Candidate

- `high = 72048` (`40.55%`)
- `medium = 41851` (`23.55%`)
- `low = 63775` (`35.89%`)

## Observed Transitions

- changed rows:
  - `32772` (`18.45%`)

Transitions:

- `high -> medium = 28125`
- `medium -> low = 4647`

Важно:

- `final_domain_state` не изменился;
- `id = 177674`
- `ood = 765`
- `unknown = 223787`

То есть этот run меняет только downstream priority stratification и не ломает
верхние decision contracts.

## Class-Level Distribution On New Run

- `A`
  - `high = 0.00%`
  - `medium = 0.00%`
  - `low = 100.00%`
- `B`
  - `high = 0.00%`
  - `medium = 0.00%`
  - `low = 100.00%`
- `F`
  - `high = 49.56%`
  - `medium = 33.01%`
  - `low = 17.43%`
- `G`
  - `high = 46.86%`
  - `medium = 38.89%`
  - `low = 14.25%`
- `K`
  - `high = 69.51%`
  - `medium = 19.36%`
  - `low = 11.13%`
- `M`
  - `high = 1.64%`
  - `medium = 33.59%`
  - `low = 64.77%`

## Primary Findings

1. Live run полностью подтвердил offline threshold review.
   - фактические распределения совпали с round 1 review
   - hidden coupling в `decide` не обнаружено

2. Более строгие thresholds сужают `high` zone достаточно заметно.
   - `high` падает с `56.38%` до `40.55%`
   - `medium` становится содержательным operational bucket

3. Threshold-only change не ломает upstream routing.
   - `id / ood / unknown` не меняются
   - меняется только priority-layer

4. `K` остается самым насыщенным high-priority классом.
   - это уже не threshold bug
   - если это будет проблемой после star-level review, следующим шагом нужен
     не новый threshold tweak, а отдельный scaling review

## Decision After Round 2

- threshold candidate run принимается как текущий active review run;
- `final_decision_review.ipynb` должен смотреть именно на него;
- отдельный `priority scaling` пакет пока не открываем;
- сначала делаем star-level review на новом run и проверяем,
  не остается ли top-zone слишком плоской operationally.

## Related

- [priority_threshold_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/priority_threshold_review_round1_ru.md)
- [priority_threshold_calibration_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/priority_threshold_calibration_tz_ru.md)
- [host_priority_calibration_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/host_priority_calibration_round1_ru.md)
