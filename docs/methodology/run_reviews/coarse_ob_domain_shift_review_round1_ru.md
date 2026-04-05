# Round 1: Domain Shift Review На Границе `O/B`

## Контекст

После предыдущих deep-dive review стало ясно:

- current coarse-artifact на собственном train-time `O/B` boundary работает почти идеально;
- starvation по `O` нет;
- downstream hot pass-source при этом почти полностью схлопывается в `B`.

Здесь проверяется следующая гипотеза:

- проблема `O -> B` вызвана не train-time incapacity модели, а domain shift / source mismatch
  между train-time coarse source и downstream hot pass-source.

Опорное ТЗ:

- [coarse_ob_domain_shift_review_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/coarse_ob_domain_shift_review_tz_ru.md)

## Live Source

- coarse model artifact:
  - `/Users/evgeniikuznetsov/Desktop/dspro-vkr/artifacts/models/gaia_id_coarse_classification__hist_gradient_boosting__2026_03_28_215003_509969`
- train-time domain:
  - `GAIA_ID_COARSE_CLASSIFICATION_TASK`
- downstream domain:
  - `lab.gaia_mk_quality_gated`
  - `quality_state = 'pass'`
  - `spectral_class IN ('O', 'B')`
  - `teff_gspphot >= 10000`

## Объемы

- `train_time`: `6000`
- `downstream_pass`: `8300`

Баланс true классов:

- `train_time`:
  - `B = 3000`
  - `O = 3000`
- `downstream_pass`:
  - `B = 7112`
  - `O = 1188`

## Поведение Current Coarse Artifact

На `train_time`:

- `true B -> B = 2995`
- `true B -> O = 1`
- `true B -> A = 4`
- `true O -> O = 3000`

На `downstream_pass`:

- `true B -> B = 7091`
- `true B -> A = 21`
- `true O -> B = 1188`
- `true O -> O = 0`

Вывод:

- current coarse-artifact не теряет `O` на собственном train-time hot boundary;
- downstream true `O` полностью попадает в `B`.

## Вероятности

Median probabilities:

- `train_time`, true `O`:
  - `median P(O) = 0.999845`
  - `median P(B) = 0.000018`
- `downstream_pass`, true `O`:
  - `median P(O) = 0.000016`
  - `median P(B) = 0.999846`

Это очень сильный сигнал:

- downstream true `O` для модели выглядят почти так же, как downstream true `B`;
- это уже не “пограничная неуверенность”, а почти полная смена области признаков.

## Физические Сдвиги

Для true `O`:

- `median teff_gspphot`:
  - `train_time = 34604.617`
  - `downstream_pass = 15613.8865`
- `median radius_feature`:
  - `train_time = 8.79655`
  - `downstream_pass = 4.778071`
- `median bp_rp`:
  - `train_time = 3.385037`
  - `downstream_pass = 0.477677`
- `median parallax_over_error`:
  - `train_time = 4.884386`
  - `downstream_pass = 21.700754`

Для true `B` тоже есть сдвиги, но заметно слабее по `teff_gspphot`.

## Missingness

По ключевым coarse features различия по пропускам не выглядят объяснением проблемы:

- на sampled review-слое `missing_share = 0.0` по основным feature columns
  в обоих доменах.

Следовательно:

- `O -> B` не объясняется тем, что downstream hot pass-source просто “дырявый”.

## Separability Train vs Downstream

Самые сильные domain-shift признаки для true `O`:

- `teff_gspphot`:
  - `separability_auc = 1.000000`
- `bp_rp`:
  - `separability_auc = 0.995113`
- `radius_feature`:
  - `separability_auc = 0.981560`
- `parallax_over_error`:
  - `separability_auc = 0.938852`
- `parallax`:
  - `separability_auc = 0.888868`

Для true `B` тоже выражены:

- `parallax_over_error = 0.899968`
- `bp_rp = 0.892009`
- `parallax = 0.869086`

Вывод:

- domain shift подтвержден;
- для true `O` он очень сильный именно по физике, а не по missingness.

## Текущая Интерпретация

С учетом предыдущих review это означает:

- starvation по `O` нет;
- train-time `O/B` separability у current coarse-artifact есть;
- downstream `O -> B` вызван уже не нехваткой support, а source/domain mismatch.

То есть сейчас преждевременный narrow retrain выглядит плохо обоснованным.

## Следующий Шаг

Следующий правильный пакет:

- не ретрейнить coarse вслепую;
- отдельно разбирать provenance и label semantics downstream true `O` pass-pool;
- проверить, насколько downstream `O` согласуются с физикой hot stars и с происхождением
  меток в этом контуре;
- только после этого решать, нужен ли:
  - source alignment,
  - relabeling,
  - или уже потом узкий retrain/class weighting.
