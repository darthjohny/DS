# Coarse O/B Boundary Review Round 1

## Цель

Этот документ фиксирует первый narrow review границы `O/B` после hot-subset
анализа.

Разбор сделан на связке:

- coarse model artifact:
  `artifacts/models/gaia_id_coarse_classification__hist_gradient_boosting__2026_03_28_215003_509969`
- baseline boundary slice:
  - `spectral_class IN ('O', 'B')`
  - `quality_state = 'pass'`
  - `teff_gspphot >= 10000 K`
- review notebook:
  [13_coarse_ob_boundary_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/archive_research/13_coarse_ob_boundary_review.ipynb)

## Сводка По Boundary Source

- `n_rows_boundary_source = 8300`
- `n_rows_scored = 8300`

Распределение true-классов:

- `B = 7112` (`85.69%`)
- `O = 1188` (`14.31%`)

То есть boundary review опирается не на микроскопический пул, а на вполне
заметный hot pass-slice.

## Что Делает Coarse-Model На Boundary

Распределение `coarse_predicted_label`:

- `B = 8279` (`99.75%`)
- `A = 21` (`0.25%`)
- `O = 0`

Confusion-like breakdown:

- true `B -> B = 7091` (`99.70%` внутри true `B`)
- true `B -> A = 21` (`0.30%`)
- true `O -> B = 1188` (`100%` внутри true `O`)
- true `O -> O = 0`

Это уже очень сильный вывод:

- asymmetry односторонняя;
- `B -> O` практически отсутствует;
- `O -> B` происходит полностью.

## Что Говорят Вероятности

Median probability summary:

- true `B`:
  - `median P(O) ≈ 0.000017`
  - `median P(B) ≈ 0.999842`
- true `O`:
  - `median P(O) ≈ 0.000016`
  - `median P(B) ≈ 0.999846`

Mean probability summary:

- true `B`:
  - `mean P(O) ≈ 0.000031`
  - `mean P(B) ≈ 0.997244`
- true `O`:
  - `mean P(O) ≈ 0.000084`
  - `mean P(B) ≈ 0.999749`

Практический смысл:

- модель даже не считает true `O` чем-то близким к `O`;
- она уверенно видит и true `O`, и true `B` как один и тот же `B`-кластер.

Это не похоже на пограничные uncertain mistakes.

## Что Видно По Физике

Median physics:

- true `B`:
  - `teff ≈ 12312 K`
  - `logg ≈ 3.79`
  - `bp_rp ≈ 0.20`
  - `radius_flame ≈ 3.53`
- true `O`:
  - `teff ≈ 15614 K`
  - `logg ≈ 3.67`
  - `bp_rp ≈ 0.48`
  - `radius_flame ≈ 4.78`

Вывод:

- hot true `O` в среднем горячее и крупнее true `B`;
- но overlap по physics все равно заметный;
- coarse-модель, похоже, не выучила отдельную boundary для hottest tail и
  схлопывает его в `B`.

## Preview Односторонних Ошибок

High-confidence `O -> B` preview показывает:

- `teff` примерно от `11149 K` до `17983 K`
- `P(B)` стабильно около `0.99985`
- `P(O)` стабильно около `0.000016`

High-confidence `B -> O` preview пуст.

Это окончательно подтверждает:

- проблема не симметричная;
- `O` не просто путается с `B`;
- модель фактически не использует `O` как рабочий coarse outcome на этом slice.

## Практическая Трактовка

После round 1 самый сильный вывод такой:

- broad contamination уже убрана на предыдущем шаге;
- текущая проблема — это узкая и очень жесткая boundary failure `O -> B`;
- blind rebalance всей coarse-модели по-прежнему не лучший первый шаг.

Сейчас scientific/logical next step выглядит так:

1. отдельно проверить train-time representation hottest `O` tail;
2. посмотреть, достаточно ли у `O` support в train/test split после текущей
   source-политики;
3. и только потом решать:
   - narrow retrain;
   - class-weighting для `O`;
   - или targeted source-cleaning/relabelling hottest tail.

## Related

- [coarse_ob_boundary_review_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_ob_boundary_review_tz_ru.md)
- [coarse_o_hot_subset_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_hot_subset_review_round1_ru.md)
- [stabilization_issue_ledger_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/stabilization_issue_ledger_ru.md)
