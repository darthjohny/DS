# Coarse O Hot-Subset Review Round 1

## Цель

Этот документ фиксирует первый narrow review physically hot `O/B-like` subset
внутри true `O` source.

Разбор сделан на связке:

- coarse model artifact:
  `artifacts/models/gaia_id_coarse_classification__hist_gradient_boosting__2026_03_28_215003_509969`
- final decision run:
  `artifacts/decisions/hierarchical_final_decision_2026_03_29_111132_270743`
- baseline subset-config:
  - `teff_gspphot >= 10000 K`
- review notebook:
  [12_coarse_o_hot_subset_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/archive_research/12_coarse_o_hot_subset_review.ipynb)

## Зачем Был Нужен Этот Шаг

Round 1 по всему `O` source уже показал:

- в true `O` source есть заметная cool contamination;
- часть surviving `O` rows ведет себя как `F/G/K/M-like` physics;
- поэтому общий `O` пул слишком шумный для немедленного retrain.

Hot-subset review должен был ответить:

- остается ли broad contamination после простого physics cut;
- или проблема сужается до более чистой границы `O -> B`.

## Сводка По Hot-Subset

При baseline `teff_gspphot >= 10000 K`:

- `n_rows_true_o_source = 6372`
- `n_rows_hot_subset = 3149`
- `share_hot_subset_in_source = 49.42%`
- `n_rows_hot_pass = 1188`
- `share_hot_pass_in_hot_subset = 37.73%`

Вывод:

- почти половина true `O` source проходит минимум на hot `O/B-like` physics;
- значит narrow review на hot-subset статистически осмыслен и не упирается в
  слишком маленький sample.

## Что Делает Quality Gate На Hot-Subset

Распределение `quality_state`:

- `unknown = 1961` (`62.27%`)
- `pass = 1188` (`37.73%`)
- `reject = 0`

Top `quality_reason`:

- `missing_radius_flame = 1754`
- `pass = 1188`
- `high_ruwe = 153`
- `low_parallax_snr = 54`

Практический вывод:

- после hot-cut главный blocker уже не `missing_core_features`, а
  `missing_radius_flame`;
- значит у physically hot subset gate-loss стал чище и лучше интерпретируется.

## Что Делает Coarse-Model На Hot Pass-Subset

Распределение `coarse_predicted_label`:

- `B = 1188` (`100%`)
- `O = 0`
- `A/F/G/K/M = 0`

Это самый важный результат round 1:

- после удаления cool contamination coarse-model больше не разбрасывает `O`
  в `F/G/K/M`;
- broad contamination исчезает;
- проблема сужается до устойчивого routing `O -> B`.

## Физический Профиль Predicted Hot-Subset

Для predicted `B`:

- `median_teff_gspphot ≈ 15614 K`
- `median_logg_gspphot ≈ 3.67`
- `median_bp_rp ≈ 0.48`
- `median_radius_flame ≈ 4.78`

Интерпретация:

- hot-subset уже выглядит физически согласованнее, чем полный `O` source;
- то есть этот subset намного лучше подходит для следующего narrow review, чем
  весь исходный `O` pool.

## Downstream Fate В Final Decision

Распределение `final_domain_state`:

- `unknown = 1961` (`62.27%`)
- `id = 1100` (`34.93%`)
- `ood = 88` (`2.79%`)

Распределение `final_coarse_class`:

- `<NA> = 2049` (`65.07%`)
- `B = 1100` (`34.93%`)
- `O = 0`

Распределение `final_decision_reason`:

- `quality_unknown = 1961`
- `refinement_accepted = 1100`
- `hard_ood = 88`

Вывод:

- если hot-subset проходит в `id`, он почти всегда попадает туда как `B`, а не
  как `O`;
- значит проблема теперь уже не broad failure всего coarse-layer, а узкая
  boundary issue между `O` и `B`.

## Что Это Меняет По Сравнению С Full O Review

На полном `O` source мы видели:

- большой увод в `F/G/K/M`
- evidence of cool contamination

На hot-subset мы видим:

- `F/G/K/M` исчезают полностью;
- high-confidence `non-O/B` preview становится пустым;
- остается чистая systematic граница `O -> B`.

Это очень полезное сужение гипотезы.

## Решение После Round 1

Сейчас не нужен blind rebalance по всему `O` source.

Следующий правильный шаг:

1. открыть отдельный narrow review для boundary `O vs B`;
2. проверить, насколько hot true `O` физически и label-wise пересекается с
   train-time `B`;
3. только после этого решать:
   - нужен ли rebalance;
   - нужен ли class-weight policy;
   - нужен ли source-cleaning для hottest tail.

## Related

- [coarse_o_hot_subset_review_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_hot_subset_review_tz_ru.md)
- [coarse_o_tail_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_tail_review_round1_ru.md)
- [stabilization_issue_ledger_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/stabilization_issue_ledger_ru.md)
