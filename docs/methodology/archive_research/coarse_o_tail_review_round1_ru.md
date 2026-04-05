# Coarse O-Tail Review Round 1

## Цель

Этот документ фиксирует первый предметный review редкого класса `O` в coarse
pipeline.

Разбор сделан на связке:

- coarse model artifact:
  `artifacts/models/gaia_id_coarse_classification__hist_gradient_boosting__2026_03_28_215003_509969`
- final decision run:
  `artifacts/decisions/hierarchical_final_decision_2026_03_29_111132_270743`
- review notebook:
  [11_coarse_o_tail_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/archive_research/11_coarse_o_tail_review.ipynb)

Задача round 1:

- отделить потери `O` на `quality_gate` от поведения coarse-model;
- показать, что происходит с surviving `O` rows;
- проверить, похожа ли физика surviving `O` на горячие `O/B` stars;
- зафиксировать evidence до любого retrain или rebalance.

## Сводка По Source

- `n_rows_o_source = 6372`
- `n_rows_o_pass = 1725`
- `share_pass = 27.07%`

Распределение `quality_state`:

- `unknown = 2423` (`38.03%`)
- `reject = 2224` (`34.90%`)
- `pass = 1725` (`27.07%`)

Top `quality_reason`:

- `missing_core_features = 2224`
- `missing_radius_flame = 2033`
- `pass = 1725`
- `high_ruwe = 284`
- `low_parallax_snr = 106`

Промежуточный вывод:

- `quality_gate` действительно режет значимую часть `O`;
- но даже после этого остается крупный pass-пул, достаточный для отдельного
  model-level review;
- значит проблема `O` не объясняется только gate-слоем.

## Что Делает Coarse-Model На Pass-Части

Считались только `quality_state = pass` строки.

Распределение `coarse_predicted_label`:

- `B = 1189` (`68.93%`)
- `F = 353` (`20.46%`)
- `G = 79` (`4.58%`)
- `A = 75` (`4.35%`)
- `K = 28` (`1.62%`)
- `M = 1` (`0.06%`)
- `O = 0`

Важное наблюдение:

- model не просто “иногда промахивается” по `O`;
- на surviving `O` rows она системно не выдает `O` вообще.

При этом уверенность очень высокая:

- `B`: `mean_confidence = 0.999744`
- `F`: `mean_confidence = 0.997912`
- `K`: `mean_confidence = 0.999853`

Значит это не пограничные uncertain-case ошибки, а уверенный systematic routing.

## Физический Профиль Predicted Groups

Median physics по predicted groups:

- `B`: `teff ≈ 15610 K`, `bp_rp ≈ 0.48`, `radius_flame ≈ 4.78`
- `F`: `teff ≈ 6597 K`, `bp_rp ≈ 0.76`, `radius_flame ≈ 3.31`
- `G`: `teff ≈ 5783 K`, `bp_rp ≈ 0.93`, `radius_flame ≈ 1.63`
- `A`: `teff ≈ 8201 K`, `bp_rp ≈ 0.88`, `radius_flame ≈ 5.58`
- `K`: `teff ≈ 4818 K`, `bp_rp ≈ 1.51`, `radius_flame ≈ 10.66`
- `M`: `teff ≈ 3602 K`, `bp_rp ≈ 4.31`, `radius_flame ≈ 17.19`

Вывод:

- часть surviving `O` действительно похожа хотя бы на hot `B/A` tail;
- но значимая часть predicted `F/G/K/M` физически уже не выглядит как hot stars;
- это сильный сигнал, что в source-пуле есть label/physics inconsistency, а не
  только model weakness.

## Downstream Fate В Final Decision

Распределение `final_domain_state` для true `O`:

- `unknown = 4647` (`72.93%`)
- `id = 1614` (`25.33%`)
- `ood = 111` (`1.74%`)

Распределение `final_coarse_class`:

- `<NA> = 4758` (`74.67%`)
- `B = 1101` (`17.28%`)
- `F = 350` (`5.49%`)
- `G = 79` (`1.24%`)
- `A = 55` (`0.86%`)
- `K = 28` (`0.44%`)
- `M = 1` (`0.02%`)

Распределение `final_decision_reason`:

- `quality_unknown = 2423`
- `quality_reject = 2224`
- `refinement_accepted = 1614`
- `hard_ood = 111`

Итог:

- `O` почти не пропадает как отдельная “невидимая” ошибка routing;
- большая часть true `O` уходит в `unknown` из-за gate;
- surviving pass-часть затем системно переводится в `B/F/...`, а не в `O`.

## Практическая Трактовка

Round 1 не поддерживает гипотезу “достаточно чуть подкрутить coarse model”.

Сейчас evidence сильнее поддерживает более осторожную трактовку:

- rare-tail `O` страдает одновременно от двух факторов;
- первый фактор: большая доля `O` теряется на availability/quality signals;
- второй фактор: surviving `O` pool уже частично физически не похож на hot `O`
  stars.

Поэтому немедленный rebalance или новый threshold для `O` пока не выглядит
научно чистым первым шагом.

## Решение После Round 1

Пока не открывать coarse retrain для `O`.

Сначала нужен отдельный narrow step:

1. выделить physically hot `O/B-like` subset внутри true `O` source;
2. отдельно проверить, сколько `O` rows конфликтуют с Gaia physics;
3. только после этого решать, нужен ли:
   - rebalance;
   - отдельная `O` policy;
   - или source-cleaning / label-consistency gate.

## Related

- [coarse_o_tail_review_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_tail_review_tz_ru.md)
- [stabilization_issue_ledger_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/stabilization_issue_ledger_ru.md)
- [star_level_review_round2_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/star_level_review_round2_ru.md)
