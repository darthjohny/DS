# Priority Review Round 1

## Цель

Этот документ фиксирует первый предметный review нового `priority`-слоя на
живом `final decision` run.

Задача review:

- проверить, является ли насыщение `priority_score` около `1.0` bug-симптомом;
- отделить арифметическую ошибку от следствия текущего ranking-контракта;
- зафиксировать реальные распределения `priority_score`,
  `host_similarity_score`, `observability_score` и `class_priority_score`.

## Базовый Run

- decision run:
  - `artifacts/decisions/hierarchical_final_decision_2026_03_29_075935_878508`

## Official Опора

- [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [scikit-learn: HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [PSCompPars semantics](https://exoplanetarchive.ipac.caltech.edu/docs/pscp_about.html)
- [Gaia DR3 astrophysical_parameters semantics](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_astrophysical_parameter_tables/ssec_dm_astrophysical_parameters.html)

Важно:

- сам `priority_score` не является official NASA или Gaia score;
- это проектный explainable ranking-layer;
- поэтому official docs здесь нужны для semantics входных сигналов, а не для
  подтверждения самой формулы ranking.

## Текущий Контракт Ranking

На момент review `priority_score` считается как:

- `0.20 * class_priority_score`
- `0.45 * host_similarity_score`
- `0.35 * observability_score`

Дополнительные project rules:

- для `A/B` действует low-priority ветка с `class_priority_score = 0.20`;
- для target-классов `F/G/K/M` и допустимых `evolution_stage`
  `class_priority_score` близок к `1.0`;
- итоговый `priority_label`:
  - `high` при `priority_score >= 0.75`
  - `medium` при `priority_score >= 0.45`
  - `low` иначе

## Live Summary

- `n_rows_priority_ranking = 177674`

Распределение `priority_label`:

- `high`: `100173` (`56.38%`)
- `medium`: `18373` (`10.34%`)
- `low`: `59128` (`33.28%`)

Квантили:

- `priority_score`
  - `p50 = 0.806156`
  - `p75 = 0.876098`
  - `p95 = 0.906085`
  - `p99 = 0.940019`
- `host_similarity_score`
  - `p50 = 0.956772`
  - `p75 = 0.997217`
  - `p95 = 0.999736`
  - `p99 = 0.999907`
- `observability_score`
  - `p50 = 0.614001`
  - `p75 = 0.662311`
  - `p95 = 0.746380`
  - `p99 = 0.831550`

Биннинг `host_similarity_score`:

- `> 0.8`: `114695` строк
- `0.6 .. 0.8`: `10646`
- `< 0.4`: `45490`

Биннинг `observability_score`:

- `0.6 .. 0.8`: `92949`
- `0.4 .. 0.6`: `53525`
- `> 0.8`: `3076`

## Review По Классам

Средний `priority_score` по `final_coarse_class`:

- `K`: `0.797013`
- `G`: `0.770004`
- `F`: `0.760516`
- `M`: `0.474911`
- `A`: `0.297214`
- `B`: `0.277904`

Вывод:

- saturation сконцентрирован прежде всего в `F/G/K`;
- `A/B` уже штатно удерживаются в low-priority ветке;
- `M` ведет себя промежуточно и не насыщается так же агрессивно, как `F/G/K`.

## Интерпретация

### Что НЕ похоже на bug

- top `priority_score ~ 1.0` воспроизводится строго из формулы и входных
  компонент;
- top-объекты одновременно имеют:
  - `class_priority_score = 1.0`
  - `host_similarity_score ~= 1.0`
  - `observability_score = 1.0`
- значит score поднимается к `1.0` арифметически корректно.

Итог:

- текущая saturation не выглядит как code defect;
- это следствие текущего ranking-контракта и формы входных распределений.

### Что выглядит как design-risk

- `host_similarity_score` слишком концентрирован в верхней части шкалы внутри
  уже принятого `id`-пула;
- при весе `0.45` он начинает доминировать в итоговом ranking;
- `high`-зона получается слишком широкой и top-кандидаты становятся слабо
  различимыми между собой.

### Что это значит practically

- текущий `priority` хорош как coarse sorting;
- текущий `priority` пока слаб как fine-grained ordering внутри top-кандидатов;
- проблема больше похожа на calibration / scaling issue host-layer, чем на
  failure observability или class-prior блока.

## Решение Round 1

На этой итерации:

- ranking-код не переписываем;
- saturation фиксируем как `scientific_review / design-risk`, а не как defect;
- в `final_decision_review.ipynb` добавляем явные priority review tables:
  - `priority_label`
  - `priority_reason`
  - component quantiles
  - mean priority by coarse class

## Следующие Шаги

1. Провести отдельный review `host_similarity_score` как вероятностного сигнала.
2. Проверить, нужен ли post-hoc calibration именно для host-model.
3. Если saturation подтвердится как operational risk, рассмотреть:
   - более строгий `high` threshold;
   - нелинейное преобразование host-score;
   - percentile-based ranking внутри accepted `id`-пула;
   - top-k / shortlist workflow вместо слишком широкой `high` категории.
