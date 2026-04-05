# Star-Level Review Round 2

## Цель

Этот документ фиксирует второй предметный разбор `final decision` run после
ужатия `priority` thresholds:

- `artifacts/decisions/hierarchical_final_decision_2026_03_29_111132_270743`

Задача round 2:

- проверить, что изменение thresholds не ломает верхний routing;
- посмотреть, стала ли `priority`-стратификация содержательнее;
- перепроверить, где именно остается проблема с классом `O`.

## Сводка По Run

- `n_rows_input = 402226`
- `n_rows_final_decision = 402226`
- `n_rows_priority_input = 177674`
- `n_rows_priority_ranking = 177674`
- `n_unique_source_id = 402226`

Распределение `final_domain_state`:

- `unknown = 223787` (`55.64%`)
- `id = 177674` (`44.17%`)
- `ood = 765` (`0.19%`)

Распределение `final_quality_state`:

- `pass = 178439` (`44.36%`)
- `reject = 159964` (`39.77%`)
- `unknown = 63823` (`15.87%`)

Распределение `priority_label`:

- `high = 72048` (`40.55%`)
- `medium = 41851` (`23.55%`)
- `low = 63775` (`35.89%`)

## Что Изменилось Относительно Baseline

Threshold-policy изменилась только в `priority`-слое:

- `high_min = 0.85`
- `medium_min = 0.55`

При этом:

- `final_domain_state` не изменился;
- `final_quality_state` не изменился;
- `final_decision_reason` не изменился.

Изменилась только downstream priority stratification:

- `high -> medium = 28125`
- `medium -> low = 4647`

Итог:

- `priority` стал заметно менее насыщенным наверху;
- никаких признаков hidden coupling между thresholds и верхним routing не видно.

## Ключевые Наблюдения

### 1. Главный Драйвер `unknown` По-Прежнему Не Модельный

Текущая top-структура `quality_reason`:

- `pass = 178439`
- `missing_core_features = 159964`
- `high_ruwe = 39966`
- `missing_radius_flame = 18406`
- `low_parallax_snr = 5451`

Текущая top-структура `review_bucket`:

- `pass = 166847`
- `reject_missing_core_features = 159873`
- `review_high_ruwe = 28388`
- `review_missing_radius_flame = 16762`
- `review_non_single_star = 14402`

Вывод:

- доля `unknown/reject` по-прежнему определяется не слабостью `coarse` или
  `refinement`, а `quality_gate` и availability core physics.

### 2. `priority` Стал Содержательнее, Но `K` Все Еще Доминирует В High-Zone

Новая class-level структура:

- `A`: `100% low`
- `B`: `100% low`
- `F`: `49.56% high`
- `G`: `46.86% high`
- `K`: `69.51% high`
- `M`: `1.64% high`

Вывод:

- tightening thresholds действительно помог;
- но если top-zone `K` останется operationally слишком широкой,
  следующим шагом нужен отдельный scaling review, а не новый random threshold tweak.

### 3. Класс `O` По-Прежнему Практически Отсутствует В Final Output

На текущем run:

- `final_coarse_class = O` встречается только `1` раз

Это согласуется с уже зафиксированным issue:

- `quality_gate` не вырезает `O` полностью;
- проблема остается в поведении coarse-model на rare-tail классе.

## Практический Вывод После Round 2

Система на текущем run ведет себя стабильнее с точки зрения `priority`:

- high-zone стала уже;
- medium-zone стала полезнее;
- upper routing не изменился.

Но два открытых вопроса остаются:

1. нужен ли `priority scaling` поверх нового threshold-run;
2. как отдельно разбирать исчезновение класса `O`.

## Следующие Шаги

1. Продолжить star-level review уже на этом run через
   [final_decision_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/final_decision_review.ipynb).
2. Провести отдельный coarse rare-tail review для `O` через
   [11_coarse_o_tail_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/archive_research/11_coarse_o_tail_review.ipynb)
   и зафиксировать findings в
   [coarse_o_tail_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_tail_review_round1_ru.md).
3. Только после этого решать, нужен ли отдельный `priority scaling` пакет.

## Related

- [priority_threshold_review_round2_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/priority_threshold_review_round2_ru.md)
- [coarse_o_tail_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_tail_review_round1_ru.md)
- [stabilization_issue_ledger_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/stabilization_issue_ledger_ru.md)
- [post_run_stabilization_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/post_run_stabilization_tz_ru.md)
