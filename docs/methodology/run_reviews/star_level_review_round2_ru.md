# Второй обзор по отдельным объектам

## Цель

Этот документ фиксирует второй предметный разбор прогона `final decision` после
ужатия порогов `priority`:

- `artifacts/decisions/hierarchical_final_decision_2026_03_29_111132_270743`

Задача второго обзора:

- проверить, что изменение порогов не ломает верхнюю маршрутизацию;
- посмотреть, стало ли распределение `priority` содержательнее;
- перепроверить, где именно остается проблема с классом `O`.

## Сводка по прогону

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

## Что изменилось относительно базового прогона

Политика порогов изменилась только в слое `priority`:

- `high_min = 0.85`
- `medium_min = 0.55`

При этом:

- `final_domain_state` не изменился;
- `final_quality_state` не изменился;
- `final_decision_reason` не изменился.

Изменилось только распределение приоритетов:

- `high -> medium = 28125`
- `medium -> low = 4647`

Итог:

- `priority` стал заметно менее насыщенным наверху;
- никаких признаков скрытой связности между порогами и верхней маршрутизацией не видно.

## Ключевые Наблюдения

### 1. Главный источник `unknown` по-прежнему не связан с моделью

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
  `refinement`, а `quality_gate` и наличием основных физических признаков.

### 2. `priority` стал содержательнее, но `K` все еще доминирует в зоне `high`

Новая class-level структура:

- `A`: `100% low`
- `B`: `100% low`
- `F`: `49.56% high`
- `G`: `46.86% high`
- `K`: `69.51% high`
- `M`: `1.64% high`

Вывод:

- ужесточение порогов действительно помогло;
- но если верхняя зона `K` останется слишком широкой на практике,
  следующим шагом нужен отдельный обзор масштабирования, а не новый подбор порогов.

### 3. Класс `O` по-прежнему практически отсутствует в итоговом выводе

На текущем прогоне:

- `final_coarse_class = O` встречается только `1` раз

Это согласуется с уже зафиксированной проблемой:

- `quality_gate` не вырезает `O` полностью;
- проблема остается в поведении coarse-модели на редком хвосте распределения.

## Практический вывод после второго обзора

Система на текущем прогоне ведет себя стабильнее с точки зрения `priority`:

- зона `high` стала уже;
- зона `medium` стала полезнее;
- верхняя маршрутизация не изменилась.

Но два открытых вопроса остаются:

1. нужен ли отдельный слой масштабирования `priority` поверх нового прогона с порогами;
2. как отдельно разбирать исчезновение класса `O`.

## Следующие Шаги

1. Продолжить обзор по отдельным объектам уже на этом прогоне через
   [final_decision_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/final_decision_review.ipynb).
2. Провести отдельный обзор редкого хвоста coarse-модели для `O` через
   [11_coarse_o_tail_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/archive_research/11_coarse_o_tail_review.ipynb)
   и зафиксировать выводы в
   [coarse_o_tail_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_tail_review_round1_ru.md).
3. Только после этого решать, нужен ли отдельный пакет масштабирования `priority`.

## Связанные документы

- [priority_threshold_review_round2_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/priority_threshold_review_round2_ru.md)
- [coarse_o_tail_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_tail_review_round1_ru.md)
- [stabilization_issue_ledger_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/stabilization_issue_ledger_ru.md)
