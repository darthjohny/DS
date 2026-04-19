# Контракт интерфейса Streamlit

Дата фиксации: `2026-04-16`

Связанные документы:

- [streamlit_interface_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/streamlit_interface_tz_ru.md)
- [external_decide_input_contract_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/external_decide_input_contract_ru.md)
- [final_decision_contract_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/final_decision_contract_ru.md)

## Зачем нужен этот документ

Интерфейс не должен читать артефакты и входные таблицы «по памяти». Этот
документ фиксирует, что именно первая версия интерфейса считает обязательным:

- какие файлы должны лежать в `run_dir`;
- какие колонки нужны для чтения итогового прогона;
- какой минимальный `CSV` можно подать во внешний запуск `decide`;
- какое минимальное состояние хранится внутри интерфейса.

## Какие артефакты читает интерфейс

Первая версия интерфейса работает только с прогоном
`hierarchical_final_decision`.

Обязательные файлы в `run_dir`:

- `decision_input.csv`
- `final_decision.csv`
- `priority_input.csv`
- `priority_ranking.csv`
- `metadata.json`

Без этого набора интерфейс не считает каталог корректным рабочим прогоном.

## Какие таблицы обязательны для просмотра запуска

### `decision_input.csv`

Обязательные колонки:

- `source_id`
- `quality_state`
- `quality_reason`
- `review_bucket`
- `ood_state`
- `ood_reason`
- `ood_decision`
- `coarse_predicted_label`
- `coarse_probability_max`

Этого достаточно, чтобы показать входной статус объекта, его качество и раннее
маршрутизирующее решение.

### `final_decision.csv`

Обязательные колонки:

- `source_id`
- `final_domain_state`
- `final_quality_state`
- `final_coarse_class`
- `final_refinement_state`
- `final_decision_reason`
- `final_decision_policy_version`
- `priority_state`

Эта таблица нужна для итогового решения, групповых сводок и карточки объекта.

### `priority_input.csv`

Обязательные колонки:

- `source_id`
- `spec_class`
- `host_similarity_score`

Эта таблица нужна для связи между итоговым решением и слоем приоритизации.

### `priority_ranking.csv`

Обязательные колонки:

- `source_id`
- `spec_class`
- `class_priority_score`
- `host_similarity_score`
- `priority_score`
- `priority_label`
- `priority_reason`

Эта таблица нужна для показа итогового ранга и верхнего списка кандидатов.

## Что интерфейс требует от `metadata.json`

Интерфейс ожидает:

- основные счетчики строк;
- список колонок сохраненных таблиц;
- распределение `final_domain_state`;
- распределение `priority_label`;
- вложенный объект `context`.

Во вложенном `context` для первой версии считаются обязательными:

- пути к основным model artifacts;
- список `refinement_model_run_dirs` для повторного CSV-запуска;
- версия policy;
- пороги `priority`;
- настройки `quality_gate`;
- имя relation или путь входного `CSV`.

Это нужно для страницы метрик, краткой сводки запуска и доверия к результату.

## Какой внешний CSV можно подать из интерфейса

Минимальный набор колонок для первой версии кнопочного запуска:

- `source_id`
- `quality_state`
- `teff_gspphot`
- `logg_gspphot`
- `mh_gspphot`
- `bp_rp`
- `parallax`
- `parallax_over_error`
- `ruwe`
- `phot_g_mean_mag`
- `radius_flame`
- `lum_flame`
- `evolstage_flame`

Рекомендуется дополнительно передавать:

- `radius_gspphot`
- `ra`
- `dec`
- `random_index`

Если внешний пользователь не воспроизводит проектный quality-слой, быстрый
демонстрационный запуск допускает колонку `quality_state = 'pass'` для всех
строк. Полное объяснение этого сценария дано в
[external_decide_input_contract_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/external_decide_input_contract_ru.md).

## Какое состояние хранит интерфейс

Минимальное состояние первой версии:

- выбранный `run_dir`;
- выбранный `source_id`;
- путь к загруженному внешнему `CSV`;
- путь к последнему созданному `run_dir`;
- последняя ошибка чтения `run_dir`;
- последняя ошибка проверки внешнего `CSV`.

Интерфейс не хранит у себя модельные решения и не дублирует прикладную
логику. Он хранит только навигационное и служебное состояние.

## Короткий вывод

Первая версия интерфейса должна быть тонкой витриной. Она читает уже готовые
артефакты, может запускать существующий `decide` по совместимому `CSV` и не
создает собственного слоя бизнес-логики поверх проекта.
