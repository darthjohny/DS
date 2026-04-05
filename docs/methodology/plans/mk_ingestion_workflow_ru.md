# MK Ingestion Workflow

## Цель

Этот документ описывает reproducible workflow новой MK-волны от внешнего spectral source до локальной БД.

Задача:

- заранее определить последовательность действий;
- разделить шаги `ADQL/Gaia Archive` и локальные шаги;
- не импровизировать при реальном скачивании и загрузке данных.

## Общая Схема

Порядок такой:

1. скачать внешний spectral source;
2. подготовить upload table;
3. загрузить ее в `Gaia Archive`;
4. выполнить positional crossmatch;
5. присоединить Gaia physics/quality columns;
6. выгрузить результат;
7. загрузить результат в локальную БД по слоям;
8. выполнить label normalization;
9. выполнить quality/OOD gate;
10. собрать training/scoring views.

## Что Делаем Локально До Gaia Archive

Локально:

- скачиваем внешний spectral catalog;
- сохраняем raw dump;
- готовим upload-friendly table;
- строим ее из `lab.gaia_mk_external_filtered`, а не из `raw`;
- не выполняем финальный parsing MK-label прямо на первом шаге;
- проверяем наличие:
  - `RA`
  - `Dec`
  - `raw_sptype`
  - внешнего object id

Результат:

- table готова к upload в `Gaia Archive`.
- точный состав user table зафиксирован в [gaia_upload_contract_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/gaia_upload_contract_ru.md).

## Что Делаем В Gaia Archive

### Шаг 1. Upload User Table

В `Gaia Archive`:

- логинимся;
- загружаем user table;
- даем таблице осмысленное имя;
- убеждаемся, что координатные поля заданы корректно.

### Шаг 2. Positional Crossmatch

Через `Gaia Archive` выполняем:

- crossmatch user table с `gaiadr3.gaia_source`

Задача:

- получить кандидатов совпадения с separation;
- не выбрасывать ambiguity до явного правила отбора.

### Шаг 3. Join С Gaia Полями

Через ADQL после crossmatch:

- присоединяем нужные columns из `gaiadr3.gaia_source`;
- при необходимости присоединяем таблицу физических параметров.

Результат:

- выгружаемый xmatch-enriched dataset.

## Что Делаем Локально После Gaia Archive

### Шаг 4. Загрузка В Локальную БД

Сначала допустим широкий raw landing relation из выгрузки `Gaia`:

- например `public.raw_landing_table`;
- это reproducible landing zone, а не canonical рабочий слой.
- текущий live export `xmatch_bmk_gaia_dr3-result.csv` уже положен именно так.

Потом по слоям:

- `gaia_mk_external_raw`
- `gaia_mk_external_crossmatch`
- `gaia_mk_external_labeled`
- `public.gaia_mk_core_enrichment_clean`
- `public.gaia_mk_flame_enrichment_clean`
- `gaia_mk_training_reference`
- `gaia_mk_quality_gated`
- `gaia_mk_unknown_review`

### Шаг 5. Parsing И Label Normalization

Локально:

- используем `lab.gaia_mk_external_crossmatch`, а не wide raw landing relation;
- берем только `xmatch_selected = TRUE`;
- join-им с `lab.gaia_mk_external_filtered` по `external_row_id`;
- парсим `raw_sptype`;
- нормализуем:
  - `spectral_class`
  - `spectral_subclass`
  - `luminosity_class`
  - `peculiarity_suffix`

Это не делаем внутри giant ADQL query.

Текущий рабочий принцип:

- `Gaia` уже отдала нам enriched raw export;
- дальнейшая label normalization выполняется локально в БД;
- повторный поход в `Gaia` для этого шага не нужен.
- `external_labeled` пока не обязан быть уникальным по `source_id`;
- конфликты по `source_id` сохраняются как отдельный audit-сигнал для следующего слоя.

### Шаг 5a. Chunked Gaia Enrichment Для Missing Official Fields

Если в текущем wide `Gaia` landing relation нет части официальных полей,
нужных для `training_reference`, делаем отдельный повторный `Gaia` шаг.

Текущий live-case:

- `public.raw_landing_table` не содержит `radius_flame`;
- source для batch-wise enrichment готовим из `lab.gaia_mk_external_labeled`;
- батчи строим локально и не смешиваем этот шаг с `training_reference`.

Подробный contract зафиксирован в:

- `docs/methodology/gaia_enrichment_chunking_ru.md`
- `docs/methodology/hierarchical_ood_strategy_ru.md`
- `docs/methodology/db_layer_closure_plan_ru.md`

### Шаг 6. Quality/OOD Gate

После parsing:

- применяем `RUWE`;
- применяем `parallax_over_error`;
- проверяем критические признаки;
- формируем `quality_state` и `ood_state`.

### Шаг 7. Training/Scoring Views

Только после этого:

- создаем новые training views;
- создаем candidate scoring view;
- подключаем loaders.

## Что Делаем Через ADQL

Через ADQL/Gaia Archive делаем:

- upload user table;
- crossmatch с `gaiadr3.gaia_source`;
- join на Gaia columns;
- выгрузку xmatch-enriched результата.

## Что Делаем Не Через ADQL

Локально делаем:

- raw catalog download;
- staging/import в локальную БД;
- materialization узкого crossmatch слоя из wide raw landing relation;
- parsing `SpType`;
- label normalization;
- quality/OOD gating;
- построение final training/scoring relation.

## Почему Делим Workflow Именно Так

- `Gaia Archive` хорош для xmatch и Gaia joins;
- локальный слой лучше подходит для parsing, versioning, audit и reproducibility;
- это делает процесс проще для отладки и легче для повторного прогона.

## Что Считается Успешным Завершением Workflow

Workflow считается закрытым, если:

- внешний source скачан;
- user table загружена в `Gaia Archive`;
- xmatch выполнен;
- enriched result попал в локальную БД;
- canonical crossmatch relation materialized из wide landing layer;
- labels нормализованы;
- quality/OOD gate выполнен;
- training/scoring views готовы к подключению.
- reusable clean assets разложены по `public`, а working/gate layers — по `lab`.

## Что Делаем После Закрытия Workflow

После закрытия этого workflow:

- идем к новой data engineering-ветке;
- подключаем новые loaders;
- запускаем новую MK training-волна без разрушения старого слоя.
