# Контур Базы Данных Проекта

Дата фиксации: `2026-04-06`

Связанные документы:

- [db_layer_closure_plan_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/db_layer_closure_plan_ru.md)
- [db_relation_policy_mk_wave_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/db_relation_policy_mk_wave_ru.md)
- [mk_ingestion_schema_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/mk_ingestion_schema_ru.md)
- [training_view_contracts_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/training_view_contracts_ru.md)

## Зачем Нужен Этот Документ

Этот документ нужен как короткая и практическая карта БД проекта.

Его задача:

- быстро показать, какие схемы и таблицы реально используются;
- объяснить, где лежат reusable данные, а где рабочие слои pipeline;
- дать понятный маршрут от внешних данных до боевого `decide`.

## Главный Принцип

В проекте используются две основные роли схем:

- `public` — reusable и относительно нейтральные слои данных;
- `lab` — рабочие, нормализованные и pipeline-специфичные relation.

Коротко:

- `public` хранит то, что можно переиспользовать в нескольких сценариях;
- `lab` хранит то, что уже связано с логикой проекта, quality gate, training,
  review и final decision.

## Схема `public`

В `public` лежат:

- чистые reference-наборы;
- reusable enrichment-таблицы Gaia;
- очищенные OOD-source и другие слои, которые не завязаны на один конкретный
  training или decision-проход.

### Ключевые relation

- `public.gaia_ref_class_*`
  Чистые reference-наборы по крупным спектральным классам.
- `public.gaia_ref_evolved_class_*`
  Reference-наборы для evolved-ветки.
- `public.gaia_id_flame_enrichment_clean`
  Reusable enrichment для coarse/ID-контура.
- `public.gaia_mk_core_enrichment_clean`
  Базовое Gaia-enrichment для MK-ветки.
- `public.gaia_mk_flame_enrichment_clean`
  FLAME-enrichment для MK-ветки.
- `public.gaia_ood_candidate_pool_clean`
  Очищенный пул объектов для OOD-задачи.

## Схема `lab`

В `lab` лежат:

- рабочие relation после нормализации и crossmatch;
- quality-gated слой;
- review-таблицы;
- training/reference relation;
- task-oriented views;
- локальные audit и summary-слои.

### Ключевые relation

- `lab.gaia_mk_external_raw`
  Сырой импорт внешнего спектрального источника.
- `lab.gaia_mk_external_filtered`
  Локально очищенный слой до Gaia crossmatch.
- `lab.gaia_mk_external_crossmatch`
  Связка external source и Gaia.
- `lab.gaia_mk_external_labeled`
  Нормализованные спектральные метки после parsing и выбора рабочего match.
- `lab.gaia_mk_training_reference`
  Нормализованный Gaia-enriched слой перед quality gate.
- `lab.gaia_mk_quality_gated`
  Главный рабочий вход для боевого `decide`.
- `lab.gaia_mk_unknown_review`
  Отдельный review-контур для `unknown / reject / ood`-случаев.
- `lab.gaia_id_coarse_reference`
  Reference-слой для coarse/ID-ветки.
- `lab.gaia_ood_training_reference`
  Reference-слой для OOD-задачи.

## Главный Поток Данных

Практический маршрут данных в проекте выглядит так:

1. внешний спектральный источник попадает в `lab.gaia_mk_external_raw`;
2. после локальной очистки он переходит в `lab.gaia_mk_external_filtered`;
3. после Gaia crossmatch появляется `lab.gaia_mk_external_crossmatch`;
4. после parsing и нормализации меток формируется
   `lab.gaia_mk_external_labeled`;
5. после обогащения Gaia-параметрами формируется
   `lab.gaia_mk_training_reference`;
6. после quality/OOD-логики формируется `lab.gaia_mk_quality_gated`;
7. uncertain-случаи уходят в `lab.gaia_mk_unknown_review`;
8. training и scoring читают task-specific views;
9. боевой `decide` читает `lab.gaia_mk_quality_gated`.

## Task-Oriented Views

Над рабочими relation строятся view для разных модельных задач.

### Основные views

- `lab.v_gaia_id_coarse_training`
  Источник для coarse-классификации `OBAFGKM`.
- `lab.v_gaia_mk_refinement_training`
  Источник для subclass/refinement-ветки.
- `lab.v_gaia_id_ood_training`
  Источник для задачи `ID vs OOD`.

Именно эти views уже превращают рабочие relation в model-ready training slices.

## Какая Таблица Является Главной Для Боевого Inference

Для текущего active baseline главным входом боевого pipeline является:

- `lab.gaia_mk_quality_gated`

Это важно, потому что:

- именно этот слой уже содержит `quality_state`;
- именно он согласован с текущим `quality_gate`;
- именно его использование делает результат сопоставимым с active baseline и
  validation run.

## Что Считать Правильной Точкой Входа Для Разных Сценариев

### Если нужен нормальный проектный pipeline

Используем:

- `lab.gaia_mk_quality_gated`

Это правильный путь для:

- боевого `decide`;
- technical notebook;
- сравнения с active baseline;
- воспроизводимой проверки проекта.

### Если нужен training-контур

Используем:

- `lab.v_gaia_id_coarse_training`
- `lab.v_gaia_mk_refinement_training`
- `lab.v_gaia_id_ood_training`

### Если нужен review uncertain-слоя

Используем:

- `lab.gaia_mk_unknown_review`

## Что Не Нужно Делать

Не нужно:

- читать `public`-таблицы напрямую в боевой `decide`;
- подменять `lab.gaia_mk_quality_gated` сырым Gaia CSV без явного понимания
  последствий;
- смешивать reusable assets из `public` и project-specific routing relation из
  `lab` в одну “универсальную” таблицу.

## Короткий Вывод

Если объяснять совсем просто, то БД проекта устроена так:

- `public` хранит базовые reusable данные;
- `lab` хранит рабочий контур проекта;
- главная боевая таблица для текущего inference-контура —
  `lab.gaia_mk_quality_gated`.

Именно от нее удобно отталкиваться и при отладке, и при проверке, и при
объяснении работы проекта внешнему человеку.
