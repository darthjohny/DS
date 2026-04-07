# Контур базы данных проекта

Дата фиксации: `2026-04-06`

Связанные документы:

- [db_relation_policy_mk_wave_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/db_relation_policy_mk_wave_ru.md)
- [mk_ingestion_schema_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/mk_ingestion_schema_ru.md)
- [training_view_contracts_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/training_view_contracts_ru.md)

## Зачем нужен этот документ

Этот документ нужен как короткая и практическая карта БД проекта.

Его задача:

- быстро показать, какие схемы и таблицы реально используются;
- объяснить, где лежат повторно используемые данные, а где рабочие слои проекта;
- дать понятный маршрут от внешних данных до боевого `decide`.

## Главный принцип

В проекте используются две основные роли схем:

- `public` — повторно используемые и относительно нейтральные слои данных;
- `lab` — рабочие, нормализованные и завязанные на логику проекта таблицы.

Коротко:

- `public` хранит то, что можно использовать в нескольких сценариях;
- `lab` хранит то, что уже связано с логикой проекта, фильтрацией качества,
  обучением, разбором спорных случаев и итоговым решением.

## Схема `public`

В `public` лежат:

- чистые reference-наборы;
- таблицы обогащения Gaia, которые можно использовать повторно;
- очищенные OOD-source и другие слои, которые не завязаны на один конкретный
  обучающий или боевой прогон.

### Ключевые таблицы

- `public.gaia_ref_class_*`
  Чистые reference-наборы по крупным спектральным классам.
- `public.gaia_ref_evolved_class_*`
  Reference-наборы для evolved-ветки.
- `public.gaia_id_flame_enrichment_clean`
  Таблица обогащения для coarse- и ID-контура.
- `public.gaia_mk_core_enrichment_clean`
  Базовое Gaia-enrichment для MK-ветки.
- `public.gaia_mk_flame_enrichment_clean`
  FLAME-enrichment для MK-ветки.
- `public.gaia_ood_candidate_pool_clean`
  Очищенный пул объектов для OOD-задачи.

## Схема `lab`

В `lab` лежат:

- рабочие таблицы после нормализации и crossmatch;
- quality-gated слой;
- таблицы разбора спорных случаев;
- обучающие и опорные таблицы;
- представления для отдельных задач;
- локальные слои аудита и сводок.

### Ключевые таблицы

- `lab.gaia_mk_external_raw`
  Сырой импорт внешнего спектрального источника.
- `lab.gaia_mk_external_filtered`
  Локально очищенный слой до Gaia crossmatch.
- `lab.gaia_mk_external_crossmatch`
  Связка external source и Gaia.
- `lab.gaia_mk_external_labeled`
  Нормализованные спектральные метки после разбора и выбора рабочего совпадения.
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

## Главный поток данных

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
8. обучение и расчеты моделей читают специализированные представления;
9. боевой `decide` читает `lab.gaia_mk_quality_gated`.

## Представления для отдельных задач

Над рабочими таблицами строятся представления для разных модельных задач.

### Основные представления

- `lab.v_gaia_id_coarse_training`
  Источник для coarse-классификации `OBAFGKM`.
- `lab.v_gaia_mk_refinement_training`
  Источник для subclass/refinement-ветки.
- `lab.v_gaia_id_ood_training`
  Источник для задачи `ID vs OOD`.

Именно эти представления превращают рабочие таблицы в готовые обучающие выборки.

## Главная таблица для боевого прогона

Для текущего базового прогона главным входом боевого контура является:

- `lab.gaia_mk_quality_gated`

Это важно, потому что:

- именно этот слой уже содержит `quality_state`;
- именно он согласован с текущей логикой фильтрации качества;
- именно его использование делает результат сопоставимым с базовым и
  проверочным прогоном.

## Какая точка входа нужна в разных сценариях

### Если нужен полный проектный маршрут

Используем:

- `lab.gaia_mk_quality_gated`

Это правильный путь для:

- боевого `decide`;
- технических ноутбуков;
- сравнения с базовым прогоном;
- воспроизводимой проверки проекта.

### Если нужен training-контур

Используем:

- `lab.v_gaia_id_coarse_training`
- `lab.v_gaia_mk_refinement_training`
- `lab.v_gaia_id_ood_training`

### Если нужен разбор спорных случаев

Используем:

- `lab.gaia_mk_unknown_review`

## Чего делать не нужно

Не нужно:

- читать таблицы `public` напрямую в боевой `decide`;
- подменять `lab.gaia_mk_quality_gated` сырым Gaia CSV без явного понимания
  последствий;
- смешивать повторно используемые данные из `public` и рабочие таблицы из
  `lab` в одну «универсальную» таблицу.

## Короткий вывод

Если объяснять совсем просто, то БД проекта устроена так:

- `public` хранит базовые повторно используемые данные;
- `lab` хранит рабочий контур проекта;
- главная боевая таблица для текущего контура обработки —
  `lab.gaia_mk_quality_gated`.

Именно от нее удобно отталкиваться и при отладке, и при проверке, и при
объяснении работы проекта внешнему человеку.
