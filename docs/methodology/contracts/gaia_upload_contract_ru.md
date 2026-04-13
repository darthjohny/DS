# Контракт загрузки в Gaia для MK-волны

## Цель

Этот документ фиксирует минимальный контракт загрузки в `Gaia Archive`
после локального шага фильтрации `B/mk`.

Задача:

- не тащить в user table лишнюю parser-логику;
- не потерять прослеживаемость между внешним источником и будущим кроссматчем;
- зафиксировать один воспроизводимый слой загрузки перед обращением к `Gaia`.

## Источник слоя загрузки

Таблицу загрузки строим не из `raw`, а из:

- `lab.gaia_mk_external_filtered`

Причина:

- в `raw` остаются строки с неподдержанным спектральным префиксом;
- `filtered` уже гарантирует координаты, непустой `raw_sptype` и готовность к следующему шагу;
- это уменьшает шум до `Gaia Archive` без потери полезных объектов.

## Минимальные Upload-колонки

В первую волну фиксируем такой набор:

- `external_row_id`
- `external_catalog_name`
- `external_object_id`
- `ra_deg`
- `dec_deg`
- `raw_sptype`

Зачем:

- `external_row_id` нужен как стабильный локальный ключ;
- `external_catalog_name` нужен для прослеживаемости и будущего сценария с
  несколькими источниками;
- `external_object_id` нужен для audit и ручной проверки;
- `ra_deg`, `dec_deg` нужны для geometric crossmatch;
- `raw_sptype` нужен, чтобы выгрузка после кроссматча сразу сохраняла исходную
  строку MK рядом с полями `Gaia`.

## Что не тянем в слой загрузки

На этом шаге не отправляем в `Gaia Archive`:

- `spectral_prefix`
- `spectral_class`
- `spectral_subclass`
- `luminosity_class`
- `parse_status`
- `parse_note`
- `has_supported_prefix`
- `has_coordinates`
- `has_raw_sptype`
- `ready_for_gaia_crossmatch`
- timestamps локальной БД

Причина:

- это уже логика парсера или локальной базы данных;
- таблица загрузки должна быть минимальной и прозрачной;
- нормализацию меток и audit-флаги не смешиваем с механикой кроссматча.

## Локальный фильтр для загрузки

В upload table идут только строки, где:

- `ready_for_gaia_crossmatch = TRUE`
- `ra_deg IS NOT NULL`
- `dec_deg IS NOT NULL`
- `raw_sptype IS NOT NULL`

На текущем состоянии БД это дает:

- `925840` строк

## Канонический локальный запрос

Канонический источник для следующего шага собирается так:

```sql
SELECT
    external_row_id,
    external_catalog_name,
    external_object_id,
    ra_deg,
    dec_deg,
    raw_sptype
FROM lab.gaia_mk_external_filtered
WHERE ready_for_gaia_crossmatch IS TRUE
  AND ra_deg IS NOT NULL
  AND dec_deg IS NOT NULL
  AND raw_sptype IS NOT NULL
ORDER BY external_row_id ASC;
```

Этот же контракт зафиксирован в коде через:

- [bmk_upload.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/db/bmk_upload.py)

## Что получаем после загрузки

После загрузки в `Gaia Archive` user table должна сохранять:

- стабильный внешний ключ `external_row_id`;
- координаты для built-in crossmatch;
- исходный `raw_sptype` рядом с будущим `source_id`.

Это позволяет потом выгрузить слой после кроссматча без отдельного ручного
слияния локальных `CSV`.

## Что остается следующим шагом

Этот contract не определяет:

- политику радиуса и пакетной обработки для кроссматча;
- правило выбора лучшего match среди нескольких кандидатов;
- финальную структуру `lab.gaia_mk_external_crossmatch`;
- label normalization после `Gaia`.

Это уже отдельный следующий шаг.

## Источники

- [astroquery Gaia: upload table and cross-match](https://astroquery.readthedocs.io/en/latest/gaia/gaia.html)
- [Gaia Crossmatch Strategy For MK Wave](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/gaia_crossmatch_strategy_ru.md)
- [MK Ingestion Workflow](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/mk_ingestion_workflow_ru.md)

## Критерий готовности контракта

Контракт считается зафиксированным, если:

- понятно, из какой таблицы строится таблица загрузки;
- список колонок не придумывается вручную по ходу;
- слой загрузки не смешан с нормализацией меток;
- следующий шаг в `Gaia` можно начинать без повторного обсуждения состава
  пользовательской таблицы.
