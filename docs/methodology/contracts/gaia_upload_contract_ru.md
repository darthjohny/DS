# Gaia Upload Contract For MK Wave

## Цель

Этот документ фиксирует минимальный upload-friendly contract для `Gaia Archive`
после локального B/mk filter-step.

Задача:

- не тащить в user table лишнюю parser-логику;
- не потерять traceability между внешним source и будущим crossmatch;
- зафиксировать один reproducible upload layer до похода в `Gaia`.

## Источник Upload-слоя

Upload table строим не из `raw`, а из:

- `lab.gaia_mk_external_filtered`

Причина:

- в `raw` остаются строки с неподдержанным spectral prefix;
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
- `external_catalog_name` нужен для traceability и будущего multi-source сценария;
- `external_object_id` нужен для audit и ручной проверки;
- `ra_deg`, `dec_deg` нужны для geometric crossmatch;
- `raw_sptype` нужен, чтобы xmatch-enriched выгрузка сразу сохраняла исходную MK-строку рядом с `Gaia`-полями.

## Что Не Тянем В Upload-слой

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

- это уже parser-local или DB-local логика;
- upload table должна быть минимальной и прозрачной;
- label normalization и audit flags не смешиваем с crossmatch-механикой.

## Локальный Filter Для Upload

В upload table идут только строки, где:

- `ready_for_gaia_crossmatch = TRUE`
- `ra_deg IS NOT NULL`
- `dec_deg IS NOT NULL`
- `raw_sptype IS NOT NULL`

На текущем состоянии БД это дает:

- `925840` строк

## Канонический Local Query

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

## Что Получаем После Upload

После загрузки в `Gaia Archive` user table должна сохранять:

- стабильный внешний ключ `external_row_id`;
- координаты для built-in crossmatch;
- исходный `raw_sptype` рядом с будущим `source_id`.

Это позволяет потом выгрузить xmatch-enriched слой без отдельного ручного merge по локальным CSV.

## Что Остается Следующим Шагом

Этот contract не определяет:

- radius и batch policy для crossmatch;
- правило выбора лучшего match среди нескольких кандидатов;
- финальную структуру `lab.gaia_mk_external_crossmatch`;
- label normalization после `Gaia`.

Это уже отдельный следующий шаг.

## Источники

- [astroquery Gaia: upload table and cross-match](https://astroquery.readthedocs.io/en/latest/gaia/gaia.html)
- [Gaia Crossmatch Strategy For MK Wave](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/gaia_crossmatch_strategy_ru.md)
- [MK Ingestion Workflow](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/mk_ingestion_workflow_ru.md)

## Критерий Готовности Контракта

Контракт считается зафиксированным, если:

- понятно, из какой relation строится upload table;
- список колонок не придумывается вручную по ходу;
- upload layer не смешан с label normalization;
- следующий Gaia-step можно начинать без повторного обсуждения состава user table.
