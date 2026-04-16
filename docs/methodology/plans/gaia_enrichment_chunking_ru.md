# Политика поэтапного обогащения Gaia-данных

## Цель

Этот документ фиксирует следующий шаг после `external_labeled`, если в текущем
wide `Gaia` export не хватает части официальных physical fields для
`training_reference`.

Текущий live-case:

- `public.raw_landing_table` уже содержит core `Gaia` fields;
- но в нем нет `radius_flame`;
- поэтому полный `training_reference` пока преждевременен.

## Источник Для Chunking

Локальный source для chunk-wise enrichment:

- `lab.gaia_mk_external_labeled`

Текущий live staging в БД на `2026-03-28`:

- `lab.gaia_mk_flame_enrichment_source_batches`
- `lab.gaia_mk_flame_enrichment_source_manifest`
- `lab.gaia_mk_flame_enrichment_source_batch_0001`
- `lab.gaia_mk_flame_enrichment_source_batch_0002`
- `lab.gaia_mk_flame_enrichment_source_batch_0003`
- `lab.gaia_mk_flame_enrichment_source_batch_0004`
- `lab.gaia_mk_flame_enrichment_source_batch_0005`
- `lab.gaia_mk_flame_enrichment_source_batch_0006`
- `lab.gaia_mk_flame_enrichment_source_batch_0007`
- `lab.gaia_mk_flame_enrichment_source_batch_0008`
- `lab.gaia_mk_flame_enrichment_source_batch_0009`

Первая безопасная политика:

- берем только `xmatch_batch_id` текущего прогона;
- берем только `source_id IS NOT NULL`;
- берем только `has_source_conflict = FALSE`.

Почему так:

- это первый conflict-free слой после post-Gaia normalization;
- он не смешивает повторный `Gaia` шаг с label dedup внутри SQL на лету.

## Что Выгружаем В Batch-Файлы

Минимальный upload contract:

- `source_id`

Правило:

- не тащим в upload batch лишние label-колонки;
- traceability к labels и `external_row_id` остается в локальной БД.

## Размер Батчей

Батчи делим детерминированно:

- сортировка по `source_id ASC`;
- фиксированный `batch_size`;
- отдельный manifest с диапазонами `source_id_min/source_id_max`.

На первой волне используем практический размер:

- `50000` строк на batch

Текущий live manifest:

- `8` батчей по `50000`
- `1` батч на `2226`

Текущий live результат после обратного импорта из `Gaia`:

- `public.gaia_mk_flame_enrichment_raw` — `402226`
- `public.gaia_mk_flame_enrichment_clean` — `402226`
- `radius_flame` coverage — `223858`
- `lum_flame` coverage — `223858`
- `evolstage_flame` coverage — `212996`

## Минимальный ADQL Шаг

После upload одного batch-а в `Gaia Archive` выполняем минимальный запрос:

```sql
SELECT
    u.source_id,
    ap.radius_flame
FROM user_<login>.<uploaded_source_id_batch> AS u
LEFT JOIN gaiadr3.astrophysical_parameters AS ap
    ON ap.source_id = u.source_id
ORDER BY u.source_id ASC;
```

Почему `LEFT JOIN`:

- он сохраняет audit по `source_id`, для которых `radius_flame` отсутствует;
- не схлопывает silently проблемные строки.

## Что Сохраняем Локально

После выгрузки из `Gaia`:

- каждый batch-result кладем как raw landing artifact;
- затем импортируем в локальную БД отдельным enrichment landing layer;
- не смешиваем эти batch-result relation-ы с `public.raw_landing_table`.

## Что Дальше

Только после этого:

- строим `lab.gaia_mk_training_reference`;
- добавляем compatibility alias для legacy V2 loader-ов, если он реально нужен.

Текущий live-state:

- `lab.gaia_mk_training_reference` уже materialized после импорта FLAME enrichment.
