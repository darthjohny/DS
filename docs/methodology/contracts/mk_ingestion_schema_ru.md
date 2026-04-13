# Схема загрузки MK-данных

## Цель

Этот документ фиксирует схему новых таблиц для MK-волны.

Задача:

- заранее определить структуру новых таблиц;
- не придумывать таблицы по ходу загрузки;
- отделить raw-импорт, кроссматч, нормализацию меток и обучающие представления.

## Общий принцип

Каждый слой хранит только свою ответственность.

Не допускается:

- raw-таблица, которая уже "случайно" стала обучающим источником;
- результат кроссматча, в который сразу вшита вся логика ранжирования;
- одна большая таблица "про все".

## Разделение схем для новой волны

До training/gate этапов используем две роли схем:

- `public`:
  - широкие таблицы первичной загрузки;
  - чистые повторно используемые наборы данных;
  - таблицы Gaia/MK enrichment, которые потом переиспользуются в нескольких
    следующих шагах.
- `lab`:
  - нормализованные рабочие слои;
  - таблицы обучения, опоры, фильтрации и решения;
  - сводки аудита и представления под конкретные задачи.

Правило:

- широкие raw-выгрузки и повторно используемые чистые наборы данных не должны
  жить только в `lab`;
- таблицы обучения и фильтрации не должны превращаться в `public`-дамп "на
  всякий случай".

## Слой 1. External Raw

Relation:

- `lab.gaia_mk_external_raw`

Назначение:

- хранит внешний spectral source почти в исходном виде.

Минимальные поля:

- `external_row_id`
- `external_catalog_name`
- `external_object_id`
- `ra_deg`
- `dec_deg`
- `raw_sptype`
- `raw_magnitude`
- `raw_source_bibcode`
- `raw_notes`
- `ingested_at_utc`

Правило:

- не нормализуем `raw_sptype` прямо здесь;
- не добавляем сюда Gaia-derived physics как будто это один и тот же источник.

## Слой 2. External Crossmatch

Широкая выгрузка `Gaia Archive` может временно лежать в таблице первичной
загрузки вне основной схемы,
например в `public.raw_landing_table`, но она не заменяет этот слой.

Relation:

- `lab.gaia_mk_external_crossmatch`

Назначение:

- хранит результат positional crossmatch external source и `Gaia`.

Минимальные поля:

- `external_row_id`
- `source_id`
- `xmatch_separation_arcsec`
- `xmatch_rank`
- `xmatch_selected`
- `xmatch_batch_id`
- `matched_at_utc`

Правило:

- если есть несколько совпадений, они не теряются молча;
- `xmatch_selected` помечает выбранный рабочий match;
- остальные кандидаты сохраняются для review path.

## Слой 3. External Labeled

Relation:

- `lab.gaia_mk_external_labeled`

Назначение:

- хранит нормализованные MK labels после parsing и selection лучшего crossmatch.

Минимальные поля:

- `xmatch_batch_id`
- `source_id`
- `external_row_id`
- `external_catalog_name`
- `external_object_id`
- `raw_sptype`
- `spectral_class`
- `spectral_subclass`
- `luminosity_class`
- `peculiarity_suffix`
- `label_parse_status`
- `label_parse_notes`
- `xmatch_separation_arcsec`
- `has_source_conflict`
- `source_conflict_count`
- `labeled_at_utc`

Правило:

- парсинг и нормализация меток живут здесь, а не в raw-таблице;
- сюда попадают только записи с выбранным рабочим match.
- слой строится локально в БД из:
  - `lab.gaia_mk_external_filtered`
  - `lab.gaia_mk_external_crossmatch`
- join идет по `external_row_id`;
- `Gaia` на этом шаге больше не участвует.
- дубли по `source_id` на этом шаге не схлопываются молча;
- `has_source_conflict` и `source_conflict_count` сохраняют audit по множественным labels на один `Gaia source_id`.

## Слой 4. Gaia-Enriched Reference

Перед этим слоем допускается набор повторно используемых чистых данных в
`public`, если они нужны сразу нескольким следующим слоям.

Текущие ожидаемые таблицы:

- `public.gaia_mk_core_enrichment_clean`
- `public.gaia_mk_flame_enrichment_clean`

Relation:

- `lab.gaia_mk_training_reference`

Назначение:

- объединяет нормализованные MK labels с нужными Gaia physical/quality columns.

Минимальные поля:

- `source_id`
- `spectral_class`
- `spectral_subclass`
- `luminosity_class`
- `peculiarity_suffix`
- `teff_gspphot`
- `logg_gspphot`
- `radius_flame`
- `mh_gspphot`
- `bp_rp`
- `parallax`
- `parallax_over_error`
- `phot_g_mean_mag`
- `ruwe`
- `gaia_enriched_at_utc`

Правило:

- это еще не quality-gated training source;
- это нормализованный опорный слой перед шагом `quality/OOD`.
- совместимый псевдоним `radius_gspphot` при необходимости строится только в
  отдельном представлении, а не хранится как основное поле Gaia.

## Слой 5. Quality-Gated Reference

Relation:

- `lab.gaia_mk_quality_gated`

Назначение:

- хранит решение раннего quality/OOD gate.

Минимальные поля:

- все ключевые поля из `gaia_mk_training_reference`
- `quality_state`
- `ood_state`
- `quality_reason`
- `quality_gate_version`
- `quality_gated_at_utc`

Правило:

- gate не затирает сырые physics/label values;
- решение gate хранится как явный слой.

## Слой 5a. Unknown / Review Layer

Relation:

- `lab.gaia_mk_unknown_review`

Назначение:

- хранит строки, которые не должны идти в обычное обучение и оценку как
  нормальный случай;
- отделяет `unknown`, `ood` и `reject` от обычного `pass`-контра.

Минимальные поля:

- `source_id`
- `spectral_class`
- `spectral_subclass`
- `luminosity_class`
- `quality_state`
- `ood_state`
- `quality_reason`
- `review_bucket`
- `quality_gate_version`
- `quality_gated_at_utc`

Правило:

- сомнительные строки не теряются внутри общей таблицы;
- следующий контур может читать эту таблицу отдельно как слой разбора и `OOD`.

## Слой 6. Training Views

Relations:

- `lab.v_gaia_mk_training_dwarfs`
- `lab.v_gaia_mk_training_evolved`
- `lab.v_gaia_mk_router_training`
- `lab.v_gaia_mk_host_training`
- `lab.v_gaia_id_coarse_training`
- `lab.v_gaia_mk_refinement_training`
- `lab.v_gaia_id_ood_training`

Назначение:

- выделяют task-oriented training scenarios.

Примеры:

- `dwarfs` и `evolved` для раздельных сценариев;
- `router_training` для coarse/spectral tasks;
- `host_training` для контуров поиска host-подобных объектов;
- `id_coarse_training` для первого классификатора по `OBAFGKM`;
- `mk_refinement_training` для слоя подклассов;
- `id_ood_training` для отдельной задачи `ID vs OOD`.

Правило:

- обучающие представления не содержат тяжелую прикладную логику;
- они только выделяют reproducible slices поверх уже подготовленных слоев.

## Слой 7. Candidate Scoring View

Relation:

- `lab.v_gaia_mk_candidate_scoring`

Назначение:

- источник для реального `score` и `prioritize`.

Минимальные поля:

- `source_id`
- `teff_gspphot`
- `logg_gspphot`
- `radius_gspphot`
- `mh_gspphot`
- `bp_rp`
- `parallax`
- `parallax_over_error`
- `phot_g_mean_mag`
- `ruwe`

Примечание:

- для представления оценки кандидатов допускается совместимый псевдоним
  `radius_gspphot`, если основной слой хранит официальное поле Gaia
  `radius_flame`.

При необходимости:

- дополнительные observability fields;
- дополнительные service columns для traceability.

## Что Не Храним Как Основной Internal Label

- `mk_label`
- `canonical_mk_string`

Такие поля допустимы только как вспомогательные поля, но не как единственная
внутренняя метка.

## Правило Версионирования

Если parsing или quality gate меняются:

- не переписываем логику молча в одном и том же слое;
- фиксируем поле версии или версионированный рабочий процесс;
- при необходимости делаем новую table/view revision.

## Критерий Готовности Схемы

Схема считается зафиксированной, если:

- слой загрузки знает, в какую таблицу писать каждый шаг;
- loaders знают, откуда читать;
- новая MK-ветка не требует разрушать существующие таблицы.
