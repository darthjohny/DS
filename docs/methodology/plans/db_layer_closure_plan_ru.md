# DB Layer Closure Plan For Hierarchical MK/OOD Wave

## Цель

Этот документ фиксирует текущее состояние live-БД на `2026-03-28` и порядок
закрытия DB-слоев до перехода в loaders, views и model-код.

Документ нужен для двух задач:

- понимать, что уже реально лежит в БД, а что еще только запланировано;
- воспроизводимо восстановить ход работ без перепросмотра всей переписки.

## Главный Принцип

Сначала закрываем DB foundation.

Только после этого:

- идем в loaders;
- идем в training/scoring views;
- идем в model-контуры `coarse`, `refinement`, `OOD/reject`.

Причина:

- данных уже достаточно;
- сейчас риск не в нехватке данных, а в смешении ролей relation и схем.

## Schema Layout `public` И `lab`

### `public`

В `public` держим:

- raw landing relation из внешних источников и `Gaia`;
- clean reusable source assets;
- relation, которые можно переиспользовать в нескольких downstream-сценариях
  без model-specific gate logic.

Типовые примеры:

- `public.raw_landing_table`
- `public.gaia_ref_class_*`
- `public.gaia_ref_evolved_class_*`
- `public.gaia_id_flame_*_raw`
- `public.gaia_ood_*_raw`
- `public.gaia_ood_candidate_pool_clean`
- `public.gaia_mk_core_enrichment_clean`
- `public.gaia_mk_flame_enrichment_raw`
- `public.gaia_mk_flame_enrichment_clean`
- `public.gaia_id_flame_enrichment_clean`

### `lab`

В `lab` держим:

- derived working layers;
- normalized label/training/gate/decision relation;
- summaries и audit relation;
- training/scoring views.

Типовые примеры:

- `lab.gaia_mk_external_*`
- `lab.gaia_id_coarse_reference`
- `lab.gaia_mk_training_reference`
- `lab.gaia_ood_training_reference`
- `lab.gaia_mk_quality_gated`
- `lab.gaia_mk_unknown_review`
- `lab.*summary`
- `lab.v_*`

### Текущее Отклонение, Которое Нужно Исправить

Сейчас `lab.gaia_id_flame_enrichment` хранит clean reusable source asset.

По согласованной схеме он должен быть отражен в `public` как:

- `public.gaia_id_flame_enrichment_clean`

`lab.gaia_id_flame_enrichment_summary` при этом может оставаться в `lab`, так как
это audit relation.

## Live State На `2026-03-28`

### 1 Слой. Coarse / ID Foundation

Локальные class-table уже есть:

- `public.gaia_ref_class_o..m` — по `3000`
- `public.gaia_ref_evolved_class_o..m` — по `3000`

Итого:

- `42000` строк coarse/ID-source.

FLAME enrichment для этого слоя уже собран:

- `public.gaia_id_flame_class_result_raw` — `21000`
- `public.gaia_id_flame_evolved_result_raw` — `21000`
- `public.gaia_id_flame_enrichment_clean` — `42000`
- `lab.gaia_id_flame_enrichment` — `42000`
- `lab.gaia_id_flame_enrichment_summary` — `14`
- `lab.gaia_id_coarse_reference` — `39413`

Ключевые факты:

- distinct `source_id`: `39413`
- `radius_flame`: `28048`
- `lum_flame`: `28048`
- `evolstage_flame`: `23282`
- `gaia_ref_class_o` и `gaia_ref_evolved_class_o` дают `0` по `FLAME`

### 2 Слой. MK Refinement Source

Ветка `B/mk -> Gaia -> labeled` уже есть:

- `lab.gaia_mk_external_raw` — `1058787`
- `lab.gaia_mk_external_filtered` — `925840`
- `lab.gaia_mk_external_crossmatch` — `824038`
- `lab.gaia_mk_external_labeled` — `809832`

Ключевые факты:

- distinct `source_id`: `564820`
- conflict-free rows: `402226`
- subclass в целом: `728021`
- subclass conflict-free: `369669`
- `luminosity_class` в целом: `298564`
- `luminosity_class` conflict-free: `131273`

Вывод:

- источник для второго слоя numerically уже достаточный;
- узкий reusable Gaia core asset уже собран как
  `public.gaia_mk_core_enrichment_clean` — `574477`;
- FLAME enrichment уже собран как:
  - `public.gaia_mk_flame_enrichment_raw` — `402226`
  - `public.gaia_mk_flame_enrichment_clean` — `402226`;
- `lab.gaia_mk_training_reference` уже собран — `402226`.

### 3 Слой. OOD Candidate Pool

Первая OOD-волна уже загружена:

- `public.gaia_ood_white_dwarf_raw` — `3000`
- `public.gaia_ood_binary_like_raw` — `3000`
- `public.gaia_ood_emission_line_raw` — `3000`
- `public.gaia_ood_carbon_star_raw` — `3000`
- `public.gaia_ood_outlier_like_raw` — `3000`

Clean unified relation:

- `public.gaia_ood_candidate_pool_clean` — `15000`
- `lab.gaia_ood_training_reference` — `15000`

Audit relation:

- `lab.gaia_ood_candidate_pool_summary` — `5`

Ключевые факты:

- distinct `source_id`: `14987`
- overlapping `source_id`: `13`
- overlap сейчас встречается между:
  - `ood_emission_line`
  - `ood_white_dwarf`

## Что Уже Соответствует Плану

Сейчас уже закрыто:

- отдельный `ID/coarse` source;
- отдельный `refinement` source;
- отдельный `OOD` candidate pool;
- separation `raw -> clean reusable source -> working layer`.

Это соответствует:

- `docs/methodology/hierarchical_ood_strategy_ru.md`
- `docs/methodology/mk_ingestion_schema_ru.md`
- `docs/methodology/quality_ood_contract_ru.md`

## Что Еще Не Закрыто

На `2026-03-28` отсутствуют:

- нет критических DB-gap по source/reference слоям;
- следующим кодовым шагом остаются loaders/views поверх уже закрытых relation.

Именно эти relation теперь являются критическим DB-gap, а не новые внешние
источники данных.

Уже закрыто в live-БД:

- `MTZ-M37`:
  - `public.gaia_id_flame_enrichment_clean`
  - `public.gaia_mk_core_enrichment_clean`
- `MTZ-M38`:
  - `public.gaia_mk_flame_enrichment_raw`
  - `public.gaia_mk_flame_enrichment_clean`
- `MTZ-M39`:
  - `lab.gaia_id_coarse_reference`
- `MTZ-M40`:
  - `lab.gaia_mk_training_reference`
  - `lab.gaia_mk_training_reference_summary`
- `MTZ-M41`:
  - `lab.gaia_ood_training_reference`
- `MTZ-M42`:
  - `lab.gaia_mk_quality_gated`
  - `lab.gaia_mk_quality_gated_summary`
  - `lab.gaia_mk_unknown_review`
  - `lab.gaia_mk_unknown_review_summary`

## Порядок Закрытия DB-Слоев

### Шаг 0. Привести Схемы К Явной Политике

Нужно:

- зафиксировать layout `public` vs `lab` в docs;
- отразить clean reusable relations в `public`;
- оставить audit и working layers в `lab`.

### Шаг 1. Закрыть `public` Reusable Assets

Нужно собрать:

- `public.gaia_mk_flame_enrichment_raw`
- `public.gaia_mk_flame_enrichment_clean`

Уже собрано:

- `public.gaia_id_flame_enrichment_clean`
- `public.gaia_mk_core_enrichment_clean`
- `public.gaia_mk_flame_enrichment_raw`
- `public.gaia_mk_flame_enrichment_clean`

Уже подготовлено для следующего `Gaia` шага:

- `lab.gaia_mk_flame_enrichment_source_batches`
- `lab.gaia_mk_flame_enrichment_source_manifest`
- `lab.gaia_mk_flame_enrichment_source_batch_0001..0009`

### Шаг 2. Закрыть Первый Layer

Нужно собрать:

- `lab.gaia_id_coarse_reference`

Назначение:

- train-grade relation для первого слоя `OBAFGKM`.

Статус:

- закрыто на live-БД.

### Шаг 3. Закрыть Второй Layer

Нужно собрать:

- `lab.gaia_mk_training_reference`

Назначение:

- train-grade relation для `spectral_subclass` и связанных MK-labels.

Статус:

- закрыто на live-БД.

### Шаг 4. Закрыть OOD Layer

Нужно собрать:

- `lab.gaia_ood_training_reference`

Назначение:

- отдельный OOD relation без смешения с `ID`.

Статус:

- закрыто на live-БД.

### Шаг 5. Закрыть Gate И Review Layer

Нужно собрать:

- `lab.gaia_mk_quality_gated`
- `lab.gaia_mk_unknown_review`

Назначение:

- отделить `pass`, `unknown`, `ood`, `reject`;
- не терять uncertain rows внутри общей выборки.

Первая практическая policy:

- `missing core features` -> `reject`
- `missing radius_flame` -> `unknown`
- `ruwe > 1.4` -> `unknown`
- `parallax_over_error < 5` -> `unknown`
- `non_single_star > 0` -> `candidate_ood`
- `classprob_dsc_combmod_star < 0.5` -> `candidate_ood`
- membership в `lab.gaia_ood_training_reference` -> `ood`

Статус:

- закрыто на live-БД.

Текущий live результат:

- `lab.gaia_mk_quality_gated` — `402226`
- `lab.gaia_mk_unknown_review` — `235379`
- `pass/in_domain` — `166847`
- `pass/candidate_ood` — `11572`
- `pass/ood` — `20`
- `unknown/in_domain` — `50442`
- `unknown/candidate_ood` — `13304`
- `unknown/ood` — `77`
- `reject/in_domain` — `148864`
- `reject/candidate_ood` — `11009`
- `reject/ood` — `91`

## Что Делаем Только После Этого

Только после закрытия этих DB-layer:

- обновляем loaders;
- собираем training/scoring views;
- идем в code-side реализацию `coarse`, `refinement`, `OOD/reject`,
  decision layer.

## Связанные Документы

- [DB Relation Policy For MK Wave](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/db_relation_policy_mk_wave_ru.md)
- [MK Ingestion Schema](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/mk_ingestion_schema_ru.md)
- [MK Ingestion Workflow](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/mk_ingestion_workflow_ru.md)
- [Hierarchical Classification And OOD Strategy](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/hierarchical_ood_strategy_ru.md)
- [Quality And OOD Contract V2](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/quality_ood_contract_ru.md)
- [V2 Maturation Micro-TZ](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/architecture/v2_maturation_micro_tz_ru.md)
