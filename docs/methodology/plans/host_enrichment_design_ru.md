# Host Enrichment Design

## Цель

Этот документ фиксирует clean design следующего data-step для host-layer.

Задача шага:

- не возвращать `priority` в mainline поверх legacy host source;
- сначала обогатить current host source официальными Gaia DR3 FLAME-полями;
- только после этого пересобирать host training relation и делать retrain.

## На Что Опираемся

### Локальные Контракты

- [db_relation_policy_mk_wave_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/db_relation_policy_mk_wave_ru.md)
- [host_target_semantics_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/host_target_semantics_ru.md)
- [host_priority_feature_contract_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/host_priority_feature_contract_ru.md)
- [host_priority_integration_path_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/host_priority_integration_path_ru.md)
- [quality_gate_host_priority_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/quality_gate_host_priority_tz_ru.md)

### Official Gaia DR3

- [Gaia DR3 gaia_source datamodel](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html)
- [Gaia DR3 astrophysical_parameters datamodel](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_astrophysical_parameter_tables/ssec_dm_astrophysical_parameters.html)
- [Gaia DR3 astrometric validation](https://gea.esac.esa.int/archive/documentation/GDR3/Catalogue_consolidation/chap_cu9val/sec_cu9val_942/ssec_cu9val_942_astrometry.html)

### Official NASA Exoplanet Archive

- [NASA Exoplanet Archive TAP guide](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html)
- [Planetary Systems Composite Parameters](https://exoplanetarchive.ipac.caltech.edu/docs/pscp_about.html)
- [Stellar Hosts Column Definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_STELLARHOSTS_columns.html)

## Live Исходная Точка

Live host-related relations сейчас выглядят так:

- `lab.host_validated_gaia_physics_result`:
  - `4458` строк;
  - `4456` distinct `source_id`;
  - уже содержит validated Gaia physics для host-контурa;
- `lab.v_nasa_gaia_train_classified`:
  - `3741` строк;
  - `3741` distinct `source_id`;
  - это уже более узкий training-oriented view.

Current review уже показал:

- `3741` строк;
- `3741` unique `source_id`;
- `radius_flame` отсутствует полностью;
- `n_rows_clean_core_ready = 0`.

Дополнительный live-check БД показал:

- `lab.v_nasa_gaia_train_classified` полностью покрывается relation
  `lab.host_validated_gaia_physics_result`;
- в `lab.host_validated_gaia_physics_result` есть еще `715` дополнительных
  distinct `source_id`, которых нет в current train-view;
- обе relation пока живут на `radius_gspphot`, а не на `radius_flame`.

Следствие:

- current host source нельзя считать clean training source для next-wave host model;
- `priority` нельзя честно вернуть в mainline без enrichment;
- silent fallback на `radius_gspphot` противоречит уже зафиксированному clean contract.
- source list для enrichment разумнее строить от более широкого
  `lab.host_validated_gaia_physics_result`, а не только от
  `lab.v_nasa_gaia_train_classified`.

## Главный Принцип

Для host enrichment используем:

- обычный `JOIN` по `source_id`;
- не координатный `crossmatch`;
- не повторное построение host source "с нуля".

Причина:

- у current host source уже есть Gaia `source_id`;
- нужные поля лежат в `gaiadr3.astrophysical_parameters`;
- crossmatch здесь будет лишней и менее чистой операцией.

## Что Именно Тянем Из Gaia

### Обязательные Поля

Минимальный enrichment contract:

- `source_id`
- `radius_flame`

### Очень Желательные Поля

Их тянем в том же шаге, чтобы не ходить в Gaia второй раз:

- `lum_flame`
- `evolstage_flame`

### Что Не Тянем Повторно

Не дублируем поля, которые уже устойчиво есть в current host source:

- `teff_gspphot`
- `logg_gspphot`
- `mh_gspphot`
- `bp_rp`
- `parallax`
- `parallax_over_error`
- `ruwe`
- `phot_g_mean_mag`

Исключение:

- если later audit покажет конфликт или drift, это будет отдельный reconciliation-step,
  а не часть текущего enrichment.

## Design По Relation

### Source List Layer

В `lab` создаем working relation с узким списком `source_id`:

- `lab.nasa_gaia_host_flame_enrichment_source`

Источник для materialization этого relation:

- `lab.host_validated_gaia_physics_result`

Правило:

- берем `DISTINCT source_id`;
- строки с `source_id IS NULL` отбрасываем;
- не ограничиваем source list только текущим `train_classified` view.

Если batching реально нужен для Gaia Archive UI:

- `lab.nasa_gaia_host_flame_enrichment_source_batch_0001`
- `lab.nasa_gaia_host_flame_enrichment_source_batch_0002`
- и далее по необходимости

Назначение:

- staging-only relation для выгрузки в Gaia;
- не reusable clean asset;
- не training relation.

### Raw Gaia Result Layer

В `public` кладем raw результат обратной выгрузки:

- `public.nasa_gaia_host_flame_enrichment_raw`

Колонки:

- `source_id`
- `radius_flame`
- `lum_flame`
- `evolstage_flame`

Назначение:

- хранить результат из Gaia максимально близко к выгрузке;
- не добавлять здесь host semantics;
- не смешивать raw enrichment и downstream decision logic.

### Clean Reusable Enrichment Layer

В `public` создаем clean reusable relation:

- `public.nasa_gaia_host_flame_enrichment_clean`

Правила:

- unique по `source_id`;
- без silent подмены `radius_flame`;
- `source_id` nullable не допускается;
- при дублях выбираем не "любую" строку, а сначала снимаем audit.

Назначение:

- reusable source asset для host retrain;
- source для downstream review и notebook-анализа;
- единый enrichment truth layer для host-wave.

### Downstream Working Layer

В `lab` создаем уже derived working relation:

- `lab.nasa_gaia_host_training_enriched`

Источник:

- `lab.host_validated_gaia_physics_result`
- `public.nasa_gaia_host_flame_enrichment_clean`

JOIN:

- только по `source_id`

Назначение:

- host-physics enriched source;
- review source для повторного host audit;
- upstream для построения уже более узкого clean host training relation;
- upstream для next-wave retrain;
- не обязан совпадать по размеру с current `lab.v_nasa_gaia_train_classified`;
- не обязан быть сразу final train-view.
- 
- Если позже потребуется сохранить старую training-семантику отдельно,
  из него можно построить downstream view:
  - `lab.v_nasa_gaia_train_classified_enriched`
  - или `lab.v_nasa_gaia_host_training_enriched`
- upstream для clean retrain.

## Column Contract Для `lab.nasa_gaia_host_training_enriched`

Минимум:

- `source_id`
- `hostname`
- `spec_class`
- `spec_subclass`
- `evolution_stage`
- `teff_gspphot`
- `logg_gspphot`
- `mh_gspphot`
- `bp_rp`
- `parallax`
- `parallax_over_error`
- `ruwe`
- `phot_g_mean_mag`
- `radius_flame`

Опционально:

- `lum_flame`
- `evolstage_flame`
- `ra_gaia`
- `dec_gaia`
- `dist_arcsec`

Compatibility-only:

- `radius_gspphot`

Правило:

- `radius_gspphot` может остаться как явное historical/compatibility поле;
- canonical radius для next-wave host contract остается только `radius_flame`.

## Materialization Порядок

1. Собираем `lab.nasa_gaia_host_flame_enrichment_source`.
2. При необходимости режем его на batching relation в `lab`.
3. Выгружаем batching relation в Gaia Archive.
4. В Gaia делаем `JOIN` user table -> `gaiadr3.astrophysical_parameters`.
5. Загружаем результат в `public.nasa_gaia_host_flame_enrichment_raw`.
6. Чистим до `public.nasa_gaia_host_flame_enrichment_clean`.
7. Джойним current host source с clean enrichment и строим `lab.nasa_gaia_host_training_enriched`.
8. При необходимости поверх enriched relation строим более узкий
   training-oriented view для confirmed-host task.
9. Переиспользуем уже существующий review-layer для повторного host audit.
10. Только потом проектируем retrain host model.

## Что Проверяем После Enrichment

Минимальный audit:

- `n_rows`
- `n_distinct_source_id`
- `n_rows_with_radius_flame`
- `share_rows_with_radius_flame`
- `n_rows_clean_core_ready`
- class distribution
- stage distribution

Отдельно проверяем:

- не исчезли ли строки из current host source после join;
- сколько enriched rows живет только в broader host physics source, а не в
  current train-view;
- нет ли дублей по `source_id` в clean enrichment relation;
- не появилось ли молчаливое смешение `radius_flame` и `radius_gspphot`.

## Что Сейчас Не Делаем

- не возвращаем `priority` в mainline прямо на этом шаге;
- не retrain-им host model до завершения enriched host relation;
- не используем `pscomppars` как primary positive label truth;
- не делаем hidden compatibility adapter в основном pipeline;
- не тянем сюда ranking logic.

## Следующий Открытый Шаг

После фиксации этого design-doc следующий implementation-step такой:

1. materialize host enrichment source relation;
2. загрузить `source_id` в Gaia;
3. забрать `radius_flame`, `lum_flame`, `evolstage_flame`;
4. собрать clean enriched host relation;
5. переснять host audit;
6. только потом открывать пакет clean host retrain и `HP-05`.

## Live Статус После Materialization

Host enrichment уже materialized локально:

- `lab.nasa_gaia_host_flame_enrichment_source`
- `public.nasa_gaia_host_flame_enrichment_raw`
- `public.nasa_gaia_host_flame_enrichment_clean`
- `lab.nasa_gaia_host_training_enriched`
- `lab.nasa_gaia_host_training_enriched_summary`

Live counts:

- `3741` raw rows
- `3741` clean rows
- `3741` enriched rows
- `3729` rows with `radius_flame`
- `3729` rows with `lum_flame`
- `3504` rows with `evolstage_flame`

Live host review на enriched relation показал:

- `3729` clean-ready rows
- `3729` unique `source_id`
- `6` supported `spec_class`
- `2` supported `evolution_stage`
- `share_rows_clean_core_ready = 1.0`

Это означает:

- blocker по отсутствию canonical radius снят;
- следующий practical step уже не новый Gaia pass, а clean host retrain.

## Критерий Готовности

Документ считается зафиксированным, если:

- понятны relation names и их schema role;
- понятен minimal Gaia enrichment contract;
- понятен порядок materialization;
- зафиксировано, что mainline path идет через enrichment, а не через legacy fallback.
