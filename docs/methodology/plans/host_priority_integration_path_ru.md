# Host Priority Integration Path

## Цель

Этот документ фиксирует engineering decision для clean host/priority integration
после `HP-01 ... HP-03`.

Задача документа:

- выбрать правильный путь подключения host-layer обратно в final pipeline;
- не смешать legacy adapter и новый clean contract;
- зафиксировать blocker и следующий implementation path.

## На Что Опираемся

- [host_target_semantics_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/host_target_semantics_ru.md)
- [host_priority_feature_contract_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/host_priority_feature_contract_ru.md)
- [quality_gate_host_priority_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/quality_gate_host_priority_tz_ru.md)
- [Gaia DR3 astrophysical_parameters datamodel](https://gaia.aip.de/metadata/gaiadr3/astrophysical_parameters/)
- [NASA Exoplanet Archive TAP guide](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html)
- [About the PSCompPars Table](https://exoplanetarchive.ipac.caltech.edu/docs/pscp_about.html)
- [The Gaia-Kepler-TESS-Host Stellar Properties Catalog](https://arxiv.org/abs/2301.11338)

## Live Факт После HP-03

Current host source review показал:

- `n_rows = 3741`
- `n_unique_source_id = 3741`
- `n_supported_classes = 6`
- `n_supported_stages = 2`
- `n_core_features = 8`
- `n_core_features_present = 7`
- `has_canonical_radius_column = False`
- `n_rows_with_canonical_radius = 0`
- `n_rows_clean_core_ready = 0`

Главный blocker:

- в текущем host source полностью отсутствует `radius_flame`.

Это означает:

- current host source нельзя честно считать clean training source для next-wave host model;
- current priority integration нельзя возвращать в mainline без нового data step;
- silent fallback на `radius_gspphot` противоречит зафиксированному clean contract.

## Рассмотренные Варианты

### Вариант A. Temporary Compatibility Adapter

Суть:

- сохранить текущий host source;
- silently or semi-explicitly подменить canonical radius через `radius_gspphot`;
- быстро вернуть host/priority в pipeline.

Плюсы:

- быстрее по времени;
- не требует нового enrichment step.

Минусы:

- ломает clean contract;
- возвращает нас в legacy-style semantics drift;
- делает notebooks и explainability двусмысленными;
- повышает риск, что `host_similarity_score` снова будет жить на другом физическом контракте,
  чем `coarse/refinement/final decision`.

### Вариант B. Clean Retrain After Host Enrichment

Суть:

- сначала обогатить current host source официальными Gaia fields, включая `radius_flame`;
- затем пересобрать host training dataset на новом contract;
- потом retrain host model;
- потом вернуть priority integration в mainline.

Плюсы:

- соответствует зафиксированному clean contract;
- убирает semantic drift между host-layer и остальным pipeline;
- делает notebooks и final decision explainability честными;
- лучше соответствует scientific reproducibility.

Минусы:

- требует еще одного data engineering шага;
- возвращает priority позже.

## Decision

Для проекта выбираем:

- `clean retrain after host enrichment`

И не выбираем:

- `temporary compatibility adapter` как mainline path.

Compatibility adapter допускается только как:

- отдельный временный экспериментальный слой;
- не основной проектный путь;
- не default behavior.

## Что Значит Host Enrichment

Следующий data step для host-layer должен дать relation, где по каждому host source object
есть минимум:

- `source_id`
- `spec_class`
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

И уже на этой relation:

- пересобираем host-vs-field training source;
- retrain host model;
- возвращаем `host_similarity_score` в final priority integration.

## Следующий Практический Порядок

1. Спроектировать clean enriched host relation.
2. Материализовать его из existing host source + official Gaia enrichment.
3. Пересобрать host training review layer уже на enriched source.
4. Только потом retrain host model.
5. Только потом вернуть `priority` в final decision pipeline.

## Live Статус После Host Enrichment

Путь уже подтвержден live materialization:

- `lab.nasa_gaia_host_training_enriched` собран;
- `3741` enriched rows materialized;
- `3729` rows уже имеют canonical `radius_flame`;
- live host review на enriched relation дает `3729` clean-ready rows.

Это означает:

- structural blocker по canonical radius снят;
- следующий открытый шаг уже не новый enrichment design, а clean host retrain.

## Что Не Делаем

- не возвращаем priority в mainline прямо сейчас;
- не называем adapter-path clean solution;
- не смешиваем `radius_flame` и `radius_gspphot` в одном поле без явного названия;
- не даем notebooks делать вид, что host-layer уже живет на новом contract.

## Критерий Готовности

`HP-04` считается зафиксированным, если:

- выбран один clean integration path;
- blocker documented явно;
- дальнейшая реализация не строится на неявном fallback.
