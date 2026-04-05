# Host Priority Feature Contract

## Цель

Этот документ фиксирует clean feature contract для новой host/priority wave.

Задача документа:

- убрать неявную зависимость от legacy host-контракта;
- отделить host-model features от priority integration features;
- явно определить canonical Gaia fields для новой волны.

## Official Опора

- [Gaia DR3 gaia_source datamodel](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html)
- [Gaia DR3 astrophysical_parameters datamodel](https://gaia.aip.de/metadata/gaiadr3/astrophysical_parameters/)
- [Gaia DR3 astrometric validation](https://gea.esac.esa.int/archive/documentation/GDR3/Catalogue_consolidation/chap_cu9val/sec_cu9val_942/ssec_cu9val_942_astrometry.html)
- [NASA Exoplanet Archive TAP guide](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html)
- [About the PSCompPars Table](https://exoplanetarchive.ipac.caltech.edu/docs/pscp_about.html)
- [Stellar Hosts Column Definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_STELLARHOSTS_columns.html)

## Главный Принцип

Host-model и priority-layer не должны скрыто смешивать:

- старый `radius_gspphot`-контур;
- новый `radius_flame`-контур;
- и произвольные legacy fallbacks.

Если нужен compatibility adapter, он должен быть:

- отдельным;
- временным;
- явно документированным.

## Источники Для Next-Wave Host Contract

### Label / Host Anchor Source

Для semantics-positive labels используем confirmed-host anchor из NASA tables
с логикой, зафиксированной в
[host_target_semantics_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/host_target_semantics_ru.md).

### Gaia Feature Source

Для физических и quality features используем clean Gaia contract новой волны:

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

### Current Local Host Source

Текущий live host source:

- `lab.v_nasa_gaia_train_classified`

по предыдущему audit и notebook-review дает:

- `3741` строк;
- `3741` unique `source_id`;
- `6` поддержанных `spec_class`;
- `2` evolutionary stages;
- subclass coverage очень узкая и в основном по `M`.

Вывод:

- current host source пригоден для first-wave host-likeness;
- он не должен притворяться полным stellar-label source.

## Canonical Radius Rule

Для новой clean host-wave canonical radius считается:

- `radius_flame`

Причина:

- это официальный Gaia DR3 field из `astrophysical_parameters`;
- он уже встроен в наш новый clean contract и quality gate;
- он отделен от legacy `radius_gspphot` compatibility path.

`radius_gspphot` на новой волне:

- не является canonical field;
- может использоваться только в явном temporary adapter;
- не должен silently подмешиваться в clean host features.

## Feature Groups

### 1. Core Host Features

Это минимальный набор для clean host-model:

- `teff_gspphot`
- `logg_gspphot`
- `mh_gspphot`
- `bp_rp`
- `parallax`
- `parallax_over_error`
- `ruwe`
- `radius_flame`

### 2. Observability Features

Это признаки, которые должны жить на стороне priority integration,
а не внутри host target semantics:

- `phot_g_mean_mag`
- `parallax`
- `parallax_over_error`
- sky-position поля, если нужны downstream

### 3. Class Context Features

Это признаки, которые могут использоваться в priority integration
или как auxiliary host context:

- `spec_class`
- `evolution_stage`
- позже при необходимости `final_refinement_label`

Но они не должны размывать саму host target semantics.

## Missing Data Policy

Для clean host contract фиксируем:

- missing core Gaia features не должны silently impute-иться на уровне dataset contract;
- строка без canonical `radius_flame` не считается clean host-training row по умолчанию;
- если нужно сохранить coverage без `radius_flame`, это делается только через отдельный
  compatibility adapter path.

## Что Не Делаем

- не смешиваем `radius_flame` и `radius_gspphot` в одном неявном поле;
- не используем `pscomppars` как единственный source truth для host labels;
- не передаем observability и host semantics как один общий black-box target;
- не тянем в clean contract признаки, которые нужны только ranking-слою, "про запас".

## Связь С Priority Integration

Правильная схема следующей волны:

1. host-model выдает `host_similarity_score`;
2. ranking/priority layer отдельно использует:
   - `class priority`
   - `host_similarity_score`
   - `observability`
3. final decision layer объясняет итоговый priority.

То есть:

- host-model не должен сам считать final priority;
- priority-layer не должен перепридумывать host target semantics.

## First-Wave Decision

Для следующего implementation-пакета фиксируем:

- canonical radius = `radius_flame`;
- `radius_gspphot` остается только compatibility field;
- clean host feature contract строим на official Gaia DR3 fields;
- observability and priority logic держим отдельно от host-model contract.

## Критерий Готовности

Документ считается зафиксированным, если:

- дальше можно строить host-source audit и integration path без semantic drift;
- понятно, какие поля входят в clean host model;
- понятно, какие поля относятся уже к priority, а не к host target.
