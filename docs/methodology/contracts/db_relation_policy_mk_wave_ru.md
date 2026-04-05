# DB Relation Policy For MK Wave

## Цель

Этот документ фиксирует naming policy и роли новых relation для MK-волны.

Главная задача:

- не трогать существующий слой БД;
- не удалять полезные текущие таблицы и view;
- заранее определить имена и ответственность новых relation.

## Главный Принцип

Существующий слой БД остается как есть.

Новая волна:

- не заменяет старые relation молча;
- не переиспользует старые имена для новой схемы;
- не разрушает существующие source.

Мы действуем так:

- добавляем новые таблицы и view рядом;
- переключаем loaders и training/scoring только на новые relation names;
- при необходимости сравниваем старые и новые контуры параллельно.

## Правило Именования

Используем единый префикс `gaia_mk_` для новой ветки.

Никаких имен:

- `new`
- `tmp`
- `final`
- `latest`
- `v2_new`

Имена должны сразу показывать роль relation.

## Предлагаемый Набор Relation

### Raw Layer

- `lab.gaia_mk_external_raw`

Назначение:

- хранит внешний spectral source в виде, максимально близком к исходному импорту.

### Crossmatch Layer

- `lab.gaia_mk_external_crossmatch`

Назначение:

- хранит связь между внешним spectral source и объектами `Gaia`.

### Labeled Layer

- `lab.gaia_mk_external_labeled`

Назначение:

- хранит нормализованные MK-метки после crossmatch и базовой очистки.

### Quality-Gated Layer

- `lab.gaia_mk_quality_gated`

Назначение:

- хранит объекты после раннего quality/OOD gate.

### Training Reference Layer

- `lab.gaia_mk_training_reference`

Назначение:

- единый нормализованный источник для дальнейших training views.
- это первый слой, где допускается жесткая политика уникальности по `source_id`.

### Training Views

- `lab.v_gaia_mk_training_dwarfs`
- `lab.v_gaia_mk_training_evolved`
- `lab.v_gaia_mk_router_training`
- `lab.v_gaia_mk_host_training`
- `lab.v_gaia_id_coarse_training`
- `lab.v_gaia_mk_refinement_training`
- `lab.v_gaia_id_ood_training`

Назначение:

- выделяют training scenarios без смешения raw/import логики в model loaders.

Current first-wave split:

- `lab.v_gaia_id_coarse_training`
  - coarse `OBAFGKM`
- `lab.v_gaia_mk_refinement_training`
  - subclass/refinement
- `lab.v_gaia_id_ood_training`
  - `ID vs OOD`

### Candidate Scoring View

- `lab.v_gaia_mk_candidate_scoring`

Назначение:

- источник для реального candidate scoring и `prioritize`.

## Правило Разделения По Слоям

### Schema Layout `public` vs `lab`

`public` используем для:

- raw landing relation;
- clean reusable source assets;
- relation, которые можно переиспользовать в нескольких downstream-сценариях
  без model-specific gate logic.

Примеры:

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

`lab` используем для:

- normalized working layers;
- training/reference/gate/decision relation;
- audit summaries;
- training/scoring views.

Примеры:

- `lab.gaia_mk_external_*`
- `lab.gaia_id_coarse_reference`
- `lab.gaia_mk_training_reference`
- `lab.gaia_ood_training_reference`
- `lab.gaia_mk_quality_gated`
- `lab.gaia_mk_unknown_review`
- `lab.*summary`
- `lab.v_*`

Правило:

- reusable source asset не должен жить только в `lab`;
- training/gate/decision relation не должны уходить в `public`.

### Raw

- хранит то, что пришло извне;
- не нормализует labels слишком рано;
- не превращается в training source напрямую.

### Crossmatch

- отвечает только за связку external source и Gaia;
- не становится финальным training source без labeled step.

### Labeled

- хранит нормализованные label-поля:
  - `spectral_class`
  - `spectral_subclass`
  - `luminosity_class`
  - `peculiarity_suffix`, если есть

### Quality-Gated

- хранит качество и решение раннего gate;
- не подменяет training logic;
- отделяет clean objects от uncertain/OOD.

### Views

- формируют сценарии:
  - training
  - scoring
  - prioritize
- не должны дублировать raw storage semantics.

## Обязательные Поля Для Новой Волны

Минимум для новой ветки:

- `source_id`
- `spectral_class`
- `spectral_subclass`
- `luminosity_class`
- `teff_gspphot`
- `logg_gspphot`
- `radius_flame`
- `mh_gspphot`
- `bp_rp`
- `parallax`
- `parallax_over_error`
- `ruwe`

Если доступно:

- `phot_g_mean_mag`
- `peculiarity_suffix`

Правило:

- legacy alias `radius_gspphot` допускается только в downstream compatibility view;
- canonical reference layer хранит официальный Gaia DR3 field `radius_flame`.

## Что Не Делаем

- не пишем новые данные в старые training views;
- не смешиваем raw/import/training/scoring в одну relation;
- не вводим новые names по ситуации без фиксации в docs.

## Критерий Готовности Policy

Policy считается зафиксированной, если:

- ingestion знает, куда писать каждый слой;
- loaders знают, откуда читать;
- новая волна не ломает старую структуру БД;
- раскладка `public` vs `lab` не оставляет двусмысленности, где лежит raw,
  где clean source asset, а где training/gate layer.
