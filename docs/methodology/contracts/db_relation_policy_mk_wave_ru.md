# Политика именования таблиц для MK-волны

## Цель

Этот документ фиксирует правила именования и роли новых таблиц для MK-волны.

Главная задача:

- не трогать существующий слой БД;
- не удалять полезные текущие таблицы и view;
- заранее определить имена и ответственность новых таблиц.

## Главный принцип

Существующий слой БД остается как есть.

Новая волна:

- не заменяет старые таблицы молча;
- не переиспользует старые имена для новой схемы;
- не разрушает существующие source.

Мы действуем так:

- добавляем новые таблицы и view рядом;
- переключаем загрузчики и обучение/оценку только на новые имена таблиц;
- при необходимости сравниваем старые и новые контуры параллельно.

## Правило именования

Используем единый префикс `gaia_mk_` для новой ветки.

Никаких имен:

- `new`
- `tmp`
- `final`
- `latest`
- `v2_new`

Имена должны сразу показывать роль таблицы.

## Предлагаемый набор таблиц

### Слой raw-данных

- `lab.gaia_mk_external_raw`

Назначение:

- хранит внешний spectral source в виде, максимально близком к исходному импорту.

### Слой кроссматча

- `lab.gaia_mk_external_crossmatch`

Назначение:

- хранит связь между внешним spectral source и объектами `Gaia`.

### Слой меток

- `lab.gaia_mk_external_labeled`

Назначение:

- хранит нормализованные MK-метки после crossmatch и базовой очистки.

### Слой после проверки качества

- `lab.gaia_mk_quality_gated`

Назначение:

- хранит объекты после раннего quality/OOD gate.

### Опорный обучающий слой

- `lab.gaia_mk_training_reference`

Назначение:

- единый нормализованный источник для дальнейших training views.
- это первый слой, где допускается жесткая политика уникальности по `source_id`.

### Обучающие представления

- `lab.v_gaia_mk_training_dwarfs`
- `lab.v_gaia_mk_training_evolved`
- `lab.v_gaia_mk_router_training`
- `lab.v_gaia_mk_host_training`
- `lab.v_gaia_id_coarse_training`
- `lab.v_gaia_mk_refinement_training`
- `lab.v_gaia_id_ood_training`

Назначение:

- выделяют training scenarios без смешения raw/import логики в model loaders.

Разделение для первой рабочей версии:

- `lab.v_gaia_id_coarse_training`
  - coarse `OBAFGKM`
- `lab.v_gaia_mk_refinement_training`
  - subclass/refinement
- `lab.v_gaia_id_ood_training`
  - `ID vs OOD`

### Представление для оценки кандидатов

- `lab.v_gaia_mk_candidate_scoring`

Назначение:

- источник для реального candidate scoring и `prioritize`.

## Правило разделения по слоям

### Разделение схем `public` и `lab`

`public` используем для:

- широких таблиц первичной загрузки;
- чистых повторно используемых наборов данных;
- таблиц, которые можно переиспользовать в нескольких следующих сценариях
  без логики, завязанной на конкретную модель.

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

- нормализованных рабочих слоев;
- таблиц обучения, опоры, фильтрации и решения;
- сводок аудита;
- представлений для обучения и оценки.

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

- повторно используемый набор данных не должен жить только в `lab`;
- таблицы обучения, фильтрации и решения не должны уходить в `public`.

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
- не должны дублировать смысл хранения raw-данных.

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

- совместимый псевдоним `radius_gspphot` допускается только в отдельном
  совместимом представлении;
- основной опорный слой хранит официальное поле Gaia DR3 `radius_flame`.

## Что Не Делаем

- не пишем новые данные в старые training views;
- не смешиваем raw-данные, импорт, обучение и оценку в одной таблице;
- не вводим новые имена по ситуации без фиксации в документации.

## Критерий готовности политики

Policy считается зафиксированной, если:

- ingestion знает, куда писать каждый слой;
- loaders знают, откуда читать;
- новая волна не ломает старую структуру БД;
- разделение `public` и `lab` не оставляет двусмысленности, где лежит raw,
  где чистый повторно используемый набор данных, а где слой обучения и
  фильтрации.
