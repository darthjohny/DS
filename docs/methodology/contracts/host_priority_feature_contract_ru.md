# Контракт признаков для host-модели и приоритета

## Цель

Этот документ фиксирует набор признаков для новой рабочей версии host-модели и
слоя приоритета.

Задача документа:

- убрать неявную зависимость от старого host-контракта;
- отделить признаки host-модели от признаков слоя приоритета;
- явно определить основной набор полей Gaia для новой версии.

## Документационная опора

- [Gaia DR3 gaia_source datamodel](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html)
- [Gaia DR3 astrophysical_parameters datamodel](https://gaia.aip.de/metadata/gaiadr3/astrophysical_parameters/)
- [Gaia DR3 astrometric validation](https://gea.esac.esa.int/archive/documentation/GDR3/Catalogue_consolidation/chap_cu9val/sec_cu9val_942/ssec_cu9val_942_astrometry.html)
- [NASA Exoplanet Archive TAP guide](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html)
- [About the PSCompPars Table](https://exoplanetarchive.ipac.caltech.edu/docs/pscp_about.html)
- [Stellar Hosts Column Definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_STELLARHOSTS_columns.html)

## Главный принцип

Host-модель и слой приоритета не должны скрыто смешивать:

- старый `radius_gspphot`-контур;
- новый `radius_flame`-контур;
- и произвольные старые обходные варианты.

Если нужен совместимый адаптер, он должен быть:

- отдельным;
- временным;
- явно документированным.

## Источники для новой версии контракта

### Источник меток и опорных host-объектов

Для положительных меток используем подтвержденные host-объекты из таблиц NASA
с логикой, зафиксированной в
[host_target_semantics_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/host_target_semantics_ru.md).

### Источник признаков Gaia

Для физических признаков и сигналов качества используем чистый набор Gaia:

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

### Текущий локальный источник host-объектов

Текущий рабочий источник:

- `lab.v_nasa_gaia_train_classified`

по предыдущему аудиту и разбору в ноутбуках дает:

- `3741` строк;
- `3741` уникальных `source_id`;
- `6` поддержанных `spec_class`;
- `2` evolutionary stages;
- subclass coverage очень узкая и в основном по `M`.

Вывод:

- текущий источник пригоден для первой рабочей версии host-like задачи;
- он не должен притворяться полным источником звездных меток.

## Основное правило по радиусу

Для новой рабочей версии основным радиусом считается:

- `radius_flame`

Причина:

- это официальное поле Gaia DR3 из `astrophysical_parameters`;
- оно уже встроено в новый набор признаков и фильтр качества;
- оно отделено от старого совместимого пути через `radius_gspphot`.

`radius_gspphot` на новой волне:

- не является основным полем;
- может использоваться только в явном временном адаптере;
- не должен неявно подмешиваться в основной набор признаков.

## Группы признаков

### 1. Core Host Features

Это минимальный набор для host-модели:

- `teff_gspphot`
- `logg_gspphot`
- `mh_gspphot`
- `bp_rp`
- `parallax`
- `parallax_over_error`
- `ruwe`
- `radius_flame`

### 2. Observability Features

Это признаки, которые должны жить на стороне слоя приоритета,
а не внутри семантики host-меток:

- `phot_g_mean_mag`
- `parallax`
- `parallax_over_error`
- sky-position поля, если нужны downstream

### 3. Class Context Features

Это признаки, которые могут использоваться в слое приоритета
или как вспомогательный контекст:

- `spec_class`
- `evolution_stage`
- позже при необходимости `final_refinement_label`

Но они не должны размывать саму семантику host-меток.

## Политика пропусков

Для новой версии контракта фиксируем:

- отсутствие основных признаков Gaia не должно неявно заполняться на уровне
  набора данных;
- строка без основного `radius_flame` не считается чистой обучающей строкой по
  умолчанию;
- если нужно сохранить покрытие без `radius_flame`, это делается только через
  отдельный совместимый адаптер.

## Чего не делаем

- не смешиваем `radius_flame` и `radius_gspphot` в одном неявном поле;
- не используем `pscomppars` как единственный источник истины для host-меток;
- не передаем наблюдаемость и host-семантику как одну общую непрозрачную цель;
- не тянем в основной контракт признаки, которые нужны только слою
  ранжирования, «про запас».

## Связь со слоем приоритета

Правильная схема следующей версии:

1. host-model выдает `host_similarity_score`;
2. ranking/priority layer отдельно использует:
   - `class priority`
   - `host_similarity_score`
   - `observability`
3. слой итогового решения объясняет итоговый приоритет.

То есть:

- host-модель не должна сама считать итоговый приоритет;
- слой приоритета не должен переопределять семантику host-меток.

## Решение для первой рабочей версии

Для текущей рабочей версии фиксируем:

- canonical radius = `radius_flame`;
- `radius_gspphot` остается только совместимым полем;
- основной набор признаков строим на официальных полях Gaia DR3;
- наблюдаемость и логика приоритета живут отдельно от host-модели.

## Критерий готовности

Документ считается зафиксированным, если:

- дальше можно строить аудит host-источника и интеграцию без смыслового дрейфа;
- понятно, какие поля входят в host-модель;
- понятно, какие поля относятся уже к слою приоритета.
