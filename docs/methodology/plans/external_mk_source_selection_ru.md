# External MK Source Selection

## Цель

Этот документ фиксирует выбор внешнего источника для полноценной MK-волны.

Нужно определить:

- какой источник берем для `spectral_subclass` и `luminosity_class`;
- почему именно его;
- почему не берем в качестве основного source только `Gaia` или только `NASA`.

## Что Требуется От Источника

Минимум, который нам нужен:

- массовые спектральные классификации;
- наличие subtype-уровня, а не только coarse class;
- возможность связать объекты с `Gaia`;
- воспроизводимость и документированность источника.

Желательно:

- координаты;
- исходная ссылка на публикацию;
- возможность нормализовать MK-схему до составных полей.

## Рассмотренные Варианты

### 1. Gaia DR3 Как Единственный Label-Source

Статус:

- не подходит как единственный источник для полной MK-волны.

Почему:

- `Gaia DR3` дает физические параметры и coarse astrophysical labels;
- это полезно как feature-backbone;
- но не дает массовый готовый каталог `G0..G9`, `K0..K9`, `V/III/IV` в том виде, который нужен нам как основной training label-contract.

Роль в проекте:

- backbone для физики и астрометрии;
- не основной внешний source для полной MK-разметки.

## 2. NASA Exoplanet Archive

Статус:

- полезен, но не подходит как основной all-sky MK-source.

Почему:

- `NASA Exoplanet Archive` содержит `st_spectype`;
- это полезно для host-side sanity-check и review;
- но покрывает host stars, а не общую reference population.

Роль в проекте:

- дополнительный host-side источник;
- не основной source для общей subclass/luminosity разметки.

## 3. CDS / VizieR B/mk

Статус:

- выбран как основной внешний spectral source для новой волны.

Источник:

- `B/mk — Catalogue of Stellar Spectral Classifications (Skiff)`

Почему:

- это специализированный каталог спектральных классификаций;
- он содержит литературу по спектральным типам, а не только производные broadband labels;
- дает:
  - координаты;
  - строку спектрального типа;
  - источник публикации;
  - комментарии;
- объем достаточный для серьезной выборки.

## Почему Выбираем Именно B/mk

- Он ближе всего к нашей задаче по смыслу.
- Он лучше соответствует MK-волне, чем coarse Gaia labels.
- Его можно crossmatch-ить с `Gaia` и затем использовать `Gaia` как физический backbone.
- Он позволяет построить составной label-contract:
  - `spectral_class`
  - `spectral_subclass`
  - `luminosity_class`
  - при необходимости `peculiarity_suffix`

## Ограничения B/mk

Выбираем его осознанно, а не как магический идеальный каталог.

Надо помнить:

- это compilation catalog из литературы;
- `SpType` нужно будет нормализовать;
- в каталоге встречаются не только идеально чистые MK-cases;
- часть записей может требовать фильтрации;
- magnitude там дана как ориентир, а не как основной photometric truth.

## Решение Для Проекта

На новой волне:

- `B/mk` берем как основной внешний source для MK-разметки;
- `Gaia` используем как физический и астрометрический backbone;
- `NASA Exoplanet Archive` оставляем как дополнительный host-side источник, а не как общий subclass-source.

## Что Это Значит Практически

Порядок такой:

1. выгружаем внешний `B/mk`;
2. делаем crossmatch с `Gaia`;
3. нормализуем `SpType` в составные поля;
4. грузим результат в новые relation names в локальной БД;
5. только после этого начинаем новую MK training-ветку.

## Источники

- [Gaia DR3 astrophysical_parameters](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_astrophysical_parameter_tables/ssec_dm_astrophysical_parameters.html)
- [Gaia DR3 DSC](https://gea.esac.esa.int/archive/documentation/GDR3/Data_analysis/chap_cu8par/sec_cu8par_apsis/ssec_cu8par_apsis_dsc.html)
- [NASA Exoplanet Archive: Stellar Hosts columns](https://exoplanetarchive.ipac.caltech.edu/docs/API_STELLARHOSTS_columns.html)
- [CDS/VizieR B/mk](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/B/mk?format=html&tex=true)

## Критерий Готовности Решения

Решение считается зафиксированным, если:

- основной внешний source выбран явно;
- роль `Gaia`, `NASA` и `B/mk` разделена;
- дальнейший ingestion plan уже не спорит о том, откуда брать labels.
