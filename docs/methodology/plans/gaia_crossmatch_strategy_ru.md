# Стратегия crossmatch Gaia для волны MK

## Цель

Этот документ фиксирует стратегию crossmatch внешнего спектрального источника с `Gaia`.

Нужно определить:

- как практически связать внешний каталог `MK` с `Gaia`;
- что делать в `Gaia Archive`;
- что сохранять в локальную БД;
- как не смешать upload, xmatch и final labeled data.

## Базовый Подход

Новая MK-ветка строится не прямым download "готовой истины", а через связку:

1. внешний spectral source;
2. upload в `Gaia Archive`;
3. positional crossmatch с `gaiadr3.gaia_source`;
4. join с нужными `Gaia`-таблицами;
5. выгрузка результата в локальную БД.

## Почему Такой Подход Нормальный

Это соответствует тому, как `Gaia Archive` работает с пользовательскими таблицами и cross-match операциями:

- пользователь может загрузить свою таблицу;
- таблица попадает в private user space;
- после этого ее можно crossmatch-ить с `gaiadr3.gaia_source`;
- затем результат можно запросить через ADQL.

## Предлагаемый Workflow

### Шаг 1. Скачать И Подготовить External Source

Из внешнего spectral catalog берем минимум:

- объектный идентификатор внешнего source;
- `RA`
- `Dec`
- исходную строку `SpType`
- bibcode или другой идентификатор источника
- magnitude, если есть, как вторичное поле

До crossmatch:

- не нормализуем слишком агрессивно;
- сохраняем сырой `SpType`;
- готовим upload table.

На текущем этапе upload table фиксируем отдельно:

- источник: `lab.gaia_mk_external_filtered`
- contract: [gaia_upload_contract_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/gaia_upload_contract_ru.md)

### Шаг 2. Загрузить Таблицу В Gaia Archive

В `Gaia Archive`:

- логинимся;
- загружаем таблицу в user space;
- используем отдельное осмысленное имя таблицы.

Если работаем через `astroquery.gaia`:

- upload делаем в пользовательскую схему `user_<login_name>`.

### Шаг 3. Подготовить RA/Dec Metadata

Перед built-in crossmatch:

- у uploaded table должны быть корректно помечены `RA` и `Dec` колонки;
- иначе `Gaia Archive` не сможет выполнить geometric cross-match штатным способом.

### Шаг 4. Выполнить Positional Crossmatch

Первый pragmatic вариант:

- crossmatch external source с `gaiadr3.gaia_source`;
- начальный radius берем консервативный и явный;
- сохраняем результат в отдельную user table.

На этом шаге ожидаем получить:

- идентификатор объекта во внешнем source;
- `gaia source_id`
- angular separation

### Шаг 5. Присоединить Нужные Gaia Поля

После xmatch делаем `ADQL join` как минимум с:

- `gaiadr3.gaia_source`
- таблицей физических параметров, если она нужна в том же проходе

На выходе хотим иметь:

- `source_id`
- `ra`, `dec`
- `parallax`
- `parallax_over_error`
- `phot_g_mean_mag`, если доступна
- `ruwe`
- `bp_rp`
- `teff_gspphot`
- `logg_gspphot`
- `radius_flame`
- `mh_gspphot`
- сырой `SpType`
- separation

### Шаг 5a. Зафиксировать Wide Gaia Export Как Raw Landing

После выгрузки из `Gaia Archive` не начинаем работать прямо с CSV как с финальным источником.

Широкий enriched export:

- сохраняем как raw artifact;
- кладем в локальную БД отдельным landing relation;
- не используем как training source и не смешиваем с label normalization.

Текущий live batch:

- файл: `xmatch_bmk_gaia_dr3-result.csv`
- локальный landing relation: `public.raw_landing_table`
- row count: `824038`
- ширина: `160` колонок

Важно:

- по официальному Gaia DR3 datamodel радиус приходит как `radius_flame`, а не `radius_gspphot`;
- текущий live landing batch не содержит FLAME-полей `radius_flame`, `lum_flame`, `evolstage_flame`;
- значит `training_reference` нельзя собирать как fully-compatible training-layer до отдельного enrichment шага.

Правило:

- wide landing relation хранит reproducible снимок результата `Gaia`;
- следующие слои читают уже не его, а узкий канонический слой crossmatch.

### Шаг 6. Сохранить Локально По Слоям

В локальной БД не складываем все в одну таблицу.

Сохраняем по слоям:

- raw import
- crossmatch result
- labeled normalized table
- quality-gated table
- training/scoring views

Текущий порядок после `Gaia` фиксируем так:

1. `public.raw_landing_table`
2. `lab.gaia_mk_external_crossmatch`
3. `lab.gaia_mk_external_labeled`
4. `lab.gaia_mk_training_reference`
5. `lab.gaia_mk_quality_gated`

## Что Делаем С Дубликатами

На этапе xmatch заранее фиксируем простое правило:

- кандидат с наименьшей separation имеет приоритет;
- если остаются конфликтные случаи, они идут в отдельный review path;
- не делаем "тихий" выбор без зафиксированного правила.

Более сложная логика может появиться позже, но не в первой итерации.

В текущем live batch:

- canonical crossmatch relation: `lab.gaia_mk_external_crossmatch`
- batch id: `xmatch_bmk_gaia_dr3__2026_03_26`
- все match-кандидаты сохраняются;
- `xmatch_selected = TRUE` помечает рабочий match с минимальной separation.

## Что Делаем С SpType

Сразу после загрузки не пытаемся использовать `SpType` как готовый final label.

Правильный порядок:

- храним сырой `SpType`;
- парсим его отдельно;
- нормализуем в:
  - `spectral_class`
  - `spectral_subclass`
  - `luminosity_class`
  - `peculiarity_suffix`

## Почему Не Делаем Это Сразу В Одной Giant Query

Потому что это ухудшит читаемость, проверяемость и отладку.

Наша стратегия:

- сначала reproducible upload/xmatch;
- потом reproducible join;
- потом reproducible normalization;
- потом загрузка в локальную БД.

## Что Делаем После Завершенного Gaia-Step

После того как wide export уже скачан и положен в БД:

- не идем повторно в `Gaia` ради label normalization;
- не режем wide landing relation на рабочие training-кусочки;
- не пытаемся парсить labels внутри `ADQL`.

Следующий шаг выполняется локально в БД:

- берем `lab.gaia_mk_external_crossmatch`
- оставляем `xmatch_selected = TRUE`
- join-им с `lab.gaia_mk_external_filtered` по `external_row_id`
- строим `lab.gaia_mk_external_labeled`

Повторные походы в `Gaia` допустимы только как отдельный future-step:

- дополнительная выгрузка новых полей;
- chunk-wise upload/download;
- отдельный landing/import path без переписывания текущего raw batch.

## Инструментальный Вариант Первой Волны

Для первой волны допустим pragmatic route:

- использовать `astroquery.gaia`;
- загрузить таблицу в `Gaia Archive`;
- выполнить штатный crossmatch;
- выгрузить результат;
- положить в локальную БД.

Это соответствует документации и не требует придумывать собственную механику crossmatch с нуля.

## Источники

- [astroquery Gaia: upload table and cross-match](https://astroquery.readthedocs.io/en/latest/gaia/gaia.html)
- [Gaia Archive Use Cases](https://www.cosmos.esa.int/web/gaia-users/archive/use-cases)
- [Gaia Archive: writing queries](https://www.cosmos.esa.int/web/gaia-users/archive/writing-queries)
- [CDS/VizieR B/mk](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/B/mk?format=html&tex=true)
- [PostgreSQL Schemas](https://www.postgresql.org/docs/current/ddl-schemas.html)

## Критерий Готовности Стратегии

Стратегия считается зафиксированной, если:

- понятно, где upload;
- понятно, где crossmatch;
- понятно, где нормализация labels;
- понятно, что именно сохраняем в локальную БД;
- дальше можно писать ingestion plan без архитектурных догадок.
