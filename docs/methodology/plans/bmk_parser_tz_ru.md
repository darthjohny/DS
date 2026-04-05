# B/mk Parser Technical Specification

## Цель

Этот документ фиксирует ТЗ на отдельный parser для внешнего каталога `B/mk`.

Задача parser-а:

- корректно прочитать `mktypes.dat` по официальному `ReadMe`;
- не гадать по структуре колонок вручную, если формат уже описан;
- привести результат к нашему staging-контракту;
- сохранить staging CSV для загрузки в локальную БД;
- не смешивать parsing, Gaia crossmatch и training logic в один модуль.

## Почему Делаем Отдельный Parser

`B/mk` нужен нам не как "еще один csv", а как внешний источник `MK`-разметки:

- `spectral_class`
- `spectral_subclass`
- `luminosity_class`

Локальной БД сейчас хватает для physics/quality слоя, но не хватает массовых `MK`-меток. Поэтому parser нужен как первый обязательный слой новой волны.

## Официальные Источники

- `B/mk ReadMe` и `Byte-by-byte Description`:
  [CDS/VizieR B/mk](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/B/mk?format=html&tex=true)
- чтение CDS-таблиц через `ReadMe`:
  [Astropy Table I/O](https://docs.astropy.org/en/stable/io/unified_table.html)
- CDS reader:
  [Astropy ascii.cds](https://docs.astropy.org/en/stable/api/astropy.io.ascii.Cds.html)
- CSV-экспорт:
  [Python csv](https://docs.python.org/3/library/csv.html)
- работа с путями:
  [Python pathlib](https://docs.python.org/3/library/pathlib.html)
- стиль Python:
  [PEP 8](https://peps.python.org/pep-0008/)
- типизация:
  [Python typing](https://docs.python.org/3/library/typing.html)
- загрузка в PostgreSQL:
  [PostgreSQL COPY](https://www.postgresql.org/docs/current/sql-copy.html)

## Главный Инженерный Принцип

Parser не должен:

- выполнять training;
- выполнять ranking;
- ходить в `Gaia` за crossmatch;
- писать напрямую в старые relation;
- смешивать parsing и business logic.

Parser должен:

- читать внешний источник;
- нормализовать staging-колонки;
- писать reproducible CSV;
- отделять первично пригодные строки от явного брака;
- давать понятный audit-summary.

## Границы Ответственности Parser-а

### Делает

- читает `ReadMe` и `mktypes.dat`;
- извлекает нужные поля;
- приводит координаты к `ra_deg` и `dec_deg`;
- сохраняет сырой `raw_sptype`;
- формирует staging CSV;
- формирует первично очищенный `filtered`-слой;
- формирует `rejected`-слой с причиной отбраковки;
- формирует статистику по качеству импорта.

### Не Делает

- не парсит `raw_sptype` в финальные `spectral_class/subclass/luminosity_class`;
- не выполняет Gaia crossmatch;
- не пишет в training views;
- не принимает решений `quality_state` / `ood_state`.

## Входные Артефакты

Parser принимает:

- `ReadMe`
- `mktypes.dat`

Parser не должен зависеть от ручного копирования колонок из веб-страницы.

## Выходные Артефакты

Parser сохраняет:

- `lab.gaia_mk_external_raw`
- `lab.gaia_mk_external_filtered`
- `lab.gaia_mk_external_rejected`

## Будущая Структура Файлов

Когда перейдем к реализации, раскладка должна быть такой:

- `src/exohost/ingestion/bmk/contracts.py`
- `src/exohost/ingestion/bmk/reader.py`
- `src/exohost/ingestion/bmk/normalization.py`
- `src/exohost/ingestion/bmk/filtering.py`
- `src/exohost/ingestion/bmk/export.py`
- `tests/unit/test_bmk_catalog_parser.py`

## Raw Contract Для Parser-а

Parser должен формировать CSV со следующими колонками:

- `external_row_id`
- `external_catalog_name`
- `external_object_id`
- `ra_deg`
- `dec_deg`
- `raw_sptype`
- `raw_magnitude`
- `raw_source_bibcode`
- `raw_notes`

## Primary Filter Contract

После raw-слоя parser должен отдельно строить:

- `filtered`
- `rejected`

### Filtered-колонки

- `external_row_id`
- `external_catalog_name`
- `external_object_id`
- `ra_deg`
- `dec_deg`
- `raw_sptype`
- `raw_magnitude`
- `raw_source_bibcode`
- `raw_notes`
- `spectral_prefix`
- `spectral_class`
- `spectral_subclass`
- `luminosity_class`
- `parse_status`
- `parse_note`
- `has_supported_prefix`
- `has_coordinates`
- `has_raw_sptype`
- `ready_for_gaia_crossmatch`

### Rejected-колонки

- `external_row_id`
- `external_catalog_name`
- `external_object_id`
- `ra_deg`
- `dec_deg`
- `raw_sptype`
- `raw_magnitude`
- `raw_source_bibcode`
- `raw_notes`
- `spectral_prefix`
- `reject_reason`

## Правила Нормализации

- `external_row_id`
  - детерминированный идентификатор строки внутри staging-файла;
- `external_catalog_name`
  - фиксированное значение `bmk`;
- `external_object_id`
  - внешний object id из каталога;
- `ra_deg`, `dec_deg`
  - только в десятичных градусах;
- `raw_sptype`
  - хранится в максимально близком к исходнику виде, без "умного" исправления;
- пустые строки
  - превращаются в `NULL`-эквивалент на уровне CSV/БД;
- пробелы по краям
  - обрезаются;
- parser не должен silently invent значения для отсутствующих полей.

## Audit Summary После Parsing

Parser после выполнения обязан печатать или сохранять summary:

- сколько строк считано всего;
- сколько строк имеют валидные координаты;
- сколько строк имеют непустой `raw_sptype`;
- сколько строк имеют `raw_sptype`, начинающийся с `O/B/A/F/G/K/M`;
- сколько строк попало в итоговый staging CSV.

После primary filter parser обязан дополнительно считать:

- сколько строк попало в `filtered`;
- сколько строк попало в `rejected`;
- сколько строк отброшено из-за `missing_coordinates`;
- сколько строк отброшено из-за `missing_raw_sptype`;
- сколько строк отброшено из-за `unsupported_spectral_prefix`.

## Имена Таблиц В БД

Для новой волны фиксируем такие relation names:

- `lab.gaia_mk_external_raw`
- `lab.gaia_mk_external_filtered`
- `lab.gaia_mk_external_rejected`
- `lab.gaia_mk_external_crossmatch`
- `lab.gaia_mk_external_labeled`
- `lab.gaia_mk_training_reference`
- `lab.gaia_mk_quality_gated`
- `lab.v_gaia_mk_training_dwarfs`
- `lab.v_gaia_mk_training_evolved`
- `lab.v_gaia_mk_router_training`
- `lab.v_gaia_mk_host_training`
- `lab.v_gaia_mk_candidate_scoring`

Этот parser работает только на первые три слоя:

- `lab.gaia_mk_external_raw`
- `lab.gaia_mk_external_filtered`
- `lab.gaia_mk_external_rejected`

## SQL Для Raw Таблицы

```sql
CREATE TABLE IF NOT EXISTS lab.gaia_mk_external_raw (
    external_row_id BIGINT PRIMARY KEY,
    external_catalog_name TEXT NOT NULL,
    external_object_id TEXT,
    ra_deg DOUBLE PRECISION NOT NULL,
    dec_deg DOUBLE PRECISION NOT NULL,
    raw_sptype TEXT NOT NULL,
    raw_magnitude DOUBLE PRECISION,
    raw_source_bibcode TEXT,
    raw_notes TEXT,
    imported_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

## SQL Для Filtered Таблицы

```sql
CREATE TABLE IF NOT EXISTS lab.gaia_mk_external_filtered (
    external_row_id BIGINT PRIMARY KEY,
    external_catalog_name TEXT NOT NULL,
    external_object_id TEXT,
    ra_deg DOUBLE PRECISION NOT NULL,
    dec_deg DOUBLE PRECISION NOT NULL,
    raw_sptype TEXT NOT NULL,
    raw_magnitude DOUBLE PRECISION,
    raw_source_bibcode TEXT,
    raw_notes TEXT,
    spectral_prefix TEXT NOT NULL,
    spectral_class TEXT NOT NULL,
    spectral_subclass INTEGER,
    luminosity_class TEXT,
    parse_status TEXT NOT NULL,
    parse_note TEXT,
    has_supported_prefix BOOLEAN NOT NULL,
    has_coordinates BOOLEAN NOT NULL,
    has_raw_sptype BOOLEAN NOT NULL,
    ready_for_gaia_crossmatch BOOLEAN NOT NULL,
    filtered_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

## SQL Для Rejected Таблицы

```sql
CREATE TABLE IF NOT EXISTS lab.gaia_mk_external_rejected (
    external_row_id BIGINT PRIMARY KEY,
    external_catalog_name TEXT,
    external_object_id TEXT,
    ra_deg DOUBLE PRECISION,
    dec_deg DOUBLE PRECISION,
    raw_sptype TEXT,
    raw_magnitude DOUBLE PRECISION,
    raw_source_bibcode TEXT,
    raw_notes TEXT,
    spectral_prefix TEXT,
    reject_reason TEXT NOT NULL,
    rejected_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

## SQL Для Загрузки CSV В Raw Таблицу

Локально в PostgreSQL загружаем staging CSV через `COPY` или `\\copy`.

Предпочтительный вариант:

```sql
\copy lab.gaia_mk_external_raw (
    external_row_id,
    external_catalog_name,
    external_object_id,
    ra_deg,
    dec_deg,
    raw_sptype,
    raw_magnitude,
    raw_source_bibcode,
    raw_notes
)
FROM '/absolute/path/to/bmk_external_raw.csv'
WITH (
    FORMAT csv,
    HEADER true,
    NULL ''
);
```

## Что Будет Crossmatch-иться Потом

После загрузки `lab.gaia_mk_external_raw` и построения `filtered/rejected` parser уже закончил свою работу.

Следующий отдельный этап делает crossmatch с:

- `gaiadr3.gaia_source`
- `gaiadr3.astrophysical_parameters`

Это не часть parser-а.

В `Gaia crossmatch` идет только:

- `lab.gaia_mk_external_filtered`

## Будущий ADQL Для Crossmatch И Enrichment

Этот запрос хранится здесь как контракт следующего шага, а не как логика parser-а.

```sql
SELECT
    u.external_row_id,
    u.external_object_id,
    u.ra_deg AS external_ra_deg,
    u.dec_deg AS external_dec_deg,
    u.raw_sptype,
    gs.source_id,
    DISTANCE(
        POINT('ICRS', u.ra_deg, u.dec_deg),
        POINT('ICRS', gs.ra, gs.dec)
    ) * 3600.0 AS xmatch_separation_arcsec,
    gs.ra,
    gs.dec,
    gs.parallax,
    gs.parallax_over_error,
    gs.ruwe,
    gs.phot_g_mean_mag,
    gs.bp_rp,
    gs.random_index,
    ap.teff_gspphot,
    ap.logg_gspphot,
    ap.radius_flame,
    ap.mh_gspphot
FROM user_<login>.mkraw_upload AS u
JOIN gaiadr3.gaia_source AS gs
    ON 1 = CONTAINS(
        POINT('ICRS', gs.ra, gs.dec),
        CIRCLE('ICRS', u.ra_deg, u.dec_deg, 5.0 / 3600.0)
    )
LEFT JOIN gaiadr3.astrophysical_parameters AS ap
    ON ap.source_id = gs.source_id
WHERE u.raw_sptype IS NOT NULL;
```

### Почему Радиус 5 Arcsec

Для первого debug-прохода используем `5 arcsec`, а не `1 arcsec`, потому что исторические координаты внешнего спектрального каталога могут иметь менее аккуратную привязку, чем современные Gaia-координаты.

Сужать радиус будем только после проверки реального распределения `xmatch_separation_arcsec`.

## Куда Пишется Результат Crossmatch

После выгрузки из `Gaia Archive` результат должен попасть в:

- `lab.gaia_mk_external_crossmatch`

Дальше отдельно строятся:

- `lab.gaia_mk_external_labeled`
- `lab.gaia_mk_training_reference`
- `lab.gaia_mk_quality_gated`

## Требования К Будущей Реализации Parser-а

- `PEP 8`
- типизация по умолчанию
- маленькие чистые функции
- одна ответственность на модуль
- комментарии только через `#` и только на русском
- без ручного fixed-width parsing, если `astropy` уже умеет читать таблицу по `ReadMe`
- без прямой записи в БД из parser-а первой версии

## Требования К Тестам Parser-а

Когда начнем кодить parser, минимально тестируем:

- чтение sample `mktypes.dat` через `ReadMe`;
- корректное получение `ra_deg/dec_deg`;
- сохранение `raw_sptype` без потери;
- обрезку пустых строк;
- формирование staging CSV с ожидаемыми колонками;
- стабильность `external_row_id`.

## QA Для Реализации Parser-а

После каждого блока реализации:

- сверка с официальной документацией `Astropy`, `Python csv`, `PostgreSQL COPY`
- `ruff`
- `mypy`
- `pyright`
- `pytest`

После завершения модуля:

- повторная сверка с `B/mk ReadMe`;
- повторная проверка, что parser не делает лишнюю бизнес-логику.

## Критерий Готовности ТЗ

ТЗ считается готовым, если:

- понятно, что parser читает;
- понятно, что parser отдает;
- заранее известна raw relation;
- заранее известен следующий crossmatch step;
- реализацию можно начинать без архитектурной импровизации.
