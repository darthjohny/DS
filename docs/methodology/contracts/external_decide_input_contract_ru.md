# Внешний Входной Контракт Для Боевого `decide`

Дата фиксации: `2026-04-06`

Связанные документы:

- [project_db_contour_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/project_db_contour_ru.md)
- [gaia_upload_contract_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/gaia_upload_contract_ru.md)
- [training_view_contracts_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/training_view_contracts_ru.md)
- [baseline_run_registry_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/baseline_run_registry_ru.md)
- [Gaia Archive: writing queries](https://www.cosmos.esa.int/web/gaia-users/archive/writing-queries)
- [Gaia DR3 `gaia_source`](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html)
- [Gaia DR3 `astrophysical_parameters`](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_astrophysical_parameter_tables/ssec_dm_astrophysical_parameters.html)

## Зачем Нужен Этот Документ

Этот документ отвечает на практический вопрос:

> какие данные должен дать внешний пользователь, чтобы текущий боевой pipeline
> их принял и обработал.

Документ нужен для двух сценариев:

- внешний пользователь хочет подать свою таблицу в проект;
- проверяющий хочет сам выгрузить данные из Gaia Archive и прогнать их без
  работы с нашей локальной БД.

## Важная Граница Сценариев

Здесь нужно сразу честно разделить два режима.

### Режим 1. Полностью сопоставимый с active baseline

Для этого нужен проектный рабочий слой:

- `lab.gaia_mk_quality_gated`

Только этот путь:

- проходит через ту же quality-логику;
- согласован с current active baseline;
- дает результат, который можно честно сравнивать с нашими notebook и
  run-review.

### Режим 2. Внешний быстрый прогон по Gaia CSV

Он тоже возможен, но это уже не полная копия active baseline.

В этом режиме пользователь:

- сам выгружает таблицу из Gaia Archive;
- локально подает ее в `decide`;
- получает рабочий inference-result;
- но должен понимать, что quality-слой в таком сценарии будет упрощен, если он
  не воспроизведет `quality_gate` из проекта.

То есть быстрый CSV-прогон полезен для проверки и знакомства с системой, но не
является идеальной заменой project DB route.

## Что Требует Текущий Боевой `decide`

Для текущего active bundle loader требует:

- `source_id`
- `quality_state`
- набор физических признаков текущего decision-bundle

Фактический feature union current active bundle:

- `teff_gspphot`
- `logg_gspphot`
- `mh_gspphot`
- `bp_rp`
- `parallax`
- `parallax_over_error`
- `ruwe`
- `phot_g_mean_mag`
- `radius_feature`
- `radius_flame`
- `lum_flame`
- `evolstage_flame`
- `radius_gspphot`

Важно:

- `radius_feature` допускает compatibility-алиас из `radius_flame` или
  `radius_gspphot`;
- `radius_gspphot` тоже может быть восстановлен через `radius_flame`;
- но `radius_flame`, `lum_flame` и `evolstage_flame` для current active bundle
  нужно считать практически обязательными.

## Минимальный CSV-Контракт Для Внешнего Прогона

Если пользователь хочет подать таблицу напрямую в `decide --input-csv`, то
минимальный практический набор колонок такой:

- `source_id`
- `quality_state`
- `teff_gspphot`
- `logg_gspphot`
- `mh_gspphot`
- `bp_rp`
- `parallax`
- `parallax_over_error`
- `ruwe`
- `phot_g_mean_mag`
- `radius_flame`
- `lum_flame`
- `evolstage_flame`

Рекомендуется дополнительно дать:

- `radius_gspphot`
- `ra`
- `dec`
- `random_index`

Зачем:

- `radius_gspphot` удобен для compatibility и review;
- `ra`, `dec` полезны для traceability и внешней сверки;
- `random_index` удобен для воспроизводимого порядка строк.

## Что Делать С `quality_state`

Это самый важный практический вопрос.

`quality_state` не приходит из Gaia Archive напрямую.

В боевом project route он появляется в:

- `lab.gaia_mk_quality_gated`

Если внешний пользователь не воспроизводит этот слой, у него есть два пути.

### Честный путь

Использовать project DB route и работать через:

- `lab.gaia_mk_quality_gated`

### Быстрый экспериментальный путь

Добавить в CSV колонку:

- `quality_state = 'pass'`

для всех строк.

Но это нужно понимать правильно:

- такой запуск обходит project quality gate;
- он удобен для технической проверки inference-контура;
- он не должен сравниваться с нашим active baseline как полностью
  эквивалентный run.

## Какой ADQL Запрос Можно Сделать В Gaia Archive

Ниже приведен практический запрос, который позволяет выгрузить минимальный
сырой Gaia DR3 слой для текущего decision-контра.

```sql
SELECT
    gs.source_id,
    gs.ra,
    gs.dec,
    gs.random_index,
    gs.phot_g_mean_mag,
    gs.bp_rp,
    gs.parallax,
    gs.parallax_over_error,
    gs.ruwe,
    ap.teff_gspphot,
    ap.logg_gspphot,
    ap.mh_gspphot,
    ap.radius_gspphot,
    ap.radius_flame,
    ap.lum_flame,
    ap.evolstage_flame
FROM gaiadr3.gaia_source AS gs
LEFT JOIN gaiadr3.astrophysical_parameters AS ap
    ON ap.source_id = gs.source_id
WHERE gs.source_id IS NOT NULL
  AND gs.parallax IS NOT NULL
  AND gs.parallax_over_error IS NOT NULL
  AND gs.ruwe IS NOT NULL
  AND gs.phot_g_mean_mag IS NOT NULL
  AND gs.bp_rp IS NOT NULL
  AND ap.teff_gspphot IS NOT NULL
  AND ap.logg_gspphot IS NOT NULL
  AND ap.mh_gspphot IS NOT NULL
  AND ap.radius_flame IS NOT NULL
  AND ap.lum_flame IS NOT NULL
  AND ap.evolstage_flame IS NOT NULL
ORDER BY gs.random_index ASC, gs.source_id ASC
```

Этот запрос:

- не копирует весь наш project route;
- но дает сырой CSV, который уже близок к current active decision contract;
- после добавления `quality_state` может быть подан в `decide --input-csv`.

## Как Превратить Gaia CSV В `decide`-Совместимый CSV

Минимальный практический путь:

1. выгрузить CSV этим ADQL-запросом;
2. добавить колонку `quality_state`;
3. заполнить ее значением `pass`;
4. сохранить CSV;
5. подать его в `decide`.

Если у пользователя есть и `radius_gspphot`, и `radius_flame`, этого уже
достаточно для current compatibility policy по радиусу.

## Что Важно Сказать Проверяющему

Корректная формулировка:

> Для полностью сопоставимого с проектом прогона следует использовать рабочую
> relation `lab.gaia_mk_quality_gated`, так как именно она содержит результат
> project quality gate. Однако для внешней технической проверки inference-слоя
> можно самостоятельно выгрузить совместимый Gaia DR3 CSV по ADQL, добавить
> колонку `quality_state` и подать файл напрямую в `decide`.

Некорректная формулировка:

- “любой Gaia CSV автоматически эквивалентен нашему active baseline”;
- “quality_state можно игнорировать совсем”;
- “сырой ADQL-экспорт полностью повторяет весь project pipeline”.

## Короткий Вывод

Для внешнего пользователя есть простой и честный ответ:

- если нужен полностью проектный маршрут, используем
  `lab.gaia_mk_quality_gated`;
- если нужен внешний быстрый запуск, выгружаем Gaia DR3 по ADQL, добавляем
  `quality_state` и подаем CSV напрямую;
- второй путь годится для проверки inference-контура, но не заменяет полностью
  project DB route.
