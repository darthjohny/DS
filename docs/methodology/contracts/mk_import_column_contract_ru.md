# MK Import Column Contract

## Цель

Этот документ фиксирует минимально достаточный набор колонок для новой MK-волны.

Задача:

- не тащить лишний шум;
- не забыть критически важные поля;
- заранее разделить:
  - external raw columns;
  - Gaia physical columns;
  - quality columns;
  - normalized label columns.

## External Raw Columns

Обязательные:

- `external_row_id`
- `external_catalog_name`
- `external_object_id`
- `ra_deg`
- `dec_deg`
- `raw_sptype`

Желательные:

- `raw_magnitude`
- `raw_source_bibcode`
- `raw_notes`

Зачем:

- без координат нет reproducible crossmatch;
- без сырого `SpType` нет нормального label parsing;
- без внешнего object id audit становится хуже.

## Gaia Physical Columns

Обязательные:

- `source_id`
- `teff_gspphot`
- `logg_gspphot`
- `radius_flame`
- `mh_gspphot`
- `bp_rp`

Зачем:

- это ядро физической feature-схемы новой волны.
- если downstream legacy loader требует `radius_gspphot`, этот alias строится отдельно
  поверх canonical `radius_flame`.

## Gaia Quality Columns

Обязательные:

- `ruwe`
- `parallax`
- `parallax_over_error`

Зачем:

- это база раннего quality/OOD gate.

## Gaia Observability Columns

Обязательные при наличии:

- `phot_g_mean_mag`

Желательные позже:

- дополнительные photometric/service fields, если они нужны для практического observability layer.

Зачем:

- без brightness-aware поля observability будет опираться на distance/quality, но не на яркость.

## Normalized Label Columns

После parsing должны появиться:

- `spectral_class`
- `spectral_subclass`
- `luminosity_class`
- `peculiarity_suffix`

Дополнительно:

- `luminosity_group`, если она вводится как укрупненный рабочий target

## Technical Traceability Columns

Рекомендуемые:

- `xmatch_separation_arcsec`
- `xmatch_rank`
- `xmatch_selected`
- `label_parse_status`
- `label_parse_notes`
- timestamps по слоям

Зачем:

- без них сложно воспроизводить и объяснять, как объект попал в новую training-ветку.

## Что Не Тянем Без Явной Причины

- декоративные каталожные поля, которые не участвуют в labels, quality или ranking;
- случайные текстовые поля "на всякий случай";
- любые derived labels, если не описано, как они получены.

## Минимально Достаточный Набор Для Первого Прохода

Если идти минимальным рабочим путем, то первый проход должен обязательно дать:

- `source_id`
- `ra_deg`
- `dec_deg`
- `raw_sptype`
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

И желательно:

- `phot_g_mean_mag`

## Критерий Готовности Контракта

Контракт считается зафиксированным, если:

- каждая колонка имеет понятную роль;
- никакая обязательная колонка не "всплывает потом";
- ingestion plan может прямо ссылаться на этот список.
