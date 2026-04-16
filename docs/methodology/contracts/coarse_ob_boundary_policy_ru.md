# Политика границы `O/B` на coarse-уровне

## Цель

После исправления парсера не смешивать для неоднозначных меток `OB...`:

- узкий горячий слой, действительно похожий на `O`
- и широкий пограничный пул `O/B`

в один общий локальный класс `O`.

## Почему это нужно

Второй обзор происхождения меток показал:

- широкий проблемный пул `O` почти целиком состоял из неоднозначных меток
  `OB...`;
- после sync этот пул распался на:
  - узкий явно `O`-подобный хвост
  - пограничный пул `O/B`

Текущие live числа:

- `B = 7112`
- `OB = 1163`
- `O = 25`

То есть теперь проблема уже не в том, что «модель не умеет `O`», а в том, что
семантику пограничного горячего пула `O/B` надо держать отдельно.

## Документационная опора

Опираемся на семантику `Gaia DR3`:

- `spectraltype_esphs`
- `flags_esphs`

из `gaiadr3.astrophysical_parameters`.

Основная идея:

- `ESP-HS` — модуль Gaia для горячих звезд `O/B/A`;
- `spectraltype_esphs` дает спектральную метку горячей звезды на стороне Gaia;
- `flags_esphs = 999` не считаем надежным сигналом.

Источники:

- [Gaia DR3 Apsis overview](https://gea.esac.esa.int/archive/documentation/GDR3/Data_analysis/chap_cu8par/sec_cu8par_intro/ssec_cu8par_intro_apsis.html)
- [Gaia DR3 astrophysical_parameters datamodel](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_astrophysical_parameter_tables/ssec_dm_astrophysical_parameters.html)

## Первая рабочая версия политики

### Надежный слой, похожий на `O`

Источник:

- `public.gaia_ob_hot_provenance_audit_clean`

Критерий:

- `spectral_class IN ('O', 'OB')`
- `esphs_class_letter = 'O'`
- `flags_esphs != 999`

Почему без hard `teff_esphs` cut:

- live review показал, что `teff_esphs >= 25000` сожмет subset до `6` строк;
- это уже слишком агрессивно для first operational policy;
- `spectraltype_esphs='O'` itself already работает как primary hot-star
  classifier signal.

### Пограничный слой `O/B`

Источник:

- тот же `public.gaia_ob_hot_provenance_audit_clean`

Критерий:

- `spectral_class IN ('O', 'OB')`
- и строка не попадает в `secure O-like`

На практике сюда попадают:

- все неоднозначные метки `OB`;
- явные локальные `O`, которые Gaia `ESP-HS` не подтверждает как надежные `O`.

## Материализованные таблицы

Рабочие таблицы в `lab`:

- `lab.gaia_ob_secure_o_like_subset`
- `lab.gaia_ob_boundary_subset`
- `lab.gaia_ob_boundary_policy_summary`
- `lab.gaia_ob_boundary_review`
- `lab.gaia_ob_boundary_review_summary`

Это именно рабочий слой политики, а не итоговый публичный набор данных.

На текущем этапе `lab.gaia_ob_boundary_review` трактуется как явный слой
разбора:

- не форсируем автоматическое разделение `O/B`;
- не считаем этот пул чистой истиной для автоматической оценки моделей;
- не пытаемся “угадать” `O` vs `B` без spectroscopy-grade external support.
- практически выносим этот пул в отдельный контур разбора до появления более
  сильной внешней спектроскопической опоры.

## Результаты базового прогона

После materialization `lab.gaia_ob_secure_o_like_subset` и
`lab.gaia_ob_boundary_subset`:

- `secure O-like`: `25`
- `O/B boundary`: `1163`

Разбивка:

- local `O`: `11 secure`, `14 boundary`
- local `OB`: `14 secure`, `1149 boundary`

Materialized summary:

- `secure_o_like / O`
  - `n_rows = 11`
  - `median_teff_esphs ≈ 33647.50 K`
- `secure_o_like / OB`
  - `n_rows = 14`
  - `median_teff_esphs ≈ 30657.71 K`
- `ob_boundary / O`
  - `n_rows = 14`
  - `median_teff_esphs ≈ 16496.00 K`
- `ob_boundary / OB`
  - `n_rows = 1149`
  - `median_teff_esphs ≈ 20000.25 K`

Причины попадания в пограничный слой:

- `ambiguous_ob_label`: `1149`
- `explicit_o_not_confirmed_by_esphs`: `14`

После отдельной materialization `lab.gaia_ob_boundary_review`:

- `review` rows: `1163`
- breakdown:
  - local `O = 14`
  - local `OB = 1149`

Review summary:

- `O / explicit_o_not_confirmed_by_esphs / B`
  - `n_rows = 12`
  - `median_teff_esphs ≈ 16496.00 K`
- `O / explicit_o_not_confirmed_by_esphs / U`
  - `n_rows = 2`
- `OB / ambiguous_ob_label / B`
  - `n_rows = 1115`
  - `median_teff_esphs ≈ 20000.25 K`

Это подтверждает, что пограничный пул на текущем этапе лучше трактовать как
слой разбора, а не как шумный обучающий набор для принудительного разделения
`O/B`.

Проверка на более жесткий temperature cut:

- внутри уже materialized `secure O-like` only `6` строк имеют
  `teff_esphs >= 25000`

Это подтверждает, что в первой рабочей версии `teff_esphs` лучше оставлять
сигналом для разбора, а не превращать в основной жесткий фильтр.

## Что делаем дальше

Для первой рабочей версии уже принято решение:

1. пограничный слой `OB` остается в явном слое разбора
2. automatic `O/B` split для boundary-пула не делаем
3. надежный `O`-подобный слой используем только как узкий горячий хвост для
   дальнейшей научной проверки

Следующий шаг уже не в парсере и не в общей политике для `OB`:

1. отдельно проверить явный `O`-хвост на текущем поведении coarse-модели
2. решить, нужен ли для этого хвоста отдельный повторный запуск обучения,
   изменение весов классов или согласование с внешней спектроскопией
