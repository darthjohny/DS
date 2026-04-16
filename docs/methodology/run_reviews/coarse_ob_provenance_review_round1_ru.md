# Первый обзор происхождения меток для `coarse O/B`

## Цель

Проверить, не связано ли рабочее схлопывание `O -> B` с ошибкой в локальных
данных, crossmatch или семантике меток, а не с coarse-моделью как таковой.

## Источники

- локальный источник для рабочего аудита:
  - `lab.gaia_ob_hot_provenance_audit_source`
- Gaia результат raw:
  - `public.gaia_ob_hot_provenance_audit_raw`
- Gaia результат clean:
  - `public.gaia_ob_hot_provenance_audit_clean`
- сводки:
  - `lab.gaia_ob_hot_provenance_audit_summary`
  - `lab.gaia_ob_hot_provenance_crosswalk_summary`

Использованные сигналы Gaia:

- `spectraltype_esphs`
- `flags_esphs`
- `teff_esphs`
- `logg_esphs`
- `ag_esphs`
- `azero_esphs`
- `ebpminrp_esphs`
- membership marker через `gaiadr3.gold_sample_oba_stars`

Опора:

- [Gaia DR3 `astrophysical_parameters`](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_astrophysical_parameter_tables/ssec_dm_astrophysical_parameters.html)
- [Gaia DR3 Apsis overview](https://gea.esac.esa.int/archive/documentation/GDR3/Data_analysis/chap_cu8par/sec_cu8par_intro/ssec_cu8par_intro_apsis.html)
- [Gaia DR3 docs index](https://gea.esac.esa.int/archive/documentation/GDR3/)

## Пул Проверки

Проверялся рабочий горячий проходной пул `O/B`:

- `spectral_class IN ('O', 'B')`
- `quality_state = 'pass'`
- `teff_gspphot >= 10000`

Размер:

- всего строк: `8300`
- `B`: `7112`
- `O`: `1188`

## Ключевые Результаты

### 0. Основной upstream-source локального `O` сам по себе неоднозначен

Все `1188` рабочих локальных `O` приходят из одного upstream catalog:

- `external_catalog_name = 'bmk'`

И почти весь этот пул состоит не из явных `O4/O7/O9`, а из сырых пограничных обозначений:

- `1157 / 1188` (`97.39%`) имеют `raw_sptype`, начинающийся с `OB...`
- только:
  - `16 / 1188` имеют вид `O<digit>...`
  - `7 / 1188` имеют вид `O(...)...`
  - `5 / 1188` равны bare `O`

Аудит parser/ingestion тоже указывает на это:

- `1172 / 1188` (`98.65%`) имеют
  - `label_parse_status = 'partial'`
  - `label_parse_notes = 'missing_integer_subclass'`

Это уже само по себе означает, что широкий локальный пул `O` в рабочем контуре
не является чистым явным источником `O`.

### 1. Локальный `B` в целом согласован с семантикой Gaia для горячих звезд

- `6873 / 7112` локальных `B` (`96.64%`) Gaia `ESP-HS` тоже относит к `B`
- медианы:
  - `teff_gspphot ≈ 12311.66 K`
  - `teff_esphs ≈ 13778.13 K`

### 2. Локальный рабочий `O` почти целиком Gaia относит к `B`

- `1127 / 1188` локальных `O` (`94.87%`) имеют `spectraltype_esphs = 'B'`
- только `27 / 1188` (`2.27%`) имеют `spectraltype_esphs = 'O'`
- медианы:
  - `teff_gspphot ≈ 15613.89 K`
  - `teff_esphs ≈ 20000.25 K`

### 3. Даже температура Gaia для горячих звезд у локального `O` обычно не похожа на `O`

Для локального `O`:

- `teff_esphs >= 30000 K`: `7 / 1188` (`0.59%`)
- `teff_esphs >= 25000 K`: `76 / 1188` (`6.40%`)
- `teff_esphs >= 20000 K`: `543 / 1188` (`45.71%`)

Это плохо согласуется с гипотезой, что рабочие локальные `O` в массе являются
классическими горячими `O`-звездами.

### 4. Membership в `gold_sample_oba_stars` не решает границу `O/B`

- local `B`: `6862 / 7112` (`96.48%`) inside gold sample
- local `O`: `1110 / 1188` (`93.43%`) inside gold sample

Этот gold sample хорошо подтверждает, что пул действительно похож на OBA, но сам по
себе почти не различает `O` и `B`.

## Интерпретация

На текущем этапе наиболее правдоподобное объяснение такое:

- это не похоже на ошибку соединения в базе;
- это не похоже на случайное дублирование `B` как `O`;
- это очень похоже на проблему разбора и происхождения меток:
  - ambiguous `OB...` labels upstream сейчас сваливаются в локальный `spectral_class='O'`;
- текущая семантика меток рабочего локального `O` плохо согласована с
  семантикой Gaia `ESP-HS` для горячих звезд;
- coarse-модель, которая в рабочем контуре схлопывает `O` в `B`, вероятно реагирует на
  реальную физическую близость этого пула к `B`, а не просто “ломается”.

Иначе говоря:

- основная проблема сейчас сидит не в устройстве coarse-контура;
- основная проблема сейчас сидит в происхождении и семантике меток рабочего `O`.

## Практический Вывод

Прямой повторный запуск обучения coarse-модели вслепую на текущем этапе не выглядит первым шагом.

Сначала логичнее:

1. исправить parser/policy для ambiguous `OB...` labels, чтобы они не считались
   чистым `O` автоматически;
2. отдельно выделить:
   - `secure O-like subset`
   - `O/B boundary pool`;
3. только потом решать, нужен ли повторный запуск обучения;
4. рассмотреть политику, где `O/B` на рабочей горячей границе трактуется как
   неопределенное пограничное семейство, а не как жесткий плоский класс.

## Следующий Шаг

- открыть отдельный обзор происхождения `O` и источников меток
- посмотреть происхождение этих рабочих `O` в локальных upstream relation
- только после этого решать:
  - согласование источников
  - relabeling
  - пограничное семейство
  - или узкий повторный запуск обучения
