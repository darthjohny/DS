# Coarse O/B Provenance Review Round 1

## Цель

Проверить, не связано ли downstream-схлопывание `O -> B` с ошибкой в локальных
данных, crossmatch или label semantics, а не с coarse-моделью как таковой.

## Источники

- локальный downstream audit source:
  - `lab.gaia_ob_hot_provenance_audit_source`
- Gaia result raw:
  - `public.gaia_ob_hot_provenance_audit_raw`
- Gaia result clean:
  - `public.gaia_ob_hot_provenance_audit_clean`
- сводки:
  - `lab.gaia_ob_hot_provenance_audit_summary`
  - `lab.gaia_ob_hot_provenance_crosswalk_summary`

Использованные official Gaia сигналы:

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

Проверялся downstream hot `O/B pass`-пул:

- `spectral_class IN ('O', 'B')`
- `quality_state = 'pass'`
- `teff_gspphot >= 10000`

Размер:

- всего строк: `8300`
- `B`: `7112`
- `O`: `1188`

## Ключевые Результаты

### 0. Основной upstream-source local `O` сам по себе ambiguous

Все `1188` downstream local `O` приходят из одного upstream catalog:

- `external_catalog_name = 'bmk'`

И почти весь этот пул состоит не из явных `O4/O7/O9`, а из сырых boundary labels:

- `1157 / 1188` (`97.39%`) имеют `raw_sptype`, начинающийся с `OB...`
- только:
  - `16 / 1188` имеют вид `O<digit>...`
  - `7 / 1188` имеют вид `O(...)...`
  - `5 / 1188` равны bare `O`

Parser/ingestion audit тоже указывает на это:

- `1172 / 1188` (`98.65%`) имеют
  - `label_parse_status = 'partial'`
  - `label_parse_notes = 'missing_integer_subclass'`

Это уже само по себе означает, что широкий local `O` pool в downstream не является
clean explicit `O` source.

### 1. Local `B` в целом согласован с Gaia hot-star semantics

- `6873 / 7112` local `B` (`96.64%`) Gaia `ESP-HS` тоже относит к `B`
- median:
  - `teff_gspphot ≈ 12311.66 K`
  - `teff_esphs ≈ 13778.13 K`

### 2. Local downstream `O` почти целиком Gaia относит к `B`

- `1127 / 1188` local `O` (`94.87%`) имеют `spectraltype_esphs = 'B'`
- только `27 / 1188` (`2.27%`) имеют `spectraltype_esphs = 'O'`
- median:
  - `teff_gspphot ≈ 15613.89 K`
  - `teff_esphs ≈ 20000.25 K`

### 3. Даже Gaia hot-star temperature у local `O` обычно не O-like

Для local `O`:

- `teff_esphs >= 30000 K`: `7 / 1188` (`0.59%`)
- `teff_esphs >= 25000 K`: `76 / 1188` (`6.40%`)
- `teff_esphs >= 20000 K`: `543 / 1188` (`45.71%`)

Это плохо согласуется с гипотезой, что downstream local `O` в массе являются
классическими hot `O` stars.

### 4. Membership в `gold_sample_oba_stars` не решает границу `O/B`

- local `B`: `6862 / 7112` (`96.48%`) inside gold sample
- local `O`: `1110 / 1188` (`93.43%`) inside gold sample

Этот gold sample хорошо подтверждает, что пул действительно OBA-like, но сам по
себе почти не различает `O` и `B`.

## Интерпретация

На текущем этапе наиболее правдоподобное объяснение такое:

- это не похоже на DB-join bug;
- это не похоже на случайное дублирование `B` как `O`;
- это очень похоже на parser/provenance issue:
  - ambiguous `OB...` labels upstream сейчас сваливаются в local `spectral_class='O'`;
- current downstream local `O` label semantics плохо согласованы с Gaia
  hot-star `ESP-HS` semantics;
- coarse-модель, которая downstream схлопывает `O` в `B`, вероятно реагирует на
  реальную физическую близость этого пула к `B`, а не просто “ломается”.

Иначе говоря:

- broad issue сейчас сидит не в coarse plumbing;
- broad issue сейчас сидит в provenance / label semantics downstream `O`.

## Практический Вывод

Прямой blind retrain coarse-модели на текущем этапе не выглядит первым шагом.

Сначала логичнее:

1. исправить parser/policy для ambiguous `OB...` labels, чтобы они не считались
   чистым `O` автоматически;
2. отдельно выделить:
   - `secure O-like subset`
   - `O/B boundary pool`;
3. только потом решать, нужен ли retrain;
4. рассмотреть policy, где `O/B` на downstream hot boundary трактуется как
   uncertain boundary family, а не как жесткий flat class.

## Следующий Шаг

- открыть `O-source provenance / label-source review`
- посмотреть origin этих downstream `O` в локальных upstream relation
- только после этого решать:
  - source alignment
  - relabeling
  - boundary family
  - или narrow retrain
