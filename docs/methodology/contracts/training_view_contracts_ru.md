# Контракты обучающих представлений для иерархической волны MK/OOD

## Цель

Этот документ фиксирует, какие обучающие представления нужны проекту после
сборки базового слоя данных:

- какие представления нужны проекту;
- какие таблицы являются для них исходными;
- какие поля и целевые признаки используются в каждой задаче;
- какие фильтры применяются в первой рабочей версии.

Документ нужен, чтобы:

- не проектировать загрузчики раньше самих представлений;
- не смешивать coarse, refinement и OOD в один общий источник;
- не придумывать фильтры на этапе кода моделей.

## Общий принцип

Каждое представление должно решать одну задачу.

Поэтому:

- coarse-, refinement- и OOD-слои не смешиваются в одном представлении;
- для каждой задачи задается свой набор полей и фильтров;
- служебные поля отделяются от целевых признаков.

## Опора на Gaia для полей

Смысл признаков опирается на модель данных Gaia DR3:

- `source_id`, `random_index`, `parallax`, `parallax_over_error`, `ruwe`,
  `phot_g_mean_mag`, `bp_rp`, `non_single_star` — из `gaia_source`
- `teff_gspphot`, `logg_gspphot`, `mh_gspphot`,
  `classprob_dsc_combmod_star` — из `astrophysical_parameters`
- `radius_flame`, `lum_flame`, `evolstage_flame` — из `astrophysical_parameters`

Важная оговорка:

- пороги и фильтры проекта не являются официальными порогами Gaia;
- официальная документация задает смысл полей;
- конкретные правила отбора фиксируются в документации проекта.

## Представление 1. `lab.v_gaia_id_coarse_training`

### Назначение

Источник для первого классификатора:

- target = крупный спектральный класс `OBAFGKM`

### Исходная таблица

- `lab.gaia_id_coarse_reference`

### Целевой признак

- `spec_class`

### Служебные поля

Разрешены как служебные:

- `is_evolved`
- `reference_membership_count`
- `has_reference_overlap`
- `random_index`

Они не являются target.

### Набор признаков

Минимальный набор признаков для первой рабочей версии:

- `teff_gspphot`
- `logg_gspphot`
- `mh_gspphot`
- `bp_rp`
- `parallax`
- `parallax_over_error`
- `ruwe`

Допустимые совместимые поля:

- `radius_feature`
  - first-wave view может строить его как
    `COALESCE(radius_flame, radius_gspphot)`
  - основным хранилищем при этом остается `radius_flame`

### Фильтрация первой рабочей версии

В представление допускаем только строки:

- `spec_class IS NOT NULL`
- `teff_gspphot IS NOT NULL`
- `logg_gspphot IS NOT NULL`
- `mh_gspphot IS NOT NULL`
- `bp_rp IS NOT NULL`
- `parallax IS NOT NULL`
- `parallax_over_error IS NOT NULL`
- `ruwe IS NOT NULL`

`radius_flame` не является обязательным для первого coarse-представления.

Причина:

- иначе coarse-срез первой рабочей версии потеряет слишком много строк;
- при этом совместимый радиус допустим только на уровне представления, а не
  исходной таблицы.

## Представление 2. `lab.v_gaia_mk_refinement_training`

### Назначение

Источник для refinement-задачи:

- subclass
- при необходимости связанная luminosity/evolution auxiliary task

### Исходная таблица

- `lab.gaia_mk_quality_gated`

### Целевой признак

Основной целевой признак:

- `spectral_subclass`

Сопутствующие поля меток:

- `spectral_class`
- `luminosity_class`
- `peculiarity_suffix`

### Набор признаков

Минимальный набор признаков для первой рабочей версии:

- `teff_gspphot`
- `logg_gspphot`
- `mh_gspphot`
- `bp_rp`
- `parallax`
- `parallax_over_error`
- `ruwe`
- `radius_flame`

Дополнительно:

- `lum_flame`
- `evolstage_flame`
- `phot_g_mean_mag`

### Фильтрация первой рабочей версии

В представление refinement допускаем только строки:

- `quality_state = 'pass'`
- `ood_state = 'in_domain'`
- `spectral_subclass IS NOT NULL`
- `has_core_features = TRUE`
- `has_flame_features = TRUE`
- где полная комбинация `spec_class + spectral_subclass` встречается не менее
  `15` раз

Правило:

- `candidate_ood`
- `unknown`
- `reject`

не попадают в обычное обучение refinement.

### Чего не делаем

- не тянем `lab.gaia_mk_unknown_review` обратно в refinement training;
- не смешиваем OOD и subclass target в один flat label;
- не строим first-wave refinement view напрямую из `external_labeled`, обходя gate.

Причина для порога `>= 15`:

- первая волна использует `30%` test split и `10-fold CV`;
- порог `10` оказался слишком хрупким для реального разбиения;
- на живом прогоне редкий хвост `O3/O4/O6/O7/O8/O9/K9` давал предупреждения о
  слишком маленьких классах в обучающих фолдах;
- порог `15` стабилизировал первую рабочую версию без лишнего усложнения.

## Представление 3. `lab.v_gaia_id_ood_training`

### Назначение

Источник для отдельной задачи `ID vs OOD`.

### Исходные таблицы

- `lab.gaia_mk_quality_gated`
- `lab.gaia_ood_training_reference`

### Целевой признак

Бинарный целевой признак:

- `domain_target`
  - `id`
  - `ood`

### Часть `ID`

В `ID` часть берем строки из `lab.gaia_mk_quality_gated`, где:

- `quality_state = 'pass'`
- `ood_state = 'in_domain'`
- `has_core_features = TRUE`

`unknown` и `candidate_ood` в обучение `ID` на первой рабочей версии не включаем.

### Часть `OOD`

В `OOD` часть берем строки из `lab.gaia_ood_training_reference`.

Дополнительно сохраняем:

- `ood_group`
- `ood_membership_count`
- `has_multi_ood_membership`

### Набор признаков

Базовый общий набор:

- `teff_gspphot`
- `logg_gspphot`
- `mh_gspphot`
- `bp_rp`
- `parallax`
- `parallax_over_error`
- `ruwe`

Допустимые дополнительные:

- `radius_flame`
- `phot_g_mean_mag`

### Фильтрация первой рабочей версии

Для OOD-view требуем:

- все основные core features не `NULL`

`radius_flame` не делаем обязательным,
потому что OOD-pool по FLAME покрыт плохо.

## Результаты первого базового прогона (`2026-03-28`)

Ниже зафиксированы результаты первого базового прогона по новым представлениям.

### Coarse

- task: `gaia_id_coarse_classification`
- model: `hist_gradient_boosting`
- test accuracy: `0.992926`
- test balanced_accuracy: `0.992379`
- test macro_f1: `0.992573`
- cv mean_accuracy: `0.992724`
- cv mean_balanced_accuracy: `0.991869`
- cv mean_macro_f1: `0.991819`
- artifacts:
  - `artifacts/benchmarks/gaia_id_coarse_classification_2026_03_28_171258_103400`

### Refinement

- task: `gaia_mk_refinement_classification`
- model: `hist_gradient_boosting`
- full subclass count after first-wave filters: `59`
- test accuracy: `0.320336`
- test balanced_accuracy: `0.187861`
- test macro_f1: `0.189683`
- cv mean_accuracy: `0.323909`
- cv mean_balanced_accuracy: `0.194267`
- cv mean_macro_f1: `0.194240`
- artifacts:
  - `artifacts/benchmarks/gaia_mk_refinement_classification_2026_03_28_172145_781713`

Вывод по первой рабочей версии:

- refinement уже работает сквозным образом;
- coarse и `OOD`-слои заметно сильнее;
- subclass-tail требует отдельной второй волны:
  либо regrouping,
  либо hierarchical-per-class refinement,
  либо отсечения редких подклассов по поддержке.

### ID vs OOD

- task: `gaia_id_ood_classification`
- model: `hist_gradient_boosting`
- test accuracy: `0.995734`
- test balanced_accuracy: `0.926215`
- test macro_f1: `0.944521`
- cv mean_accuracy: `0.996025`
- cv mean_balanced_accuracy: `0.926544`
- cv mean_macro_f1: `0.947880`
- artifacts:
  - `artifacts/benchmarks/gaia_id_ood_classification_2026_03_28_172217_384006`

Общий вывод первой волны:

- `coarse` уже показывает качество, достаточное для основного рабочего контура;
- `ID/OOD` уже достаточно стабилен для отдельной ветки с учетом отклонения
  сомнительных объектов;
- `refinement` работоспособен, но пока остается первым приближением, а не
  финальной исследовательской схемой второго слоя.

## Политика загрузчиков

После материализации представлений код не читает их через разрозненные
`SQL`-запросы внутри обучающего контура.

Для первой рабочей версии фиксируем следующее:

- [hierarchical_dataset_contracts.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/contracts/hierarchical_dataset_contracts.py)
  содержит контракты таблиц только для новых представлений;
- каждый loader живет в отдельном файле:
  - [load_gaia_id_coarse_training_dataset.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/datasets/load_gaia_id_coarse_training_dataset.py)
  - [load_gaia_mk_refinement_training_dataset.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/datasets/load_gaia_mk_refinement_training_dataset.py)
  - [load_gaia_id_ood_training_dataset.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/datasets/load_gaia_id_ood_training_dataset.py)
- training-frame normalization вынесен отдельно в
  [hierarchical_training_frame.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/features/hierarchical_training_frame.py)
  и не смешан со старой подготовкой таблиц `V2`.

### Политика coarse-загрузчика

- loader читает `lab.v_gaia_id_coarse_training`;
- stage-label не приходит из view напрямую;
- совместимое отображение строится из `is_evolved -> evolution_stage`;
- `spec_class` остается главным target первого слоя.

### Политика refinement-загрузчика

- loader читает `lab.v_gaia_mk_refinement_training`;
- отображение на стороне кода:
  - `spectral_class -> spec_class`
  - `spectral_subclass -> spec_subclass`
  - `luminosity_class -> evolution_stage`
- mapping `luminosity_class -> evolution_stage` идет через
  [mk_evolution_stage.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/labels/mk_evolution_stage.py),
  а не через скрытый SQL-case в training-code.

### Политика загрузчика для `ID/OOD`

- loader читает `lab.v_gaia_id_ood_training`;
- target остается бинарным: `domain_target`;
- если один `source_id` входит сразу в несколько OOD-group,
  frame-preparation явно схлопывает эти строки до одной training-row;
- это делается не молча:
  - `has_multi_ood_membership` сохраняется;
  - `ood_membership_count` сохраняется;
  - `ood_group_members` сохраняет список групп;
  - `ood_group` переводится в `multi_ood`.

Причина:

- для binary `ID vs OOD` training нам нужна уникальность по `source_id`;
- для бинарного обучения `ID vs OOD` нужна уникальность по `source_id`;
- overlap не должен приводить к leakage между train/test;
- traceability при этом не теряется.

## Что делаем после фиксации контракта

Только после фиксации этих контрактов представлений:

1. materialize/create views в БД;
2. сделать DB/profile audit по каждому view;
3. потом писать loaders;
4. только потом запускать первые baseline train/benchmark runs.

## Связанные документы

- [db_layer_closure_plan_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/db_layer_closure_plan_ru.md)
- [quality_ood_contract_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/quality_ood_contract_ru.md)
- [hierarchical_ood_strategy_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/hierarchical_ood_strategy_ru.md)

Внутренний микро-план maturation-этапа ведется вне публичного контура
репозитория.
