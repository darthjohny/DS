# Training View Contracts For Hierarchical MK/OOD Wave

## Цель

Этот документ фиксирует следующий переход после закрытия DB foundation:

- какие training views нужны проекту;
- какие relation являются их upstream-source;
- какие поля и target используются в каждой задаче;
- какие фильтры считаются first-wave policy.

Документ нужен, чтобы:

- не проектировать loaders раньше самих training view;
- не смешивать coarse, refinement и OOD в один общий source;
- не выдумывать фильтры на этапе model-кода.

## Инженерный Инвариант

Для всех следующих code-side шагов действует один и тот же стандарт:

- `1 файл = 1 ответственность`
- без монолитных loader/view-модулей
- `PEP 8`
- явная типизация
- простое решение раньше сложного
- без лишних зависимостей
- после каждого маленького куска:
  - micro-QA
  - `ruff`
  - точечный `mypy/pyright`
  - целевые тесты
- после закрытия микро-ТЗ:
  - scoped big-QA только по написанному слою

Это правило касается не только кода, но и SQL/view materialization:

- один view = одна задача;
- один DB-модуль = один тип view;
- никакого giant SQL-файла "на все training scenarios".

## Official Gaia Опора Для Полей

Смысл признаков опирается на official Gaia DR3 datamodel:

- `source_id`, `random_index`, `parallax`, `parallax_over_error`, `ruwe`,
  `phot_g_mean_mag`, `bp_rp`, `non_single_star` — из `gaia_source`
- `teff_gspphot`, `logg_gspphot`, `mh_gspphot`,
  `classprob_dsc_combmod_star` — из `astrophysical_parameters`
- `radius_flame`, `lum_flame`, `evolstage_flame` — из `astrophysical_parameters`

Важная оговорка:

- project thresholds и training filters не считаются official Gaia cut;
- official docs задают смысл полей;
- конкретные training-slice правила фиксируются отдельно как first-wave policy.

## View 1. `lab.v_gaia_id_coarse_training`

### Назначение

Источник для первого классификатора:

- target = крупный спектральный класс `OBAFGKM`

### Upstream Source

- `lab.gaia_id_coarse_reference`

### Target

- `spec_class`

### Auxiliary Поля

Разрешены как служебные:

- `is_evolved`
- `reference_membership_count`
- `has_reference_overlap`
- `random_index`

Они не являются target.

### Feature Contract

Минимальный first-wave feature set:

- `teff_gspphot`
- `logg_gspphot`
- `mh_gspphot`
- `bp_rp`
- `parallax`
- `parallax_over_error`
- `ruwe`

Допустимые compatibility-поля:

- `radius_feature`
  - first-wave view может строить его как
    `COALESCE(radius_flame, radius_gspphot)`
  - canonical storage при этом остается за `radius_flame`

### First-Wave Filter Policy

В view допускаем только строки:

- `spec_class IS NOT NULL`
- `teff_gspphot IS NOT NULL`
- `logg_gspphot IS NOT NULL`
- `mh_gspphot IS NOT NULL`
- `bp_rp IS NOT NULL`
- `parallax IS NOT NULL`
- `parallax_over_error IS NOT NULL`
- `ruwe IS NOT NULL`

`radius_flame` не является обязательным для первого coarse view.

Причина:

- иначе first-wave coarse slice потеряет слишком много строк;
- при этом compatibility radius допустим только на уровне view, а не source relation.

## View 2. `lab.v_gaia_mk_refinement_training`

### Назначение

Источник для refinement-задачи:

- subclass
- при необходимости связанная luminosity/evolution auxiliary task

### Upstream Source

- `lab.gaia_mk_quality_gated`

### Target

Основной target:

- `spectral_subclass`

Сопутствующие label-поля:

- `spectral_class`
- `luminosity_class`
- `peculiarity_suffix`

### Feature Contract

Минимальный first-wave feature set:

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

### First-Wave Filter Policy

В refinement view допускаем только строки:

- `quality_state = 'pass'`
- `ood_state = 'in_domain'`
- `spectral_subclass IS NOT NULL`
- `has_core_features = TRUE`
- `has_flame_features = TRUE`
- support по full subclass `spec_class + spectral_subclass >= 15`

Правило:

- `candidate_ood`
- `unknown`
- `reject`

не попадают в normal refinement training.

### Что Не Делаем

- не тянем `lab.gaia_mk_unknown_review` обратно в refinement training;
- не смешиваем OOD и subclass target в один flat label;
- не строим first-wave refinement view напрямую из `external_labeled`, обходя gate.

Причина для порога `>= 15`:

- первая волна использует `30%` test split и `10-fold CV`;
- support `10` в полном срезе оказался слишком хрупким для real split;
- на живом прогоне редкий хвост `O3/O4/O6/O7/O8/O9/K9` давал warning-ы
  про слишком маленькие классы в train-folds;
- cutoff `15` стабилизировал first-wave baseline без переусложнения policy.

## View 3. `lab.v_gaia_id_ood_training`

### Назначение

Источник для отдельной задачи `ID vs OOD`.

### Upstream Source

- `lab.gaia_mk_quality_gated`
- `lab.gaia_ood_training_reference`

### Target

Бинарный target:

- `domain_target`
  - `id`
  - `ood`

### ID Side

В `ID` часть берем строки из `lab.gaia_mk_quality_gated`, где:

- `quality_state = 'pass'`
- `ood_state = 'in_domain'`
- `has_core_features = TRUE`

`unknown` и `candidate_ood` в ID-train первую волну не включаем.

### OOD Side

В `OOD` часть берем строки из `lab.gaia_ood_training_reference`.

Дополнительно сохраняем:

- `ood_group`
- `ood_membership_count`
- `has_multi_ood_membership`

### Feature Contract

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

### First-Wave Filter Policy

Для OOD-view требуем:

- все основные core features не `NULL`

`radius_flame` не делаем обязательным,
потому что OOD-pool по FLAME покрыт плохо.

## First Baseline Results (`2026-03-28`)

Ниже зафиксированы первые live baseline-run по новым view.

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

First-wave вывод:

- refinement numerically уже работает end-to-end;
- coarse и OOD baseline заметно сильнее;
- subclass-tail требует отдельной второй волны:
  либо regrouping,
  либо hierarchical-per-class refinement,
  либо support-aware pruning.

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

- `coarse` baseline уже production-like по качеству;
- `ID/OOD` baseline уже strong enough для отдельной ветки reject-aware logic;
- `refinement` baseline работоспособен, но пока это именно baseline, а не финальная
  исследовательская схема второго слоя.

## Code-Side Loader Policy

После materialized views код не читает их ad hoc SQL-строками из training-runner.

Первая code-side волна фиксируется так:

- [hierarchical_dataset_contracts.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/contracts/hierarchical_dataset_contracts.py)
  содержит relation-contracts только для новых view;
- каждый loader живет в отдельном файле:
  - [load_gaia_id_coarse_training_dataset.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/datasets/load_gaia_id_coarse_training_dataset.py)
  - [load_gaia_mk_refinement_training_dataset.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/datasets/load_gaia_mk_refinement_training_dataset.py)
  - [load_gaia_id_ood_training_dataset.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/datasets/load_gaia_id_ood_training_dataset.py)
- training-frame normalization вынесен отдельно в
  [hierarchical_training_frame.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/features/hierarchical_training_frame.py)
  и не смешан с legacy V2 frame-preparation.

### Coarse Loader Policy

- loader читает `lab.v_gaia_id_coarse_training`;
- stage-label не приходит из view напрямую;
- code-side compatibility mapping строится из `is_evolved -> evolution_stage`;
- `spec_class` остается главным target первого слоя.

### Refinement Loader Policy

- loader читает `lab.v_gaia_mk_refinement_training`;
- code-side mapping:
  - `spectral_class -> spec_class`
  - `spectral_subclass -> spec_subclass`
  - `luminosity_class -> evolution_stage`
- mapping `luminosity_class -> evolution_stage` идет через
  [mk_evolution_stage.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/labels/mk_evolution_stage.py),
  а не через скрытый SQL-case в training-code.

### ID/OOD Loader Policy

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
- overlap не должен приводить к leakage между train/test;
- traceability при этом не теряется.

## Что Делаем После Фиксации Contracts

Только после фиксации этих view-contract:

1. materialize/create views в БД;
2. сделать DB/profile audit по каждому view;
3. потом писать loaders;
4. только потом запускать первые baseline train/benchmark runs.

## Связанные Документы

- [DB Layer Closure Plan For Hierarchical MK/OOD Wave](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/db_layer_closure_plan_ru.md)
- [Quality And OOD Contract V2](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/quality_ood_contract_ru.md)
- [Hierarchical Classification And OOD Strategy](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/hierarchical_ood_strategy_ru.md)
- [V2 Maturation Micro-TZ](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/architecture/v2_maturation_micro_tz_ru.md)
