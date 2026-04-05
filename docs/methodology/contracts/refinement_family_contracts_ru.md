# Refinement Family Contracts

## Цель

Этот документ фиксирует second-wave contract для family-based refinement.

Нужен он для того, чтобы:

- не продолжать refinement как один flat `59`-class task;
- не тащить новую decomposition-логику сразу в код;
- сначала зафиксировать family-view contracts, а уже потом materialize-ить DB views
  и писать loaders.

## Инженерный Инвариант

Для second-wave refinement действуют те же правила:

- `1 файл = 1 ответственность`
- без giant refinement-модуля "на все классы"
- `PEP 8`
- явная типизация
- простая логика раньше сложной
- без лишних зависимостей
- после каждого небольшого куска:
  - micro-QA
  - `ruff`
  - точечный `mypy/pyright`
  - целевые тесты
- после завершения микро-ТЗ:
  - scoped big-QA только по написанному слою

## Official Опора

### Multiclass Docs

Official scikit-learn docs фиксируют:

- multiclass classification поддерживается большинством classifiers из коробки;
- отдельные multiclass meta-estimator-стратегии нужны только если пользователь
  сознательно хочет экспериментировать с альтернативной decomposition.

Практический вывод:

- second-wave refinement не нужно начинать с `OvR`/`OvO`-оберток;
- сначала надо изменить сам task decomposition.

Официальный источник:

- [Multiclass and multioutput algorithms](https://scikit-learn.org/stable/modules/multiclass.html)

### Cross-Validation Docs

Official docs предупреждают:

- stratification решает только часть проблем при несбалансированных/редких классах;
- редкий tail все равно нужно учитывать отдельно.

Практический вывод:

- family decomposition здесь является project inference из official CV behavior
  и live support audit;
- это не "официальное предписание scikit-learn", а инженерно корректная реакция на
  observed rare-tail structure.

Официальный источник:

- [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html)

### HistGradientBoostingClassifier

Official docs фиксируют:

- `HistGradientBoostingClassifier` хорошо подходит для больших datasets;
- поддерживает missing values;
- поддерживает multiclass tasks.

Практический вывод:

- family views проектируем без смены baseline estimator family;
- задача этого шага — не заменить classifier, а улучшить task decomposition.

Официальный источник:

- [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)

## Upstream Source

Все second-wave family views строятся от:

- `lab.v_gaia_mk_refinement_training`

Причина:

- это уже gated in-domain source;
- в нем уже есть core features, FLAME enrichment и label fields;
- через него second-wave design не ломает coarse/OOD contracts.

## Common Family Contract

### Общий Shape

Каждый family view должен быть отдельной relation:

- `lab.v_gaia_mk_refinement_training_a`
- `lab.v_gaia_mk_refinement_training_b`
- `lab.v_gaia_mk_refinement_training_f`
- `lab.v_gaia_mk_refinement_training_g`
- `lab.v_gaia_mk_refinement_training_k`
- `lab.v_gaia_mk_refinement_training_m`

`O`-family view не создаем в second wave.

### Common Required Filters

Каждый family view обязан применять:

- `quality_state = 'pass'`
- `ood_state = 'in_domain'`
- `spectral_class = '<family>'`
- `spectral_subclass IS NOT NULL`
- explicit feature completeness:
  - `teff_gspphot IS NOT NULL`
  - `logg_gspphot IS NOT NULL`
  - `mh_gspphot IS NOT NULL`
  - `bp_rp IS NOT NULL`
  - `parallax IS NOT NULL`
  - `parallax_over_error IS NOT NULL`
  - `ruwe IS NOT NULL`
  - `radius_flame IS NOT NULL`
  - `lum_flame IS NOT NULL`
  - `evolstage_flame IS NOT NULL`
  - `phot_g_mean_mag IS NOT NULL`
- support по full subclass `spectral_class + spectral_subclass >= 15`

### Common Feature Contract

Во всех family views first-wave common feature set одинаковый:

- `teff_gspphot`
- `logg_gspphot`
- `mh_gspphot`
- `bp_rp`
- `parallax`
- `parallax_over_error`
- `ruwe`
- `radius_flame`
- `lum_flame`
- `evolstage_flame`
- `phot_g_mean_mag`

### Common Label Contract

Каждый family view обязан отдавать:

- `source_id`
- `spectral_class`
- `spectral_subclass`
- `luminosity_class`
- `peculiarity_suffix`
- `full_subclass_label`
  - `spectral_class || spectral_subclass::text`

### Target Policy

Second-wave refinement target:

- `spectral_subclass`

`full_subclass_label` сохраняется как traceable human-readable label,
но не считается основным target field.

Причина:

- внутри family-view `spectral_class` уже фиксирован;
- subclass-digit становится нормальным class target без смешения с другими
  coarse groups;
- final decision layer потом приклеивает coarse prefix обратно.

### Task Names

Code-side second-wave family tasks называются так:

- `gaia_mk_refinement_a_classification`
- `gaia_mk_refinement_b_classification`
- `gaia_mk_refinement_f_classification`
- `gaia_mk_refinement_g_classification`
- `gaia_mk_refinement_k_classification`
- `gaia_mk_refinement_m_classification`

Эти task names маппятся `1:1` на family views и не смешиваются с legacy flat task
`gaia_mk_refinement_classification`.

## Family-Specific Contracts

### `lab.v_gaia_mk_refinement_training_a`

- spectral class: `A`
- target cardinality: `10`
- full subclasses:
  - `A0 ... A9`
- policy:
  - all subclasses остаются включенными
- expected row budget:
  - `27354`

### `lab.v_gaia_mk_refinement_training_b`

- spectral class: `B`
- target cardinality: `10`
- full subclasses:
  - `B0 ... B9`
- policy:
  - all subclasses остаются включенными
- expected row budget:
  - `9958`

### `lab.v_gaia_mk_refinement_training_f`

- spectral class: `F`
- target cardinality: `10`
- full subclasses:
  - `F0 ... F9`
- policy:
  - all subclasses остаются включенными
- expected row budget:
  - `41003`

### `lab.v_gaia_mk_refinement_training_g`

- spectral class: `G`
- target cardinality: `10`
- full subclasses:
  - `G0 ... G9`
- policy:
  - all subclasses остаются включенными
- expected row budget:
  - `35384`

### `lab.v_gaia_mk_refinement_training_k`

- spectral class: `K`
- target cardinality: `9`
- included subclasses:
  - `K0 ... K8`
- explicit exclusion:
  - `K9`
- reason:
  - `K9` support = `7`, below default cutoff `15`
- expected row budget:
  - `28866`

### `lab.v_gaia_mk_refinement_training_m`

- spectral class: `M`
- target cardinality: `10`
- included subclasses:
  - `M0 ... M9`
- note:
  - `M9` support = `25`
  - subclass marked as borderline for future stricter policy
- default second-wave policy:
  - `M9` remains included
- expected row budget:
  - `12761`

## Coarse-Only Policy

### `O`

`O` не получает second-wave family view.

Причина:

- total rows `40`
- full subclasses `6`
- ни один subclass не проходит cutoff `15`

Практический контракт:

- `O` остается coarse-only;
- при необходимости идет в `unknown/review`;
- не forced-fit-ится в subclass family.

## Что Не Делаем На Этом Шаге

- не materialize-им views прямо в этом документе;
- не пишем loaders;
- не вводим class-specific feature engineering;
- не добавляем class-specific estimators без необходимости;
- не смешиваем calibration policy с view contract.

## Критерий Готовности

`MTZ-M50` считается закрытым, когда:

- семейство refinement views определено явно;
- для каждой family relation зафиксированы filters, features и target;
- explicit exclusions и coarse-only policy записаны отдельно;
- следующий шаг может уже materialize-ить DB views без архитектурной импровизации.

## Live Status

На `2026-03-28` family views уже materialized в live БД:

- `lab.v_gaia_mk_refinement_training_a`: `26693`
- `lab.v_gaia_mk_refinement_training_b`: `9881`
- `lab.v_gaia_mk_refinement_training_f`: `40705`
- `lab.v_gaia_mk_refinement_training_g`: `34639`
- `lab.v_gaia_mk_refinement_training_k`: `28482`
- `lab.v_gaia_mk_refinement_training_m`: `8552`
