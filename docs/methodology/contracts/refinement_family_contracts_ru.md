# Контракты refinement-семейств

## Цель

Этот документ фиксирует контракт второй рабочей версии для refinement,
разбитого по семействам.

Нужен он для того, чтобы:

- не продолжать refinement как одну плоскую задачу на `59` классов;
- не тащить новую decomposition-логику сразу в код;
- сначала зафиксировать контракты представлений по семействам, а уже потом
  материализовать представления в БД и писать загрузчики.

## Общий принцип

Документ фиксирует только устройство семейства задач, а не нашу внутреннюю
организацию работы по файлам и шагам.

Здесь задаются:

- состав семейств;
- общие фильтры;
- набор признаков;
- правила, какие классы остаются только на coarse-уровне.

## Документационная опора

### Multiclass Docs

Документация `scikit-learn` фиксирует:

- многоклассовая классификация поддерживается большинством классификаторов из
  коробки;
- отдельные multiclass meta-estimator-стратегии нужны только если пользователь
  сознательно хочет экспериментировать с альтернативной decomposition.

Практический вывод:

- refinement второй рабочей версии не нужно начинать с `OvR`/`OvO`-оберток;
- сначала надо изменить сам task decomposition.

Официальный источник:

- [Multiclass and multioutput algorithms](https://scikit-learn.org/stable/modules/multiclass.html)

### Cross-Validation Docs

Документация предупреждает:

- stratification решает только часть проблем при несбалансированных/редких классах;
- редкий tail все равно нужно учитывать отдельно.

Практический вывод:

- разбиение по семействам здесь является выводом проекта из поведения
  кросс-валидации и обзора поддержки по классам;
- это не "официальное предписание scikit-learn", а инженерно корректная реакция на
  наблюдаемую редкую хвостовую структуру.

Официальный источник:

- [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html)

### HistGradientBoostingClassifier

Документация фиксирует:

- `HistGradientBoostingClassifier` хорошо подходит для больших наборов данных;
- поддерживает missing values;
- поддерживает multiclass tasks.

Практический вывод:

- семейства проектируем без смены базового семейства моделей;
- задача этого шага — не заменить классификатор, а улучшить разбиение задачи.

Официальный источник:

- [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)

## Исходная таблица

Все семейства второй рабочей версии строятся от:

- `lab.v_gaia_mk_refinement_training`

Причина:

- это уже отфильтрованный внутридоменный источник;
- в нем уже есть core features, FLAME enrichment и label fields;
- через него новая версия не ломает контракты `coarse/OOD`.

## Общий контракт семейств

### Общая форма

Каждое семейство должно быть отдельным представлением:

- `lab.v_gaia_mk_refinement_training_a`
- `lab.v_gaia_mk_refinement_training_b`
- `lab.v_gaia_mk_refinement_training_f`
- `lab.v_gaia_mk_refinement_training_g`
- `lab.v_gaia_mk_refinement_training_k`
- `lab.v_gaia_mk_refinement_training_m`

`O`-family view не создаем в second wave.

### Общие обязательные фильтры

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

### Общий контракт признаков

Во всех семействах первой рабочей версии набор признаков одинаковый:

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

### Общий контракт меток

Каждый family view обязан отдавать:

- `source_id`
- `spectral_class`
- `spectral_subclass`
- `luminosity_class`
- `peculiarity_suffix`
- `full_subclass_label`
  - `spectral_class || spectral_subclass::text`

### Политика целевого признака

Целевой признак второй рабочей версии:

- `spectral_subclass`

`full_subclass_label` сохраняется как прослеживаемая человекочитаемая метка,
но не считается основным target field.

Причина:

- внутри family-view `spectral_class` уже фиксирован;
- цифра подкласса становится нормальным целевым классом без смешения с другими
  coarse groups;
- слой итогового решения потом добавляет coarse-префикс обратно.

### Имена задач

Code-side second-wave family tasks называются так:

- `gaia_mk_refinement_a_classification`
- `gaia_mk_refinement_b_classification`
- `gaia_mk_refinement_f_classification`
- `gaia_mk_refinement_g_classification`
- `gaia_mk_refinement_k_classification`
- `gaia_mk_refinement_m_classification`

Эти имена задач сопоставляются `1:1` с представлениями по семействам и не
смешиваются со старой плоской задачей `gaia_mk_refinement_classification`.

## Контракты по семействам

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

## Политика coarse-only

### `O`

`O` не получает second-wave family view.

Причина:

- total rows `40`
- full subclasses `6`
- ни один subclass не проходит cutoff `15`

Практический контракт:

- `O` остается coarse-only;
- при необходимости идет в `unknown/review`;
- не подгоняется принудительно в семейство подклассов.

## Чего не делаем на этом шаге

- не materialize-им views прямо в этом документе;
- не пишем loaders;
- не вводим class-specific feature engineering;
- не добавляем class-specific estimators без необходимости;
- не смешиваем политику калибровки с контрактом представлений.

## Критерий готовности

Документ считается зафиксированным, когда:

- семейство refinement views определено явно;
- для каждого семейства зафиксированы фильтры, признаки и целевой признак;
- explicit exclusions и coarse-only policy записаны отдельно;
- следующий шаг может уже материализовать представления в БД без архитектурной
  импровизации.

## Состояние реализации

На `2026-03-28` семейства уже материализованы в рабочей БД:

- `lab.v_gaia_mk_refinement_training_a`: `26693`
- `lab.v_gaia_mk_refinement_training_b`: `9881`
- `lab.v_gaia_mk_refinement_training_f`: `40705`
- `lab.v_gaia_mk_refinement_training_g`: `34639`
- `lab.v_gaia_mk_refinement_training_k`: `28482`
- `lab.v_gaia_mk_refinement_training_m`: `8552`
