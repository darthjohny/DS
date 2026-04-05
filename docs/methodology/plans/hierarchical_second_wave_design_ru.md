# Hierarchical Second-Wave Design

## Цель

Этот документ фиксирует следующий этап после закрытия first-wave baseline:

- не пытаться дожать весь `refinement` одним flat multiclass run;
- перейти к second-wave design для `refinement`, calibration и decision layer;
- сначала зафиксировать contracts и только потом идти в код.

Документ нужен, чтобы:

- не раздувать training-code ad hoc логикой;
- не превращать иерархическую схему в новый монолит;
- заранее закрепить, какие решения считаются official-backed, а какие являются
  project policy.

## Инженерный Инвариант

Для `MTZ-M49 ... MTZ-M53` действует тот же рабочий стандарт:

- `1 файл = 1 ответственность`
- без giant-модулей и "общих" файлов на все stages
- `PEP 8`
- явная типизация
- простая Python-логика раньше сложной
- без лишних зависимостей
- после каждого маленького куска:
  - micro-QA
  - `ruff`
  - точечный `mypy/pyright`
  - целевые тесты
- после закрытия микро-ТЗ:
  - scoped big-QA только по написанному слою

Это правило касается и DB/views, и training-code, и reporting.

## Что Уже Известно По First Wave

### Coarse

- task: `gaia_id_coarse_classification`
- model: `hist_gradient_boosting`
- test accuracy: `0.992926`
- test balanced_accuracy: `0.992379`
- test macro_f1: `0.992573`

Вывод:

- coarse-layer уже достаточно стабилен для роли первого stage.

### Refinement

- task: `gaia_mk_refinement_classification`
- model: `hist_gradient_boosting`
- full subclass count after first-wave filters: `59`
- test accuracy: `0.320336`
- test balanced_accuracy: `0.187861`
- test macro_f1: `0.189683`

Вывод:

- flat refinement numerically работает;
- но в текущем виде это baseline, а не финальная конфигурация второго слоя.

### ID vs OOD

- task: `gaia_id_ood_classification`
- model: `hist_gradient_boosting`
- test accuracy: `0.995734`
- test balanced_accuracy: `0.926215`
- test macro_f1: `0.944521`

Вывод:

- отдельный OOD-stage уже имеет сильный baseline;
- его нужно не "переизобретать", а правильно встроить в decision layer.

## Official Опора Для Следующего Этапа

### HistGradientBoostingClassifier

Official scikit-learn docs фиксируют:

- `HistGradientBoostingClassifier` быстрее обычного `GradientBoostingClassifier`
  на больших данных (`n_samples >= 10_000`);
- estimator нативно поддерживает `NaN`;
- multiclass поддерживается из коробки;
- для multiclass строится `n_classes` деревьев на итерацию.

Практический вывод для проекта:

- coarse и OOD baseline логично продолжать именно на HGB как первом рабочем
  baseline;
- не нужно сейчас усложнять схему meta-estimator-ами только ради "красоты".

Официальный источник:

- [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)

### Multiclass Strategy

Official scikit-learn docs отдельно подчеркивают:

- все classifiers в scikit-learn умеют multiclass classification из коробки;
- модуль `sklearn.multiclass` нужен только если пользователь сознательно
  хочет экспериментировать с другой multiclass strategy;
- `OneVsRest` считается fair default choice, когда эксперимент с meta-strategy
  действительно нужен.

Практический вывод для проекта:

- second-wave refinement не нужно начинать с premature `OvR`/`OvO` оберток;
- сначала надо изменить сам task decomposition, а не оборачивать тот же flat task
  новой meta-strategy.

Официальный источник:

- [Multiclass and multioutput algorithms](https://scikit-learn.org/1.5/modules/multiclass.html)

### Cross-Validation При Редких Классах

Official docs по cross-validation предупреждают:

- stratification делает folds более однородными;
- это может скрывать часть естественной межфолдовой вариативности при редких
  классах;
- rare-class effect нельзя игнорировать, даже если split формально stratified.

Практический вывод для проекта:

- second-wave refinement нельзя проектировать как giant flat subclass-task с
  длинным rare tail;
- нужен отдельный support audit по coarse-группам и explicit policy,
  какие subclasses реально идут в train.

Официальный источник:

- [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html)

### Calibration И Threshold Tuning

Official scikit-learn docs фиксируют:

- calibrated probabilities нужно строить на данных, независимых от train subset;
- `CalibratedClassifierCV` делает это через CV;
- для multiclass calibration идет по классам в `OneVsRest` fashion;
- `TunedThresholdClassifierCV` позволяет подбирать threshold под выбранную metric,
  например `balanced_accuracy`;
- `balanced_accuracy` официально рекомендован для imbalanced datasets.

Практический вывод для проекта:

- next-wave decision layer должен использовать calibration и threshold tuning
  как отдельный слой, а не hard-coded константы в inference;
- binary `ID/OOD` stage можно и нужно проектировать с официальным threshold-tuning
  контуром;
- для multiclass stages calibration рассматриваем отдельно от task decomposition.

Официальные источники:

- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [TunedThresholdClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html)
- [balanced_accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)
- [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)

## Second-Wave Design Decisions

### Decision 1. Coarse Stage Не Ломаем

`coarse` baseline уже достаточно сильный.

На second wave:

- не переписываем его архитектуру;
- не заменяем без необходимости estimator family;
- используем его как стабильный первый stage.

### Decision 2. Flat Refinement Не Считаем Финальной Формой

Second-wave refinement проектируем не как один общий flat subclass-task на все
`59` subclasses, а как coarse-conditioned family.

Принцип:

- сначала coarse class;
- потом refinement только внутри соответствующей coarse-группы;
- rare-tail policy задается отдельно для каждой coarse-группы.

Это снижает:

- target cardinality на один model-run;
- влияние несвязанных subclass-конфузий между далекими классами;
- искусственное раздувание rare tail.

### Decision 3. Refinement Включаем Не Для Всех Coarse Classes Одинаково

Second-wave refinement должен идти через explicit enable-list.

Порядок:

1. сделать per-class support audit;
2. зафиксировать refinement-enabled coarse classes;
3. для остальных оставить только coarse-level decision или `unknown/review`.

Это означает:

- не каждый coarse class обязан получить свой second-wave subclass-model;
- тонкие ветки не должны насиловаться ради симметрии.

### Decision 4. OOD Gate Делаем Ранним Decision Stage

Operational order второй волны проектируем так:

1. quality gate;
2. `ID/OOD` gate;
3. coarse classifier;
4. class-specific refinement;
5. host / priority layer.

Причина:

- OOD не должен forced-fit-иться в normal stellar class раньше времени;
- отдельный OOD baseline уже достаточно силен для раннего screening.

### Decision 5. Unknown / Review Не Прячем В Probability Noise

`unknown/review` остается явным relation/layer, а не побочным эффектом inference.

Unknown возникает, если:

- объект не прошел quality gate;
- объект попал в `candidate_ood` / `ood`;
- coarse или refinement confidence не проходят agreed threshold;
- coarse class не входит в refinement-enabled list;
- subclass support policy запрещает confident refinement.

## Что Делаем Во Второй Волне

### 1. Per-Class Support Audit

Нужно собрать audit по каждой coarse-группе:

- сколько unique `source_id`;
- сколько full subclasses;
- support по каждому subclass;
- какой tail останется при cutoff `>= 15`, `>= 20`, `>= 30`.

Цель:

- не гадать, а формально выбрать refinement-enabled classes.

### Audit Result (`2026-03-28`)

Live audit уже снят и materialized в БД:

- `lab.gaia_mk_refinement_support_audit`
- `lab.gaia_mk_refinement_support_audit_summary`

Источник audit:

- `lab.v_gaia_mk_refinement_training`

Общий объем current refinement source:

- `155373` строк
- `155373` unique `source_id`
- `7` coarse classes
- `66` full subclasses

#### Результат По Coarse Classes

- `A`
  - `10` subclasses
  - min support `489`
  - retain `100%` при cutoff `15`, `20`, `30`
- `B`
  - `10` subclasses
  - min support `66`
  - retain `100%` при cutoff `15`, `20`, `30`
- `F`
  - `10` subclasses
  - min support `301`
  - retain `100%` при cutoff `15`, `20`, `30`
- `G`
  - `10` subclasses
  - min support `673`
  - retain `100%` при cutoff `15`, `20`, `30`
- `K`
  - `10` subclasses
  - min support `7`
  - retain `99.976%` при cutoff `15`, `20`, `30`
  - выпадает только `K9`
- `M`
  - `10` subclasses
  - min support `25`
  - retain `100%` при cutoff `15` и `20`
  - retain `99.804%` при cutoff `30`
  - на cutoff `30` выпадает `M9`
- `O`
  - `6` subclasses
  - total rows `40`
  - min support `1`
  - retain `0%` уже при cutoff `15`

#### Rare Tail

Поддержка `< 35`:

- `O3 = 1`
- `O4 = 4`
- `O6 = 5`
- `K9 = 7`
- `O7 = 8`
- `O8 = 11`
- `O9 = 11`
- `M9 = 25`

## First Second-Wave Policy

### Refinement-Enabled Coarse Classes

Вторая волна refinement запускается для:

- `A`
- `B`
- `F`
- `G`
- `K`
- `M`

### Coarse-Only Class

Во второй слой refinement пока не идет:

- `O`

Policy:

- `O` остается coarse-only;
- при необходимости уходит в `unknown/review`, а не forced-fit-ится в subclass-stage.

### Default Support Policy

Second-wave default cutoff:

- full subclass support `>= 15`

Причина:

- этот cutoff уже стабилизировал first-wave benchmark;
- для `A/B/F/G/K/M` он практически не режет useful mass;
- при этом убирает откровенно нежизнеспособный rare tail.

### Explicit Exclusions На Старте Second Wave

На первом second-wave iteration не включаем в refinement families:

- весь `O` tail
- `K9`

`M9`:

- остается допустимым на default cutoff `15`;
- но помечается как borderline subclass;
- при более строгом family-specific cutoff `30` он будет исключен.

### 2. Refinement Families

Вместо одного flat refinement source проектируем семейство relation/view:

- `lab.v_gaia_mk_refinement_training_a`
- `lab.v_gaia_mk_refinement_training_b`
- `lab.v_gaia_mk_refinement_training_f`
- `lab.v_gaia_mk_refinement_training_g`
- `lab.v_gaia_mk_refinement_training_k`
- `lab.v_gaia_mk_refinement_training_m`

Это не обязательный final список, а target shape для design.

Final family contracts второй волны зафиксированы отдельно в:

- [refinement_family_contracts_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/refinement_family_contracts_ru.md)

Классы с недостаточным support:

- остаются coarse-only;
- или уходят в `unknown/review`;
- но не заставляют весь refinement-task деградировать.

### 3. Calibration Layer

Нужно отдельно спроектировать probability contract:

- `coarse_probability`
- `ood_probability`
- `refinement_probability`
- `confidence_margin`

Правило:

- calibration не живет inside core model class;
- calibration — отдельный stage поверх fitted classifier.

Second-wave calibration/threshold policy зафиксирована отдельно в:

- [calibration_threshold_policy_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/calibration_threshold_policy_ru.md)

### 4. Threshold Policy

Нужно отдельно зафиксировать:

- threshold для `OOD`;
- threshold для `unknown/review`;
- threshold для запуска refinement после coarse.

При этом:

- `ID/OOD` threshold проектируем через official `TunedThresholdClassifierCV`;
- multiclass `coarse/refinement` confidence thresholds остаются project policy,
  но подбираются только на validation.

### 5. Final Decision Contract

Вторая волна должна закончиться не просто новыми моделями, а explicit
decision contract:

- `final_domain_state`
  - `id`
  - `candidate_ood`
  - `ood`
  - `unknown`
- `final_coarse_class`
- `final_refinement_label`
- `final_refinement_state`
  - `not_attempted`
  - `accepted`
  - `rejected_to_unknown`
- `priority_state`

Final contract зафиксирован отдельно в:

- [final_decision_contract_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/final_decision_contract_ru.md)

## Что Не Делаем Во Второй Волне

- не добавляем ensemble/voting до фиксации calibrated baseline;
- не вводим giant meta-runner "на все stages";
- не прячем threshold logic внутрь legacy inference;
- не смешиваем calibration, decision policy и training source в одном модуле;
- не обучаем `unknown` как ad hoc помойку без отдельного contract.

## Expected End State

После second-wave design и реализации проект должен иметь:

- устойчивый coarse stage;
- ранний OOD gate;
- refinement как family of class-specific tasks;
- отдельный `unknown/review` outcome;
- decision contract, который можно честно объяснить в ВКР и поддерживать в коде.

## Current Code Status

На `2026-03-28` first code-side second-wave slice уже реализован:

- отдельный DB helper для family views;
- отдельные dataset contracts и feature contracts;
- отдельные family loaders;
- отдельные family training-frame helpers;
- отдельный task registry;
- отдельные benchmark/training runners;
- отдельный CLI dispatch для family tasks.

Что еще не реализовано в коде:

- higher-level runner/CLI поверх final decision bridge;
- calibrated handoff logic для `coarse -> refinement`;
- explicit `candidate_ood` threshold policy layer поверх binary tuned threshold.

На `2026-03-28` второй code-slice тоже закрыт:

- отдельные post-hoc calibration helpers;
- отдельные threshold-tuning helpers;
- отдельный `ID/OOD` gate contract;
- отдельный runner для calibrated+tuned binary gate.

Live smoke-run (`HistGradientBoosting`, `gaia_id_ood_classification`, `limit=5000`):

- test accuracy: `0.969272`
- test balanced_accuracy: `0.968762`
- test macro_f1: `0.961488`
- tuned threshold: `0.041898`

Вывод:

- `ID/OOD` second-wave gate уже работает end-to-end;
- следующий кодовый шаг теперь не в calibration/tuning, а в final decision-layer
  orchestration.

На `2026-03-28` третий code-slice тоже закрыт:

- отдельный handoff module для `coarse -> refinement`;
- отдельный final decision routing module;
- unit-level routing contract для `quality`, `OOD`, `coarse-only`,
  `accepted refinement` и `rejected_to_unknown`.

На `2026-03-28` четвертый code-slice тоже закрыт:

- отдельный probability-summary helper;
- отдельный compact coarse scoring helper;
- отдельный compact refinement-family scoring helper;
- отдельный bridge между stage outputs и final decision routing.

Вывод:

- core decision logic и merge layer больше не живут только в docs;
- `candidate_ood` secondary policy и decision runner тоже вынесены в отдельные модули.

На `2026-03-28` шестой code-slice тоже закрыт:

- отдельный priority input adapter;
- отдельный priority integration module поверх explainable ranking;
- `final_decision` больше не пробрасывает stale `priority_state` из upstream frame;
- targeted unit-tests для priority input и priority integration.

Вывод:

- priority теперь действительно живет после final routing;
- ranking-contract не смешан с decision routing;

На `2026-03-28` седьмой code-slice тоже закрыт:

- отдельный threshold-policy artifact layer для `ID/OOD`;
- отдельный saved-artifact bundle loader для `OOD/coarse/refinement/host`;
- отдельный higher-level runner поверх saved artifacts;
- отдельный persistence-layer для `decision_input/final_decision/priority_*`;
- отдельная `decide` CLI-команда;
- targeted unit-tests для artifacts, bundle, runner и CLI.

Вывод:

- second-wave bridge теперь закрыт не только на frame-level, но и на saved-artifact уровне;
- end-to-end orchestration можно запускать без ручного склеивания stage outputs;
- следующий практический этап уже не в routing-коде, а в observability notebook
  и end-to-end validation run.
