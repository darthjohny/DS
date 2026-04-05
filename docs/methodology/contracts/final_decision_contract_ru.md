# Final Decision Contract

## Цель

Этот документ фиксирует final decision layer второй волны.

Нужен он для того, чтобы:

- не растворить routing-логику между `quality`, `OOD`, `coarse`, `refinement`
  и `priority` по разным модулям;
- заранее определить итоговые states проекта;
- сохранить traceability для каждого решения;
- не смешивать model output и business decision.

## Инженерный Инвариант

При будущей реализации этого слоя действует тот же стандарт:

- `1 файл = 1 ответственность`
- final decision orchestration отдельно от base model code
- threshold logic отдельно от data loading
- priority integration отдельно от final classification mapping
- `PEP 8`
- явная типизация
- простая логика раньше сложной
- без лишних зависимостей
- после каждого небольшого куска:
  - micro-QA
  - `ruff`
  - точечный `mypy/pyright`
  - целевые tests
- после завершения микро-ТЗ:
  - scoped big-QA только по написанному слою

## Основание Контракта

Этот contract опирается на уже зафиксированные project layers:

- [quality_ood_contract_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/quality_ood_contract_ru.md)
- [hierarchical_ood_strategy_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/hierarchical_ood_strategy_ru.md)
- [hierarchical_second_wave_design_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/hierarchical_second_wave_design_ru.md)
- [refinement_family_contracts_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/refinement_family_contracts_ru.md)
- [calibration_threshold_policy_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/calibration_threshold_policy_ru.md)

И на official model-side semantics:

- calibrated probabilities и threshold tuning — из scikit-learn docs:
  [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html),
  [TunedThresholdClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html)

Важно:

- official docs задают semantics вероятностей и threshold-tuning;
- сами decision states проекта — это project contract.

## High-Level Routing Order

Final decision layer обязан применять stages в фиксированном порядке:

1. `quality gate`
2. `ID/OOD gate`
3. `coarse classification`
4. `refinement handoff`
5. `refinement decision`
6. `priority integration`

Это не рекомендация, а explicit routing order.

Причина:

- плохие или OOD-объекты не должны forced-fit-иться в normal refinement;
- refinement не должен запускаться раньше coarse;
- priority не должен маскировать upstream uncertainty.

## Final Output Contract

Каждая итоговая запись должна уметь хранить минимум:

- `source_id`
- `final_domain_state`
- `final_quality_state`
- `final_coarse_class`
- `final_coarse_confidence`
- `final_refinement_label`
- `final_refinement_state`
- `final_refinement_confidence`
- `final_decision_reason`
- `final_decision_policy_version`
- `priority_state`

Допустимые nullable fields:

- `final_coarse_class`
- `final_refinement_label`
- `final_refinement_confidence`
- `priority_state`

## Final States

### `final_domain_state`

Допустимые значения:

- `id`
- `candidate_ood`
- `ood`
- `unknown`

Смысл:

- `id`
  - объект считается нормальным объектом целевого stellar-domain
- `candidate_ood`
  - объект не признан clean `id`, но еще не переведен в hard `ood`
- `ood`
  - объект считается внешним по отношению к normal stellar-domain
- `unknown`
  - система не имеет права сделать уверенное normal-domain решение

### `final_quality_state`

Допустимые значения:

- `pass`
- `unknown`
- `reject`

Это не дублирует `final_domain_state`, а хранит отдельный quality outcome.

### `final_refinement_state`

Допустимые значения:

- `not_attempted`
- `accepted`
- `rejected_to_unknown`

Смысл:

- `not_attempted`
  - refinement вообще не запускался
- `accepted`
  - refinement дал final accepted subclass result
- `rejected_to_unknown`
  - refinement запускался, но уверенного решения не дал

## Routing Rules

## Rule 1. Quality Reject Идет Раньше Всего

Если:

- `quality_state = 'reject'`

то:

- `final_domain_state = 'unknown'`
- `final_quality_state = 'reject'`
- `final_coarse_class = NULL`
- `final_refinement_state = 'not_attempted'`
- `priority_state = NULL`

Причина:

- rejected object не должен участвовать ни в normal classification,
  ни в priority pipeline.

## Rule 2. Hard OOD Не Идет В Normal Classification

Если:

- `ood_state = 'ood'`

то:

- `final_domain_state = 'ood'`
- `final_refinement_state = 'not_attempted'`
- `final_refinement_label = NULL`
- `priority_state = NULL` или explicit low-priority outside normal pipeline

Project policy первой реализации:

- hard `ood` не идет в refinement
- hard `ood` не идет в planet-host priority logic

## Rule 3. Candidate OOD Не Считается Clean ID

Если:

- `ood_state = 'candidate_ood'`

то:

- `final_domain_state` не может быть сразу `id`
- объект либо:
  - остается `candidate_ood`
  - либо уходит в `unknown`

First decision policy:

- `candidate_ood` не схлопываем с normal `id` до отдельного validated rule.

## Rule 4. Coarse Запускается Только Для Clean In-Domain Objects

Coarse classification разрешена только если одновременно:

- `quality_state = 'pass'`
- upstream `ood_state = 'in_domain'`

Иначе:

- object не идет в normal coarse stage.

## Rule 5. Refinement Не Обязателен Для Каждого Clean Object

Refinement запускается только если одновременно:

- `final_domain_state = 'id'`
- coarse class входит в refinement-enabled list
- coarse confidence проходит agreed threshold

Если это не выполнено:

- `final_refinement_state = 'not_attempted'`
- объект остается на coarse-level outcome

### Refinement-Enabled Coarse Classes

На текущем этапе:

- `A`
- `B`
- `F`
- `G`
- `K`
- `M`

`O`:

- остается coarse-only.

## Rule 6. Refinement Failure Переводит В `unknown`, А Не В Случайный Subclass

Если refinement stage запущен, но:

- family confidence ниже threshold
  или
- family policy считает subclass invalid/borderline

то:

- `final_refinement_state = 'rejected_to_unknown'`
- `final_domain_state = 'unknown'`
- `final_refinement_label = NULL`

Причина:

- лучше explicit uncertainty, чем forced wrong subclass.

## Rule 7. Priority Layer Идет Последним

`priority_state` заполняется только после final classification routing.

`priority_state` разрешен только если:

- `final_domain_state = 'id'`
- `final_quality_state = 'pass'`

Для:

- `candidate_ood`
- `ood`
- `unknown`

priority либо `NULL`, либо explicit low-priority outside normal observation queue.

## Traceability Contract

Final decision relation/view обязан хранить:

- `quality_reason`
- `ood_reason`
- `coarse_model_name`
- `coarse_model_version`
- `coarse_policy_version`
- `refinement_model_name`
- `refinement_model_version`
- `refinement_family_name`
- `refinement_policy_version`
- `ood_model_name`
- `ood_model_version`
- `ood_threshold_policy_version`
- `decision_timestamp_utc`

Причина:

- любое итоговое решение должно быть восстановимо;
- unknown/OOD outcomes не должны быть "черным ящиком".

## Minimal Human-Readable Outcomes

Система должна уметь объяснить минимум 4 типа исхода:

### 1. Normal Classified

- `final_domain_state = 'id'`
- `final_coarse_class` задан
- `final_refinement_state = 'accepted'`
- `final_refinement_label` задан

Пример смысла:

- звезда отнесена к normal stellar domain
- coarse class определен
- subclass принят
- priority можно считать дальше

### 2. Coarse-Only

- `final_domain_state = 'id'`
- `final_coarse_class` задан
- `final_refinement_state = 'not_attempted'`

Пример смысла:

- объект нормальный, но fine refinement либо не нужен, либо не разрешен policy.

### 3. Unknown / Review

- `final_domain_state = 'unknown'`

Возможные причины:

- quality fail
- low coarse confidence
- refinement reject
- coarse class coarse-only with no accepted subclass path

### 4. OOD

- `final_domain_state = 'ood'`
  или
- `final_domain_state = 'candidate_ood'`

Смысл:

- объект не должен идти как normal in-domain star в основной pipeline.

## Что Не Делаем

- не смешиваем `candidate_ood` и `unknown` без явного policy version
- не подменяем `unknown` low-confidence guessed subclass-ом
- не пускаем `ood` в normal priority flow
- не храним final decision как одну колонку "label"
- не прячем routing order внутри нескольких несвязанных helper-функций

## MTZ-M52 Deliverable

`MTZ-M52` считается закрытым, когда:

- final decision states перечислены явно;
- routing order зафиксирован явно;
- traceability fields перечислены явно;
- дальнейшая code-side реализация может идти без архитектурной импровизации.

## Current Code Status (`2026-03-28`)

Initial code-side implementation уже вынесена в отдельные модули:

- [refinement_handoff.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/posthoc/refinement_handoff.py)
- [final_decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/posthoc/final_decision.py)

Что уже реализовано:

- отдельная handoff policy между `coarse` и `refinement`;
- отдельный final routing frame builder;
- отдельные stage-scoring helpers для `coarse` и `refinement family`;
- отдельный bridge для merge real stage outputs в final routing input;
- отдельный saved-artifact bundle loader для `OOD/coarse/refinement/host`;
- отдельный threshold-policy artifact contract для `ID/OOD` gate;
- отдельный higher-level runner поверх saved artifacts;
- отдельный persistence-layer для final decision artifacts;
- отдельный priority input adapter поверх final decision contract;
- отдельный priority integration module поверх explainable ranking layer;
- отдельная `decide` CLI-команда поверх saved artifacts;
- quality-first и `OOD`-first routing order зафиксирован в коде;
- `candidate_ood` не схлопывается с `id`;
- `O` остается coarse-only через refinement enable-list;
- refinement failure переводится в `unknown`, а не в forced subclass;
- `priority_state` больше не протекает из upstream frame раньше priority stage.

Live orchestration поверх реальных stage outputs теперь закрыта на frame-level:

- bridge уже умеет принимать реальные stage outputs;
- отдельный decision runner уже собирает `OOD -> coarse -> refinement -> final decision`;
- priority integration считается отдельным layer после final routing;
- higher-level runner/CLI теперь работает поверх saved model artifacts и отдельного
  threshold-policy artifact.

Для `ID/OOD` это принципиально важно:

- tuned threshold не растворяется в CLI flags;
- threshold policy хранится отдельно от estimator artifact;
- notebook и end-to-end run читают один и тот же versioned threshold contract.

Scoped QA по initial decision-layer slice:

- `ruff` ok
- `mypy` ok
- `pyright` ok
- targeted `pytest` ok
