# Контракт итогового решения

## Цель

Этот документ фиксирует слой итогового решения для второй рабочей версии.

Нужен он для того, чтобы:

- не растворить логику маршрутизации между `quality`, `OOD`, `coarse`, `refinement`
  и `priority` по разным модулям;
- заранее определить итоговые состояния проекта;
- сохранить прослеживаемость для каждого решения;
- не смешивать выходы моделей и прикладное решение.

## Общий принцип

Этот документ описывает не реализацию, а правила, которым должен
соответствовать слой итогового решения.

Поэтому:

- порядок шагов должен быть зафиксирован явно;
- итоговые состояния должны быть перечислены явно;
- причины решений должны сохраняться вместе с результатом.

## Основание контракта

Этот контракт опирается на уже зафиксированные слои проекта:

- [quality_ood_contract_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/quality_ood_contract_ru.md)
- [hierarchical_ood_strategy_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/hierarchical_ood_strategy_ru.md)
- [hierarchical_second_wave_design_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/hierarchical_second_wave_design_ru.md)
- [refinement_family_contracts_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/refinement_family_contracts_ru.md)
- [calibration_threshold_policy_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/calibration_threshold_policy_ru.md)

И на официальную семантику работы с вероятностями и порогами:

- calibrated probabilities и threshold tuning — из scikit-learn docs:
  [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html),
  [TunedThresholdClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html)

Важно:

- официальная документация задает смысл вероятностей и настройки порогов;
- сами состояния итогового решения задаются правилами проекта.

## Порядок маршрутизации

Слой итогового решения обязан применять шаги в фиксированном порядке:

1. `quality gate`
2. `ID/OOD gate`
3. `coarse classification`
4. `refinement handoff`
5. `refinement decision`
6. `priority integration`

Это не рекомендация, а обязательный порядок.

Причина:

- плохие или OOD-объекты не должны силой подгоняться под обычный refinement;
- refinement не должен запускаться раньше coarse;
- priority не должен маскировать upstream uncertainty.

## Выходной контракт

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

Допустимые пустые поля:

- `final_coarse_class`
- `final_refinement_label`
- `final_refinement_confidence`
- `priority_state`

## Итоговые состояния

### `final_domain_state`

Допустимые значения:

- `id`
- `candidate_ood`
- `ood`
- `unknown`

Смысл:

- `id`
  - объект считается нормальным объектом целевой звездной области
- `candidate_ood`
  - объект не признан чистым `id`, но еще не переведен в жесткий `ood`
- `ood`
  - объект считается внешним по отношению к обычной звездной области
- `unknown`
  - система не имеет права сделать уверенное решение

### `final_quality_state`

Допустимые значения:

- `pass`
- `unknown`
- `reject`

Это не дублирует `final_domain_state`, а хранит отдельный результат проверки качества.

### `final_refinement_state`

Допустимые значения:

- `not_attempted`
- `accepted`
- `rejected_to_unknown`

Смысл:

- `not_attempted`
  - refinement вообще не запускался
- `accepted`
  - refinement дал принятый итоговый подкласс
- `rejected_to_unknown`
  - refinement запускался, но уверенного решения не дал

## Правила маршрутизации

## Правило 1. Отклонение по качеству идет раньше всего

Если:

- `quality_state = 'reject'`

то:

- `final_domain_state = 'unknown'`
- `final_quality_state = 'reject'`
- `final_coarse_class = NULL`
- `final_refinement_state = 'not_attempted'`
- `priority_state = NULL`

Причина:

- отклоненный объект не должен участвовать ни в обычной классификации,
  ни в расчете приоритета.

## Правило 2. Жесткий `OOD` не идет в обычную классификацию

Если:

- `ood_state = 'ood'`

то:

- `final_domain_state = 'ood'`
- `final_refinement_state = 'not_attempted'`
- `final_refinement_label = NULL`
- `priority_state = NULL` или явно указывает на низкий приоритет вне обычной очереди наблюдений

В первой рабочей версии:

- hard `ood` не идет в refinement
- hard `ood` не идет в слой приоритета для объектов, похожих на host-профиль

## Правило 3. `candidate_ood` не считается обычным `ID`

Если:

- `ood_state = 'candidate_ood'`

то:

- `final_domain_state` не может быть сразу `id`
- объект либо:
  - остается `candidate_ood`
  - либо уходит в `unknown`

В первой рабочей версии:

- `candidate_ood` не схлопываем с обычным `id` до появления отдельно
  подтвержденного правила.

## Правило 4. Coarse запускается только для чистых внутридоменных объектов

Coarse classification разрешена только если одновременно:

- `quality_state = 'pass'`
- upstream `ood_state = 'in_domain'`

Иначе:

- объект не идет в обычный coarse-этап.

## Правило 5. Refinement не обязателен для каждого чистого объекта

Refinement запускается только если одновременно:

- `final_domain_state = 'id'`
- coarse-класс входит в список, для которого разрешен refinement
- coarse confidence проходит agreed threshold

Если это не выполнено:

- `final_refinement_state = 'not_attempted'`
- объект остается на coarse-уровне

### Классы, для которых разрешен refinement

На текущем этапе:

- `A`
- `B`
- `F`
- `G`
- `K`
- `M`

`O` остается только на coarse-уровне.

## Правило 6. Ошибка refinement переводит в `unknown`, а не в случайный подкласс

Если refinement stage запущен, но:

- family confidence ниже threshold
  или
- family policy считает subclass invalid/borderline

то:

- `final_refinement_state = 'rejected_to_unknown'`
- `final_domain_state = 'unknown'`
- `final_refinement_label = NULL`

Причина:

- лучше явная неопределенность, чем принудительно выбранный неверный подкласс.

## Правило 7. Слой приоритета идет последним

`priority_state` заполняется только после final classification routing.

`priority_state` разрешен только если:

- `final_domain_state = 'id'`
- `final_quality_state = 'pass'`

Для:

- `candidate_ood`
- `ood`
- `unknown`

priority либо `NULL`, либо явно указывает на низкий приоритет вне обычной
очереди наблюдений.

## Контракт прослеживаемости

Итоговая таблица решения обязана хранить:

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

## Минимально объяснимые исходы

Система должна уметь объяснить минимум 4 типа исхода:

### 1. Нормальная классификация

- `final_domain_state = 'id'`
- `final_coarse_class` задан
- `final_refinement_state = 'accepted'`
- `final_refinement_label` задан

Пример смысла:

- звезда отнесена к обычной звездной области;
- крупный класс определен;
- подкласс принят;
- приоритет можно считать дальше.

### 2. Только coarse-уровень

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

- объект не должен идти как обычная внутридоменная звезда в основной контур.

## Чего не делаем

- не смешиваем `candidate_ood` и `unknown` без явного policy version
- не подменяем `unknown` low-confidence guessed subclass-ом
- не пускаем `ood` в обычный контур приоритета
- не храним final decision как одну колонку "label"
- не прячем routing order внутри нескольких несвязанных helper-функций

## Критерий готовности документа

Документ считается зафиксированным, когда:

- final decision states перечислены явно;
- routing order зафиксирован явно;
- traceability fields перечислены явно;
- дальнейшая code-side реализация может идти без архитектурной импровизации.

## Состояние реализации (`2026-03-28`)

Первичная реализация уже вынесена в отдельные модули:

- [refinement_handoff.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/posthoc/refinement_handoff.py)
- [final_decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/posthoc/final_decision.py)

Что уже реализовано:

- отдельная handoff policy между `coarse` и `refinement`;
- отдельный final routing frame builder;
- отдельные stage-scoring helpers для `coarse` и `refinement family`;
- отдельный bridge для объединения реальных выходов стадий во вход итоговой
  маршрутизации;
- отдельный загрузчик сохраненных артефактов для `OOD/coarse/refinement/host`;
- отдельный контракт пороговой политики для `ID/OOD`-гейта;
- отдельный верхнеуровневый runner поверх сохраненных артефактов;
- отдельный слой сохранения итоговых артефактов решения;
- отдельный priority input adapter поверх final decision contract;
- отдельный модуль интеграции приоритета поверх объяснимого слоя ранжирования;
- отдельная `decide` CLI-команда поверх сохраненных артефактов;
- quality-first и `OOD`-first routing order зафиксирован в коде;
- `candidate_ood` не схлопывается с `id`;
- `O` остается coarse-only через refinement enable-list;
- refinement failure переводится в `unknown`, а не в forced subclass;
- `priority_state` больше не протекает из upstream frame раньше priority stage.

Рабочая оркестрация поверх реальных выходов стадий теперь закрыта на уровне
таблиц и кадров:

- bridge уже умеет принимать реальные stage outputs;
- отдельный decision runner уже собирает `OOD -> coarse -> refinement -> final decision`;
- priority integration считается отдельным слоем после final routing;
- верхнеуровневый runner/CLI теперь работает поверх сохраненных модельных
  артефактов и отдельной пороговой политики.

Для `ID/OOD` это принципиально важно:

- tuned threshold не растворяется в CLI flags;
- threshold policy хранится отдельно от estimator artifact;
- ноутбук и сквозной прогон читают один и тот же версионированный контракт
  порогов.

Проверка по начальному срезу decision-layer:

- `ruff` ok
- `mypy` ok
- `pyright` ok
- targeted `pytest` ok
