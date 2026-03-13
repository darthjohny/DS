# ТЗ по следующим большим блокам: OOD/Unknown и baseline-comparison

Дата: 12 марта 2026 года

Статус документа:

- это historical planning document;
- production-блок `OOD/Unknown` уже реализован в `src/router_model/*`,
  `src/priority_pipeline/*` и связанных SQL migration;
- comparison-блок уже живёт в `analysis/model_comparison/` и описан
  канонически в [model_comparison_protocol_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/model_comparison_protocol_ru.md);
- этот файл полезен как история проектных решений, но не как основной
  источник текущего repo-state.

## 1. Цель документа

Зафиксировать следующий большой этап проекта как отдельное repo-ready ТЗ:

- сначала внедрить `OOD/Unknown` как production-contract;
- затем добавить baseline-блок для сравнений в ВКР;
- заранее учесть, какие артефакты, схемы, отчёты, notebook и QA
  изменятся после перехода к open-set поведению router.

Документ намеренно не содержит кода. Его задача — зафиксировать:

- архитектурный порядок работ;
- разбиение по файлам и пакетам;
- микро-ТЗ;
- карту внедрения;
- минимальный, но достаточный тестовый контур.

## 2. Ключевое решение по порядку

Рекомендуемый порядок:

1. Спроектировать и внедрить `OOD/Unknown`.
2. Обновить persist/schema, QA, notebook и отчёты под новую картину.
3. Только после этого строить baseline-framework и comparative report.

Причина:

- `OOD/Unknown` меняет label-contract router;
- меняет ветвление production pipeline;
- меняет `reason_code`, распределения классов и top-N отчёты;
- меняет notebook и графики;
- не проходит в текущие check constraints result-таблиц БД.

Если делать baseline раньше, часть сравнений и артефактов придётся
пересобирать после появления `UNKNOWN`.

## 3. Большой блок A: OOD/Unknown

### 3.1 Цель блока

Добавить в production router режим reject-option, при котором объект может
быть отнесён не к одному из известных физических классов, а к `UNKNOWN`.

### 3.2 Что должно получиться

- router умеет возвращать `UNKNOWN` для неуверенных или неполных объектов;
- `UNKNOWN` не попадает в host-ветку;
- pipeline корректно строит low-priority stub для unknown-объектов;
- БД принимает новые значения в result-таблицах;
- QA, calibration preview и notebook отражают долю unknown;
- README и docs описывают open-set поведение явно.

### 3.3 Предпочтительный scope первой волны

Первая волна `OOD/Unknown` должна быть минимальной и прагматичной:

- не вводить новые DB-колонки без явной необходимости;
- использовать уже существующие diagnostics:
  `router_log_posterior`, `posterior_margin`, `router_similarity`,
  `second_best_label`, missing features;
- кодировать неизвестный объект через уже существующие поля:
  `predicted_spec_class='UNKNOWN'`,
  `predicted_evolution_stage='unknown'`,
  `router_label='UNKNOWN'`,
  `priority_tier='LOW'`,
  `reason_code='ROUTER_UNKNOWN'` или близкий по смыслу код.

Важно:

- даже без новых колонок SQL migration всё равно нужна,
  потому что текущие DB constraints разрешают только
  `A/B/F/G/K/M/O` и `dwarf/evolved`.

### 3.4 Целевая файловая карта блока OOD/Unknown

| Файл | Роль |
| --- | --- |
| `docs/ood_unknown_tz_ru.md` | Канонический документ по open-set контракту и порогам reject |
| `src/router_model/ood.py` | Правила reject-option, OOD decision logic и helper-функции |
| `src/router_model/artifacts.py` | Расширение metadata router-артефакта под OOD policy |
| `src/router_model/labels.py` | Нормализация и константы `UNKNOWN`/`unknown` |
| `src/router_model/score.py` | Связка `raw scoring -> OOD decision -> RouterScoreResult` |
| `src/router_model/cli.py` | Preview accepted vs unknown после retrain/score |
| `src/priority_pipeline/branching.py` | Разделение `host`, `low_known`, `unknown` без разрастания `decision.py` |
| `src/priority_pipeline/constants.py` | Новые reason-code и, при необходимости, новые значения контрактов |
| `src/priority_pipeline/pipeline.py` | Новое ветвление и сборка unknown-stub |
| `src/priority_pipeline/persist.py` | Проверка совместимости persist после изменения значений label |
| `src/decision_calibration/runtime.py` | Корректный учёт unknown-строк в base scoring preview |
| `src/decision_calibration/reporting.py` | Поля unknown-count/unknown-share в summary |
| `analysis/router_eda/open_set.py` | Дополнительные open-set diagnostics и графики |
| `sql/*_router_unknown_constraints.sql` | Migration для result-table constraints |

### 3.5 Что сознательно не делаем в первой волне

- не строим сложный OOD-detector вне router posterior;
- не добавляем отдельный ML-модуль под OOD;
- не меняем host-model ради unknown;
- не усложняем persist-схему без прямой необходимости.

## 4. Большой блок B: baseline-comparison

### 4.1 Цель блока

Закрыть требование ВКР по нескольким моделям и получить единый comparative
benchmark вокруг текущей `Gaussian V1`.

### 4.2 Что должно получиться

- есть единый protocol сравнения;
- есть `Baseline 1: Legacy Gaussian`;
- есть `Baseline 2: RandomForest`;
- есть единый отчёт с closed-set метриками;
- при желании позже можно добавить `MLP`, не ломая каркас.

### 4.3 Архитектурное решение для baseline-слоя

Baseline-логика должна жить не в production `src/`, а в отдельном
research-layer, чтобы:

- не засорять production pipeline экспериментальными моделями;
- держать сравнение воспроизводимым, но изолированным;
- не превращать `src/` в монолит из боевого и исследовательского кода.

Рекомендуемый пакет:

- `analysis/model_comparison/`

### 4.4 Целевая файловая карта baseline-блока

| Файл | Роль |
| --- | --- |
| `docs/model_comparison_protocol_ru.md` | Единый protocol сравнения, split, metrics, output contract |
| `analysis/model_comparison/__init__.py` | Публичный API исследовательского comparison-layer |
| `analysis/model_comparison/data.py` | Загрузка benchmark datasets и единый split |
| `analysis/model_comparison/protocol.py` | Конфигурация seeds, режимов сравнения и target-задач |
| `analysis/model_comparison/metrics.py` | Closed-set и, позже, open-set метрики |
| `analysis/model_comparison/legacy_gaussian.py` | Первый baseline |
| `analysis/model_comparison/random_forest.py` | Второй baseline |
| `analysis/model_comparison/reporting.py` | Сборка markdown/CSV/summary-таблиц |
| `analysis/model_comparison/cli.py` | Точка входа для запуска benchmark |
| `experiments/model_comparison/` | Артефакты сравнений для ВКР |

Отдельной волной, не в первом проходе:

- `analysis/model_comparison/mlp_baseline.py`

### 4.5 Как baseline соотносится с OOD

Чтобы не раздуть scope, сравнение лучше сразу разделить на два режима:

- `closed-set benchmark`:
  обязателен для всех моделей;
- `open-set benchmark`:
  обязателен в первой волне только для production Gaussian router с reject-option.

Иначе baseline-блок станет зависеть от OOD сложнее, чем это нужно для ВКР.

## 5. Какие артефакты и отчёты изменятся после OOD

После внедрения `OOD/Unknown` изменятся:

- result-таблицы БД и их constraints;
- значения `predicted_spec_class`, `predicted_evolution_stage`, `router_label`;
- `reason_code` в priority results;
- доля объектов, проходящих в host-ветку;
- распределения классов в router EDA;
- calibration preview и markdown summary;
- preprocessing/EDA notebooks, если они показывают class distribution;
- QA-отчёты и итоговые графики для ВКР.

Это нужно учитывать заранее, чтобы потом не править вслепую:

- `README.md`;
- `experiments/QA/*`;
- `notebooks/eda/00_data_extraction_and_preprocessing.ipynb`;
- будущие notebooks/model-comparison и презентационные графики.

## 6. Карта внедрения

### Этап 0. Заморозить контракт и scope

1. Зафиксировать общий design doc.
2. Определить closed-set и open-set режимы сравнения.
3. Решить, какие поля остаются без новых DB-колонок.

### Этап 1. OOD/Unknown design

1. Описать contract `UNKNOWN`.
2. Зафиксировать reject policy и diagnostics.
3. Спланировать SQL migration.

### Этап 2. OOD/Unknown implementation

1. Внедрить OOD logic в `router_model`.
2. Внедрить отдельный branching в `priority_pipeline`.
3. Поднять миграцию БД.
4. Обновить calibration/reporting.

### Этап 3. Sync артефактов и документации

1. Обновить README.
2. Обновить notebook и QA.
3. Обновить traceability docs.

### Этап 4. Baseline framework

1. Завести `analysis/model_comparison/`.
2. Зафиксировать protocol.
3. Реализовать общий reporting.

### Этап 5. Baseline models

1. `Legacy Gaussian`
2. `RandomForest`
3. Comparative report

### Этап 6. Optional next

1. `MLP`
2. baseline open-set wrappers, если останется время и смысл

## 7. Микро-ТЗ

### Микро-ТЗ 1

Создать `docs/ood_unknown_tz_ru.md`.

Критерий готовности:

- явно описано, когда router имеет право выдать `UNKNOWN`;
- описаны входные diagnostics;
- описано, какие поля и значения появляются в runtime/persist.

### Микро-ТЗ 2

Добавить `src/router_model/ood.py`.

Критерий готовности:

- OOD logic выделена из `score.py` в отдельный модуль;
- reject policy не размазана по нескольким местам;
- `score.py` остаётся читаемым и не превращается в монолит.

### Микро-ТЗ 3

Обновить router artifact contract.

Критерий готовности:

- metadata router-модели знает про OOD policy;
- сохранение и загрузка артефакта не ломаются;
- backward compatibility продумана явно.

### Микро-ТЗ 4

Выделить отдельный branching-layer в production pipeline.

Критерий готовности:

- появляется `src/priority_pipeline/branching.py`;
- `decision.py` остаётся модулем про факторы и scoring;
- unknown-объекты не попадают в host-scoring.

### Микро-ТЗ 5

Подготовить SQL migration под `UNKNOWN`.

Критерий готовности:

- result-таблицы принимают `UNKNOWN` и `unknown`;
- существующие DB integration tests можно адаптировать без ломки persist-контракта.

### Микро-ТЗ 6

Обновить calibration/reporting/QA после OOD.

Критерий готовности:

- отчёты считают unknown rows и unknown share;
- top-N кандидаты не смешиваются с unknown;
- QA и notebook больше не предполагают закрытый мир по умолчанию.

### Микро-ТЗ 7

Создать `docs/model_comparison_protocol_ru.md`.

Критерий готовности:

- зафиксированы split, random seed, метрики, режимы сравнения;
- явно разделены `closed-set` и `open-set`;
- baseline scope не конфликтует с OOD roadmap.

### Микро-ТЗ 8

Завести пакет `analysis/model_comparison/`.

Критерий готовности:

- есть `data.py`, `protocol.py`, `metrics.py`, `reporting.py`, `cli.py`;
- package не содержит production-зависимостей в стиле `persist=True`.

### Микро-ТЗ 9

Добавить `Legacy Gaussian`.

Критерий готовности:

- baseline работает на общем protocol;
- результаты воспроизводимы;
- отчёт строится в том же формате, что и для других моделей.

### Микро-ТЗ 10

Добавить `RandomForest`.

Критерий готовности:

- baseline работает на общем protocol;
- фиксирован `random_state`;
- есть summary по train/test.

### Микро-ТЗ 11

Подготовить лёгкий тестовый контур для обоих блоков.

Критерий готовности:

- unit/smoke-тесты есть, но без тяжёлого retrain на больших данных;
- покрыты runtime-контракты и reporting-контракты.

## 8. Минимальный тестовый контур

### Для OOD/Unknown

- unit-test: missing features -> `UNKNOWN`;
- unit-test: низкая уверенность -> `UNKNOWN`;
- unit-test: явный in-domain sample -> known class;
- pipeline unit-test: unknown не идёт в host-ветку;
- DB integration test: persist принимает `UNKNOWN` после migration.

### Для baseline

- smoke-test benchmark split/protocol;
- smoke-test `Legacy Gaussian`;
- smoke-test `RandomForest`;
- test детерминизма при фиксированном seed;
- test формата итогового отчёта.

## 9. Что не делать

На этом этапе не стоит:

- лезть сразу в `GMM/router mixture`;
- тащить baseline-модели в `src/`;
- делать тяжёлые e2e benchmark tests на больших выборках;
- раздувать OOD до отдельного большого subsystems раньше времени;
- смешивать closed-set и open-set результаты в один неразличимый отчёт.

## 10. Связанные документы

- [vkr_requirements_traceability_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/vkr_requirements_traceability_ru.md)
- [preprocessing_and_comparison_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/preprocessing_and_comparison_tz_ru.md)
- [preprocessing_pipeline_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/preprocessing_pipeline_ru.md)
- [ood_unknown_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/ood_unknown_tz_ru.md)
