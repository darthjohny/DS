# Микро-ТЗ На Регресс-Тестирование

Связанные документы:

- [regression_testing_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/regression_testing_tz_ru.md)
- [project_cleanup_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/project_cleanup_tz_ru.md)
- [pre_battle_tuning_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/pre_battle_tuning_tz_ru.md)

## Общий Инвариант Для Всех Шагов

Для каждого шага:

- `1 файл = 1 ответственность`;
- все шапки и комментарии на русском;
- опора на официальную документацию `pytest`, `pandas` и Python;
- явная типизация;
- без больших opaque snapshot-файлов;
- проверяем поведение, а не случайный шум.

После каждого шага:

- `ruff`
- targeted `mypy`
- targeted `pyright`
- targeted `pytest`

## Порядок Работы

### RTZ-REG01. Зафиксировать политику регресс-слоя

- Цель: формально определить, что считаем регрессом в проекте.
- Что делаем:
  - фиксируем роли `unit / integration / smoke / regression`;
  - фиксируем, что живет в `tests/regression`;
  - фиксируем, что в регресс не включаем архив и живую БД.
- Результат:
  - документ политики и явный контракт слоя.
- Статус:
  - закрыто.

### RTZ-REG02. Создать каркас `tests/regression`

- Цель: завести новый слой в дереве тестов.
- Что делаем:
  - создаем:
    - `tests/regression/__init__.py`
    - `tests/regression/conftest.py`
    - доменные подпапки:
      - `decision`
      - `posthoc`
      - `reporting`
      - `fixtures`
- Результат:
  - новый слой не смешан с `unit`.
- Статус:
  - закрыто.

### RTZ-REG03. Подготовить frozen fixtures

- Цель: не тащить живую БД в регресс.
- Что делаем:
  - выделяем маленькие входные CSV/JSON для:
    - `quality_gate`
    - `priority`
    - `decide`
  - делаем их компактными и читаемыми;
  - документируем источник и смысл каждой fixture.
- Важно:
  - fixtures должны быть малыми и стабильными.
- Статус:
  - закрыто.

### RTZ-REG04. Сделать общие regression-helpers

- Цель: убрать копипаст и хрупкие ручные сравнения.
- Что делаем:
  - при необходимости вводим маленькие helpers:
    - для чтения frozen fixtures;
    - для проверки обязательных колонок;
    - для scalar/dataframe asserts.
- Важно:
  - helpers только общие и минимальные.
- Статус:
  - закрыто.

### RTZ-REG05. Написать регресс на `quality_gate`

- Файл:
  - `tests/regression/posthoc/test_quality_gate_policy_regression.py`
- Проверяем:
  - `reject_missing_core_features` остается hard reject;
  - tuned policy меняет только разрешенный слой;
  - `quality_reason` и `review_bucket` не теряют согласованность.
- Статус:
  - закрыто.

### RTZ-REG06. Написать регресс на `priority`

- Файл:
  - `tests/regression/posthoc/test_priority_policy_regression.py`
- Проверяем:
  - tuned thresholds дают ожидаемое сжатие `high`;
  - `high / medium / low` остаются согласованными;
  - ranking-contract не ломается.
- Статус:
  - закрыто.

### RTZ-REG07. Написать малый `decide` roundtrip regression

- Файл:
  - `tests/regression/decision/test_decide_roundtrip_regression.py`
- Проверяем:
  - на frozen input создается полный run bundle;
  - присутствуют обязательные output files;
  - ключевые колонки и labels допустимы.
- Статус:
  - закрыто.

### RTZ-REG08. Написать регресс на artifact schema

- Файл:
  - `tests/regression/decision/test_decision_artifact_schema_regression.py`
- Проверяем:
  - `metadata.json`;
  - `final_decision.csv`;
  - `priority_ranking.csv`;
  - обязательные поля и допустимые значения.
- Статус:
  - закрыто.

### RTZ-REG09. Написать регресс на high-priority cohort

- Файл:
  - `tests/regression/decision/test_high_priority_cohort_regression.py`
- Проверяем:
  - cohort непустой;
  - у cohort есть базовые host-like признаки;
  - ключевые summary-колонки сохраняются.
- Статус:
  - закрыто.

### RTZ-REG10. Написать регресс на review summary

- Файл:
  - `tests/regression/reporting/test_final_decision_summary_regression.py`
- Проверяем:
  - итоговые сводки не теряют базовые причины и группы;
  - технический review-слой остается совместим с notebook/docs.
- Статус:
  - закрыто.

### RTZ-REG11. Привязать регресс-слой к `pyproject.toml` и README

- Цель: сделать новый тестовый слой видимым и понятным.
- Что делаем:
  - проверяем collection;
  - при необходимости уточняем `pytest`-конфиг;
  - добавляем упоминание нового слоя в README тестов/проекта.
- Статус:
  - закрыто.

### RTZ-REG12. Зафиксировать QA-политику регресс-слоя

- Цель: определить, как этот слой запускать дальше.
- Что делаем:
  - прописываем:
    - быстрый локальный запуск;
    - scoped запуск по доменам;
    - когда гонять весь regression suite.
- Статус:
  - закрыто.

## Предлагаемое Дерево Исполнения

```text
Блок A. Политика и каркас
├── RTZ-REG01
├── RTZ-REG02
├── RTZ-REG03
└── RTZ-REG04

Блок B. Поведенческий регресс posthoc
├── RTZ-REG05
└── RTZ-REG06

Блок C. Сквозной регресс decision
├── RTZ-REG07
├── RTZ-REG08
└── RTZ-REG09

Блок D. Review и интеграция в проект
├── RTZ-REG10
├── RTZ-REG11
└── RTZ-REG12
```

## Логика Выполнения

1. Сначала оформляем политику и каркас.
2. Потом готовим frozen fixtures.
3. Затем страхуем posthoc-политику.
4. После этого пишем малый end-to-end regression на `decide`.
5. Затем страхуем artifact schema и top cohort.
6. В конце вплетаем новый слой в общую документацию и QA-процедуру.

## Критерий Готовности

- в проекте есть отдельный `tests/regression`;
- регресс-слой запускается отдельно и не конфликтует с `unit`;
- ключевые системные инварианты `quality_gate`, `priority` и `decide`
  страхуются отдельными тестами;
- fixtures небольшие, понятные и документированные;
- новый слой проходит `ruff`, `mypy`, `pyright`, `pytest`;
- документация проекта явно знает про новый слой.
