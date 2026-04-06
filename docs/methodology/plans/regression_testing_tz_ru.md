# Большое ТЗ На Регресс-Тестирование Проекта

Дата фиксации: `2026-04-06`

Связанные документы:

- [project_cleanup_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/project_cleanup_tz_ru.md)
- [pre_battle_tuning_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/pre_battle_tuning_tz_ru.md)
- [file_header_policy_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/file_header_policy_ru.md)
- [notebook_qa_policy_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/notebook_qa_policy_ru.md)

## Зачем Нужен Этот Пакет

Сейчас у проекта уже сильный `unit`-слой, есть `smoke` и есть один
`integration`-тест. Этого достаточно, чтобы хорошо страховать отдельные
функции, небольшие модули и часть сквозных путей.

Но между этими слоями остается зазор:

- `unit` подтверждает локальную корректность;
- `smoke` подтверждает, что пакет стартует;
- а отдельного слоя, который страхует поведение системы после правок, почти нет.

Именно этот зазор и должен закрыть регресс-пакет.

## Цель

Построить понятный и воспроизводимый слой регресс-тестирования, который:

- проверяет не только отдельные функции, но и ожидаемое поведение системы;
- страхует ключевые инварианты `quality_gate`, `priority`, `decide` и artifact
  bundle;
- работает на небольших замороженных входах, а не на живой БД;
- остается быстрым, читаемым и поддерживаемым;
- не дублирует `unit`, а дополняет его.

## Что Считаем Успехом

После завершения этого пакета проект должен иметь отдельный регресс-слой:

- [tests/regression](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/regression)

Этот слой должен:

- запускаться отдельно от `unit` и `smoke`;
- использовать небольшие стабильные fixtures;
- проверять ключевые инварианты pipeline;
- хранить тесты по доменам, а не одной кучей;
- иметь понятные шапки и комментарии на русском;
- проходить `ruff`, `mypy`, `pyright`, `pytest`.

## Что Не Входит В Пакет

В этот пакет не входит:

- полный боевой прогон на всей рабочей БД;
- нагрузочное и performance-тестирование;
- переписывание активной бизнес-логики без подтвержденной пользы;
- замена `unit`-тестов snapshot-ами;
- автоматическое сравнение больших `.csv`-артефактов целиком;
- проверка архивного исследовательского слоя.

## Инженерный Инвариант

Для каждого шага пакета действуют те же правила, что и для остального проекта:

- `1 файл = 1 ответственность`;
- простое решение раньше сложного;
- явная типизация;
- опора на официальную документацию;
- без неявных магических фикстур;
- без хрупких giant-snapshot тестов;
- шапки файлов и комментарии на русском;
- после каждого изменения:
  - `ruff`
  - targeted `mypy`
  - targeted `pyright`
  - targeted `pytest`

## Official Опора

### pytest

- [pytest good practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html)
- [pytest fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html)
- [pytest tmp_path](https://docs.pytest.org/en/stable/how-to/tmp_path.html)
- [pytest parametrization](https://docs.pytest.org/en/stable/example/parametrize.html)

### Python И Типизация

- [Python typing](https://docs.python.org/3/library/typing.html)
- [Python pathlib](https://docs.python.org/3/library/pathlib.html)
- [Python json](https://docs.python.org/3/library/json.html)

### pandas И DataFrame-проверки

- [pandas.testing.assert_frame_equal](https://pandas.pydata.org/docs/reference/api/pandas.testing.assert_frame_equal.html)

## На Что Смотрели В Похожих Проектах

### scikit-learn

Что берем себе:

- tests лежат рядом с доменами и страхуют поведение по слоям;
- для bug-fix и важных изменений ожидаются non-regression tests;
- performance-регрессии держатся отдельно от обычного тестового слоя.

Источник:

- [scikit-learn contributing](https://sklearn.org/1.8/developers/contributing.html)

### pandas

Что берем себе:

- тесты лучше писать до или одновременно с изменением поведения;
- тесты должны жить рядом с соответствующим слоем;
- для `DataFrame` и `Series` нужно использовать профильные assert-хелперы, а не
  ручные сравнения.

Источник:

- [pandas contributing](https://pandas.pydata.org/pandas-docs/version/2.1/development/contributing_codebase.html)

### scikit-bio

Что берем себе:

- отдельный test-файл на модуль или на четкий аспект модуля;
- маленькие проверяемые случаи важнее одной большой непрозрачной проверки;
- тестовое дерево должно помогать восстанавливать интерфейс и поведение.

Источник:

- [scikit-bio coding guide](https://scikit.bio/devdoc/code_guide.html)

## Каким Должен Быть Наш Регресс-Слой

Нам нужен не “слой тестов на все подряд”, а осмысленная прослойка между
`unit` и `smoke`.

### 1. Регресс На Поведение `quality_gate`

Что страхуем:

- hard reject не размывается;
- `review`-сигналы не путаются с `reject`;
- tuning policy меняет только ожидаемые состояния;
- инварианты `quality_reason` и `review_bucket` сохраняются.

### 2. Регресс На Поведение `priority`

Что страхуем:

- пороги `high / medium / low` работают по ожидаемой схеме;
- верхняя зона не возвращается к нечитабельному насыщению;
- `high priority` cohort сохраняет базовые host-like свойства.

### 3. Малый Сквозной Регресс `decide`

Что страхуем:

- небольшой фиксированный вход проходит полный pipeline;
- на выходе получаем ожидаемый набор artifact-файлов;
- сохраняются ключевые колонки и допустимые значения;
- не ломается связка `decision_input -> final_decision -> priority_ranking`.

### 4. Регресс На Артефакты

Что страхуем:

- структура `metadata.json`;
- обязательные файлы run bundle;
- обязательные колонки таблиц;
- допустимые классы, причины и policy labels.

### 5. Регресс На Ключевые Исследовательские Инварианты

Что страхуем:

- `OB boundary` не начинает тихо возвращаться в `O`;
- `high priority` cohort не расползается в шумную смесь;
- top cohort summary остается физически правдоподобным;
- базовые выводы исследования не ломаются неочевидной правкой.

## Предлагаемое Дерево Регресс-Слоя

```text
tests/regression/
├── __init__.py
├── conftest.py
├── fixtures/
│   ├── decide_input_small.csv
│   ├── quality_gate_small.csv
│   ├── priority_small.csv
│   └── expected/
│       ├── final_decision_columns.json
│       └── priority_labels.json
├── decision/
│   ├── test_decide_roundtrip_regression.py
│   ├── test_decision_artifact_schema_regression.py
│   └── test_high_priority_cohort_regression.py
├── posthoc/
│   ├── test_quality_gate_policy_regression.py
│   └── test_priority_policy_regression.py
└── reporting/
    └── test_final_decision_summary_regression.py
```

## Ответственности По Файлам

### `tests/regression/conftest.py`

Хранит только:

- общие frozen fixtures;
- helpers для чтения маленьких CSV/JSON;
- компактные scalar/dataframe asserts, если они реально нужны нескольким тестам.

Не хранит:

- бизнес-логику;
- полные snapshot-данные внутри Python-кода;
- большие generated bundles.

### `tests/regression/decision/test_decide_roundtrip_regression.py`

Проверяет:

- малый end-to-end `decide`;
- создание run-dir;
- наличие обязательных выходных файлов;
- допустимые значения ключевых колонок.

### `tests/regression/decision/test_decision_artifact_schema_regression.py`

Проверяет:

- схему `metadata.json`;
- обязательные columns в `final_decision.csv`;
- обязательные columns в `priority_ranking.csv`;
- согласованность policy-labels и run metadata.

### `tests/regression/decision/test_high_priority_cohort_regression.py`

Проверяет:

- верхний shortlist не распадается по структуре;
- high-priority cohort остается непустым;
- cohort summary держит базовые физические и ranking-инварианты.

### `tests/regression/posthoc/test_quality_gate_policy_regression.py`

Проверяет:

- baseline/tuned policy меняют только допустимую часть `quality_gate`;
- `reject_missing_core_features` не размывается;
- `review`-логика работает по согласованным ожиданиям.

### `tests/regression/posthoc/test_priority_policy_regression.py`

Проверяет:

- tuned thresholds дают ожидаемые переходы `high -> medium`;
- верхняя зона сжимается без разрушения ranking-контракта;
- high/medium/low labels остаются согласованными.

### `tests/regression/reporting/test_final_decision_summary_regression.py`

Проверяет:

- ключевые сводки review-слоя на frozen run-data;
- обязательные причины и группы не исчезают;
- пользовательский summary сохраняет форму, нужную notebook и docs.

## Принципы Реализации Регресса

### Не Делаем Хрупкие Exact Snapshot На Все Подряд

Плохо:

- сравнивать весь `final_decision.csv` целиком с большим эталоном;
- фиксировать каждое число до последнего знака;
- тащить большие живые артефакты в git.

Хорошо:

- сравнивать структуру и ключевые инварианты;
- на малых fixtures сравнивать ожидаемые таблицы через
  `assert_frame_equal`;
- где нужно, фиксировать небольшие expected-JSON/CSV.

### Отделяем Поведение От Косметики

Регресс-тест должен падать, если:

- сломалось поведение;
- пропал нужный файл;
- изменилась структура данных;
- съехал важный инвариант.

Регресс-тест не должен падать, если:

- поменялся порядок несущественных строк;
- изменилась неважная колонка вне контракта;
- поменялась косметика текста вне пользовательского контракта.

## Что Получим В Итоге

После внедрения этого слоя у проекта появится полноценная “система
страховки”:

- `unit` продолжат страховать локальную логику;
- `integration` останется для отдельных сквозных путей;
- `smoke` останется для минимального старта;
- `regression` начнет страховать поведение системы как продукта исследования.

Именно это сейчас является самым полезным следующим инженерным шагом.
