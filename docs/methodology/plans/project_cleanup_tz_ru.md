# Большое ТЗ На Уборку И Доведение Проекта До Боевого Состояния

Дата фиксации: `2026-04-04`

Связанный аудит:

- [project_cleanup_audit_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/project_cleanup_audit_ru.md)

Связанный текущий stabilization-пакет:

- [post_run_stabilization_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/post_run_stabilization_tz_ru.md)

## Цель

Довести текущий проект до более зрелого инженерного состояния без смены научной
постановки и без хаотичной переделки логики.

Речь идет не о разработке новой волны пайплайна, а о системной уборке уже
собранного контура:

- сделать активный код проще и понятнее;
- убрать перегруженные фасады;
- распилить крупные файлы по ответственности;
- привести тестовое дерево к понятной структуре;
- отделить активный слой от архива исследований;
- довести документацию и шапки файлов до читаемого состояния;
- зафиксировать политику по notebook QA, архиву и `.gitignore`.

## Что Считаем Успехом

После закрытия этого пакета проект должен:

- проходить `ruff`, `mypy`, `pyright`, `pytest` на активном слое;
- иметь понятные публичные границы пакетов;
- не содержать тяжелых файлов, которые явно противоречат правилу
  `1 файл = 1 ответственность`;
- иметь тестовое дерево, в котором легко найти нужный слой;
- иметь четко отделенный архив исследовательских материалов;
- иметь активную документацию, в которой легко понять:
  - где нормативный документ;
  - где план;
  - где исторический разбор;
  - где архив.

## Что Не Входит В Этот Пакет

В этот пакет не входит:

- новая научная постановка;
- новое обучение моделей “на всякий случай”;
- изменение физической логики пайплайна без доказанного дефекта;
- удаление архивных материалов без заранее принятой политики;
- крупная переделка notebook-содержания ради косметики.

## Инженерный Инвариант Для Каждого Шага

Во все шаги этого пакета вшиваются одинаковые правила.

- `1 файл = 1 ответственность`
- без новых монолитов
- простое решение раньше сложного
- `PEP 8`
- явная типизация
- зависимости только по реальной необходимости
- код писать по официальной документации Python и библиотек
- шапка файла должна объяснять:
  - что делает файл;
  - зачем он нужен;
  - что у него на входе;
  - что у него на выходе;
  - какой соседний слой идет следующим
- комментарии только там, где без них реально хуже читать

После каждого небольшого куска:

- micro-QA
- `ruff`
- точечный `mypy`
- точечный `pyright`
- targeted `pytest`
- ручная проверка:
  - не усложнили ли код;
  - не размазали ли ответственность;
  - не сломали ли контракты соседнего слоя

После завершения микро-ТЗ:

- scoped big-QA только по затронутому слою

## Official Опора

### Python И Структура Кода

- [Python tutorial](https://docs.python.org/3/tutorial/)
- [typing](https://docs.python.org/3/library/typing.html)
- [collections.abc](https://docs.python.org/3/library/collections.abc.html)
- [pathlib](https://docs.python.org/3/library/pathlib.html)
- [argparse](https://docs.python.org/3/library/argparse.html)
- [dataclasses](https://docs.python.org/3/library/dataclasses.html)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)

### pytest И Проверка Кода

- [pytest good practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html)
- [pytest fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html)
- [pytest monkeypatch](https://docs.pytest.org/en/stable/how-to/monkeypatch.html)

### Типизация И Линтеры

- [mypy docs](https://mypy.readthedocs.io/en/stable/)
- [pyright docs](https://microsoft.github.io/pyright/#/)
- [Ruff docs](https://docs.astral.sh/ruff/)

### Ноутбуки И Исполнение

- [Jupyter docs](https://docs.jupyter.org/en/latest/)
- [nbclient docs](https://nbclient.readthedocs.io/en/latest/)

### Библиотеки Проекта

- [pandas user guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [SQLAlchemy 2.0 docs](https://docs.sqlalchemy.org/en/20/)
- [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html)

## Главные Проблемы, Которые Этот Пакет Должен Закрыть

### 1. Перегруженные активные файлы

Нужно распилить по ответственности:

- [src/exohost/db/bmk_labeled.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/db/bmk_labeled.py)
- [src/exohost/reporting/final_decision_review.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/reporting/final_decision_review.py)
- [src/exohost/features/hierarchical_training_frame.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/features/hierarchical_training_frame.py)
- [src/exohost/reporting/model_pipeline_review.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/reporting/model_pipeline_review.py)
- [src/exohost/db/bmk_parser_sync.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/db/bmk_parser_sync.py)
- [src/exohost/ranking/priority_score.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/ranking/priority_score.py)

### 2. Перегруженный фасад DB-пакета

Нужно привести в порядок:

- [src/exohost/db/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/db/__init__.py)

### 3. Плоское и тяжело читаемое тестовое дерево

Нужно привести в порядок:

- [tests](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests)

### 4. Крупные тестовые файлы

Нужно распилить по слоям проверки:

- [tests/unit/test_bmk_catalog_parser.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/test_bmk_catalog_parser.py)
- [tests/unit/test_db_bmk_labeled.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/test_db_bmk_labeled.py)
- [tests/unit/test_cli_decide.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/test_cli_decide.py)
- [tests/unit/test_model_pipeline_review.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/test_model_pipeline_review.py)
- [tests/unit/test_host_calibration_review.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/test_host_calibration_review.py)

### 5. Неоформленная политика архива исследований

Нужно принять решение по:

- [src/exohost/reporting/archive_research](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/reporting/archive_research)
- [src/exohost/datasets/archive_research](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/datasets/archive_research)
- [analysis/notebooks/archive_research](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/archive_research)
- [tests/archive_research](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/archive_research)
- [docs/methodology/archive_research](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research)

### 6. Не до конца оформленная политика документации и `.gitignore`

Нужно привести в порядок:

- [docs/methodology](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology)
- [.gitignore](/Users/evgeniikuznetsov/Desktop/dspro-vkr/.gitignore)

## Крупные Блоки Работ

### Блок A. Границы Пакетов И Фасады

Что делаем:

- приводим в порядок публичные входы пакетов;
- убираем избыточные фасады;
- фиксируем, что является публичным API слоя, а что нет.

Ожидаемый результат:

- понятные `__init__`
- меньше скрытых импортов
- меньше широких реэкспортов

### Блок B. Распил Крупных Активных Модулей

Что делаем:

- выносим из тяжелых файлов подслои:
  - contracts
  - loading
  - normalization
  - summaries
  - frame builders
  - scoring

Ожидаемый результат:

- файл не держит в себе весь жизненный цикл сразу
- тесты проще читать и поддерживать

### Блок C. Перестройка Тестового Дерева

Что делаем:

- вводим структуру внутри `tests/unit`
- отделяем активные тесты от архива
- решаем судьбу `tests/contract`

Ожидаемый результат:

- тесты ищутся по слою
- проще восстанавливать контекст
- видно, где активное покрытие, а где архив

### Блок D. Уборка Тестов И QA-политики

Что делаем:

- распиливаем тяжелые тестовые файлы
- выравниваем шапки тестов
- явно разделяем:
  - smoke
  - unit
  - integration
  - notebook syntax check
  - scoped notebook execution

### Блок E. Политика Архива И Документации

Что делаем:

- приводим к явному виду:
  - что считается активным документом;
  - что считается историческим документом;
  - что считается архивом исследования.

### Блок F. Политика Рабочего Дерева И Ignore

Что делаем:

- убираем бытовой мусор;
- проверяем, какие правила действительно нужны в `.gitignore`;
- не добавляем в ignore активные материалы без принятой политики.

## Критерий Завершения Большого Пакета

Большое ТЗ считается закрытым, когда:

- активный кодовый слой приведен к разумной модульности;
- главный фасадный долг снят;
- тестовое дерево стало структурированным;
- архив отделен и понятен;
- документы и `.gitignore` приведены к осмысленной политике;
- активный контур снова проходит:
  - `ruff`
  - `mypy`
  - `pyright`
- `pytest`

## Следующий Документ

Детальная разбивка на шаги вынесена отдельно:

Внутренний микро-план cleanup-пакета ведется вне публичного контура
репозитория.

## Текущее Продвижение

- Старт cleanup-пакета выполнен.
- `MTZ-C01` закрыт:
  - [src/exohost/db/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/db/__init__.py)
    сокращен до минимального package-marker без широкого `re-export`.
- `MTZ-C02` закрыт:
  - [bmk_labeled.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/db/bmk_labeled.py)
    разделен на узкие модули по ответственности;
  - крупный тест слоя распилен на отдельные проверки SQL, export, load и
    statistics;
  - типовая граница COUNT-запросов зафиксирована явной нормализацией значения
    и regression-тестом.
- `MTZ-C03` закрыт:
  - [bmk_parser_sync.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/db/bmk_parser_sync.py)
    разделен на отдельные слои contracts, SQL, validation, scalar helpers и
    execution;
  - CLI-слой сохранен без лишнего фасада;
  - тесты распилены по тем же границам ответственности, и orchestration слой
    получил отдельные unit-проверки commit/rollback поведения.
- `MTZ-C04` закрыт:
  - [final_decision_review.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/reporting/final_decision_review.py)
    превращен в тонкий публичный слой для notebook-review;
  - review-логика вынесена в отдельные модули bundle, distributions, priority и
    star-level;
  - тесты распилены по тем же границам ответственности, а общие test-data
    вынесены в отдельный testkit.
- `MTZ-C05` закрыт:
  - [model_pipeline_review.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/reporting/model_pipeline_review.py)
    превращен в тонкий публичный слой для pipeline notebook-review;
  - review-логика вынесена в отдельные модули bundle, summary, stage
    observability и artifact summaries;
  - тесты распилены по тем же границам ответственности, а общие test-data
    вынесены в отдельный testkit.
- `MTZ-C06` закрыт:
  - [hierarchical_training_frame.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/features/hierarchical_training_frame.py)
    превращен в тонкий публичный слой для hierarchical feature-подготовки;
  - feature-логика вынесена в отдельные модули contracts, common helpers,
    coarse, refinement и ID/OOD;
  - тесты распилены по тем же границам ответственности.
- `MTZ-C07` закрыт:
  - [priority_score.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/ranking/priority_score.py)
    стал тонким публичным слоем для ranking-score API;
  - ranking-логика вынесена в отдельные модули contracts, scalar helpers,
    rules и frame builder;
  - тесты распилены на отдельные проверки scalar helpers, score rules и
    ranking frame, а общие test-data вынесены в отдельный testkit.
- `MTZ-C08` закрыт:
  - дерево [tests/unit](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit)
    перестроено по доменам:
    `cli`, `contracts`, `datasets`, `db`, `evaluation`, `features`,
    `ingestion`, `labels`, `models`, `notebooks`, `posthoc`, `ranking`,
    `reporting`, `training`;
  - shared testkit-модули перенесены в соответствующие доменные подпакеты;
  - структура тестов больше не является плоской свалкой, а `pytest`
    collection сохранен без дополнительных костылей в конфигурации.
- `MTZ-C09` закрыт:
  - пустой слой `tests/contract` удален как фиктивная структура “на
    будущее”;
  - активный тестовый контур теперь не содержит подвешенного contract-слоя,
    дублирующего [tests/unit/contracts](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/contracts).
- `MTZ-C10` закрыт:
  - в [pyproject.toml](/Users/evgeniikuznetsov/Desktop/dspro-vkr/pyproject.toml)
    зафиксирована явная политика `pytest` для архивного слоя через
    `norecursedirs = ["archive_research"]`;
  - в [tests/archive_research/README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/archive_research/README.md)
    описано, что архивные проверки не входят в активный контур качества и
    служат только исследовательским следом.
- `MTZ-C11` закрыт:
  - в активном [src/exohost](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost)
    не осталось файлов без верхней шапки;
  - в активном test-слое выровнены оставшиеся файлы без содержательных
    шапок, прежде всего в `db` и `reporting`.
- `MTZ-C12` закрыт:
  - создан отдельный документ
    [notebook_qa_policy_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/notebook_qa_policy_ru.md)
    с явным разделением между smoke-проверкой notebook и адресным
    исполнением через `nbclient`;
  - роль
    [test_analysis_notebooks.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/notebooks/test_analysis_notebooks.py)
    и правила notebook QA синхронизированы с
    [analysis/notebooks/README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/README.md).
- `MTZ-C13` закрыт:
  - создан отдельный документ
    [archive_research_policy_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/archive_research_policy_ru.md)
    с общей политикой архива исследований;
  - archive README-файлы в `tests`, `analysis/notebooks`, `src/exohost/reporting`
    и `src/exohost/datasets` синхронизированы с этой политикой;
  - подтверждено, что активный код не импортирует архивный слой напрямую.
- `MTZ-C14` закрыт:
  - [docs/methodology](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology)
    перестроен по ролям:
    `contracts`, `plans`, `run_reviews`, `stabilization`, `archive_research`;
  - создана корневая навигация в
    [docs/methodology/README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/README.md)
    и README-файлы для новых роль-каталогов;
  - абсолютные ссылки по репозиторию перепривязаны на новую структуру.
- `MTZ-C15` закрыт:
  - [.gitignore](/Users/evgeniikuznetsov/Desktop/dspro-vkr/.gitignore)
    дополнен правилами для локальной IDE-обвязки, временных каталогов и
    coverage-артефактов;
  - создан документ
    [working_tree_policy_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/working_tree_policy_ru.md)
    с явной политикой versioned и локального рабочего дерева;
  - бытовой мусор из активного контура вычищен: `.DS_Store`, `__pycache__`,
    `*.pyc` и кэши проверок.
- `MTZ-C16` закрыт:
  - пакетные runtime-зависимости вынесены в
    [requirements-runtime-v2.txt](/Users/evgeniikuznetsov/Desktop/dspro-vkr/requirements-runtime-v2.txt),
    а [pyproject.toml](/Users/evgeniikuznetsov/Desktop/dspro-vkr/pyproject.toml)
    больше не тянет локальный dev/notebook-слой в `project.dependencies`;
  - [requirements-v2.txt](/Users/evgeniikuznetsov/Desktop/dspro-vkr/requirements-v2.txt)
    оставлен как полный локальный манифест для runtime, notebooks и QA;
  - удалены неиспользуемые активным кодом и QA-зависимости:
    `fastapi`, `uvicorn`, `astroquery`, `nbconvert`;
  - добавлена явная политика зависимостей:
    [dependency_policy_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/dependency_policy_ru.md).
