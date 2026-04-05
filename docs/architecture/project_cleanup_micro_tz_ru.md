# Micro-TZ На Уборку Проекта

Связанные документы:

- [project_cleanup_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/project_cleanup_tz_ru.md)
- [project_cleanup_audit_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/project_cleanup_audit_ru.md)

## Текущее Состояние

- `MTZ-C01` закрыт:
  - [src/exohost/db/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/db/__init__.py)
    перестал быть широким фасадом;
  - package-level импорты `from exohost.db import ...` в активном коде не
    использовались;
  - DB-пакет оставлен минимальным, без широкого `re-export`.
- `MTZ-C02` закрыт:
  - [bmk_labeled.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/db/bmk_labeled.py)
    стал тонким orchestration-слоем;
  - labeled-материализация разделена на отдельные модули:
    - contracts
    - SQL builders
    - export
    - validation
    - statistics;
  - тесты распилены по тем же границам ответственности, и для COUNT-helper
    добавлен отдельный regression-тест.
- `MTZ-C03` закрыт:
  - [bmk_parser_sync.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/db/bmk_parser_sync.py)
    стал тонким entrypoint-слоем;
  - downstream sync разделен на отдельные модули:
    - contracts
    - SQL builders
    - scalar helpers
    - validation
    - execution;
  - тесты распилены по тем же границам ответственности:
    - SQL
    - scalar helpers
    - validation
    - execution.
- `MTZ-C04` закрыт:
  - [final_decision_review.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/reporting/final_decision_review.py)
    стал тонким публичным API для notebook-слоя;
  - review-логика разделена на отдельные модули:
    - contracts
    - bundle loading
    - distributions
    - priority review
    - star-level review;
  - большой test-файл распилен на отдельные проверки bundle, distributions,
    priority и star-level слоя;
  - для shared test-data введен отдельный testkit без копипаста.
- `MTZ-C05` закрыт:
  - [model_pipeline_review.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/reporting/model_pipeline_review.py)
    стал тонким публичным API для pipeline notebook-слоя;
  - review-логика разделена на отдельные модули:
    - contracts
    - bundle loading
    - summary
    - stage observability frames
    - artifact summaries;
  - большой test-файл распилен на отдельные проверки bundle/summary, stage
    observability и artifact summaries;
  - для shared test-data введен отдельный testkit.
- `MTZ-C06` закрыт:
  - [hierarchical_training_frame.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/features/hierarchical_training_frame.py)
    стал тонким публичным API для hierarchical feature-слоя;
  - логика разделена на отдельные модули:
    - contracts
    - common helpers
    - coarse frame
    - refinement frame
    - ID/OOD frame;
  - test-файл распилен на отдельные проверки coarse, refinement и ID/OOD.
- `MTZ-C07` закрыт:
  - [priority_score.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/ranking/priority_score.py)
    стал тонким публичным API для ranking-слоя;
  - логика разделена на отдельные модули:
    - contracts
    - scalar helpers
    - score rules
    - ranking frame;
  - большой test-файл распилен на отдельные проверки rules, frame и
    scalar helpers, а общие test-data вынесены в отдельный testkit.
- `MTZ-C08` закрыт:
  - дерево [tests/unit](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit)
    перестроено по доменным подпакетам;
  - shared testkit-модули перенесены в соответствующие домены;
  - структура тестов сохранена без поломки `pytest`, `mypy` и `pyright`.
- `MTZ-C09` закрыт:
  - пустой слой [tests/contract](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/contract)
    удален;
  - активный тестовый контур больше не содержит фиктивного contract-каталога.
- `MTZ-C10` закрыт:
  - архивный слой [tests/archive_research](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/archive_research)
    явно выведен из активного `pytest`-контура;
  - в архиве зафиксирована собственная README-политика.
- `MTZ-C11` закрыт:
  - в активном `src` не осталось файлов без верхней шапки;
  - в активных тестах выровнены оставшиеся отсутствующие шапки.
- `MTZ-C12` закрыт:
  - smoke-роль `test_analysis_notebooks.py` зафиксирована явно;
  - список notebook для scoped `nbclient` QA вынесен в отдельную политику и
    синхронизирован с README notebook-слоя.
- `MTZ-C13` закрыт:
  - общая политика архива исследований вынесена в отдельный документ;
  - archive README-файлы синхронизированы между кодом, notebook, тестами и
    документами.
- `MTZ-C14` закрыт:
  - `docs/methodology` перестроен по роль-каталогам;
  - корневая и внутренняя README-навигация добавлена;
  - абсолютные ссылки на документы обновлены по репозиторию.
- `MTZ-C15` закрыт:
  - `.gitignore` дополнен локальными правилами рабочего дерева;
  - политика рабочего дерева вынесена в отдельный документ;
  - бытовой мусор из активного контура вычищен.
- Следующий рабочий шаг:
  - `MTZ-C16` — разобрать `requirements-v2.txt`.

## Общий Инвариант Для Всех Шагов

Для каждого шага обязательны:

- `1 файл = 1 ответственность`
- простая реализация раньше сложной
- `PEP 8`
- явная типизация
- сверка с официальной документацией Python и библиотек
- без скрытых фасадов и лишних слоев совместимости

После каждого микро-шага:

- `ruff`
- точечный `mypy`
- точечный `pyright`
- targeted `pytest`
- ручная проверка:
  - не вырос ли новый монолит
  - упростился ли код
  - не появилась ли лишняя абстракция

После завершения модуля или пакета:

- scoped big-QA только по затронутому слою

## Порядок Работы

### MTZ-C01. Зафиксировать публичную политику DB-пакета

- Цель: определить, нужен ли широкий фасад [db/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/db/__init__.py) вообще.
- Что делаем:
  - разбираем текущие импорты;
  - определяем минимальный публичный набор;
  - решаем, что остается в пакете, а что идет в прямые импорты.
- Результат:
  - понятная политика публичного DB API.
- Проверки:
  - `ruff`
  - `mypy`
  - `pyright`
  - targeted `pytest` по DB-слою
- Критерий готовности:
  - `db/__init__.py` перестает быть перегруженным центральным фасадом.

### MTZ-C02. Распилить `bmk_labeled.py`

- Цель: убрать из одного файла весь жизненный цикл labeled-материализации.
- Что разделяем:
  - schema/DDL
  - source-query
  - export
  - copy/load
  - count/statistics
- Результат:
  - несколько узких DB-модулей вместо одного большого.
- Тесты:
  - распилить [test_db_bmk_labeled.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/test_db_bmk_labeled.py) по тем же границам ответственности.
- Критерий готовности:
  - labeled-слой читается как набор самостоятельных операций.

### MTZ-C03. Распилить `bmk_parser_sync.py`

- Цель: развести синхронизацию parser-полей и пересборку summary-таблиц.
- Что разделяем:
  - update sync
  - validation
  - summary refresh
  - low-level scalar helpers
- Результат:
  - отдельные DB-модули на sync и summary.
- Тесты:
  - привести в порядок [test_db_bmk_parser_sync.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/test_db_bmk_parser_sync.py).

### MTZ-C04. Распилить `final_decision_review.py`

- Цель: убрать большой review-комбайн.
- Что разделяем:
  - bundle loading
  - summary tables
  - distribution frames
  - priority review
  - star-level result tables
- Результат:
  - notebook review-слой собирается из нескольких узких модулей.
- Тесты:
  - распилить [test_final_decision_review.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/test_final_decision_review.py).

### MTZ-C05. Распилить `model_pipeline_review.py`

- Цель: развести pipeline summary, metric selection и artifact summaries.
- Что разделяем:
  - bundle loading
  - summary row selection
  - stage overview
  - model artifact summary
  - threshold artifact summary
- Тесты:
  - распилить [test_model_pipeline_review.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/test_model_pipeline_review.py).

### MTZ-C06. Распилить `hierarchical_training_frame.py`

- Цель: разнести подготовку `coarse`, `refinement` и `ID/OOD`.
- Что разделяем:
  - общие scalar helpers
  - `coarse` frame
  - `refinement` frame
  - `ID/OOD` frame
  - схлопывание multi-membership
- Тесты:
  - распилить [test_hierarchical_training_frame.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/test_hierarchical_training_frame.py).

### MTZ-C07. Распилить `priority_score.py`

- Цель: отделить contracts, scalar coercion, explainable reason и ranking-frame builder.
- Что разделяем:
  - thresholds/weights contracts
  - low-level scalar conversion
  - score computation
  - reason builder
  - frame builder
- Тесты:
  - привести в порядок [test_priority_score.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/test_priority_score.py).

### MTZ-C08. Перестроить дерево `tests/unit`

- Цель: убрать плоскую свалку тестов в одном каталоге.
- Что делаем:
  - вводим подпапки по доменам:
    - `db`
    - `cli`
    - `reporting`
    - `posthoc`
    - `training`
    - `datasets`
    - `features`
    - `models`
    - `evaluation`
    - `contracts`
  - обновляем `pytest`-путь так, чтобы collection не ломался.
- Критерий готовности:
  - по дереву тестов сразу видно, что они страхуют.

### MTZ-C09. Разобраться с `tests/contract`

- Цель: убрать пустой и подвешенный слой.
- Варианты:
  - наполнить реальными contract-тестами;
  - или удалить как фиктивную структуру.
- Критерий готовности:
  - в проекте нет пустого тестового слоя “на будущее”.

### MTZ-C10. Привести в порядок архивные тесты

- Цель: перестать смешивать архив и активное тестовое дерево.
- Что делаем:
  - определяем, должны ли архивные тесты вообще жить под `tests/`;
  - если да, то явно помечаем их как неисполняемый архив;
  - если нет, выносим в исследовательский архив вне активного тестового контура.

### MTZ-C11. Выровнять шапки активных файлов и тестов

- Цель: сделать вход в проект проще.
- Что делаем:
  - добавляем содержательные шапки в активные рабочие файлы;
  - добавляем отсутствующие шапки в тесты;
  - проверяем, что шапка объясняет роль файла в цепочке.

### MTZ-C12. Зафиксировать политику notebook QA

- Цель: четко разделить:
  - синтаксический smoke
  - scoped execution важных notebook
- Что делаем:
  - описываем роль [test_analysis_notebooks.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/test_analysis_notebooks.py);
  - фиксируем, какие notebook должны прогоняться через `nbclient` в scoped QA.

### MTZ-C13. Зафиксировать политику архива исследований

- Цель: определить, что считается:
  - активным кодом
  - активной документацией
  - активными notebooks
  - архивом исследования
- Что делаем:
  - приводим ссылки и структуру к единой политике;
  - проверяем, не смешаны ли active и archive слои.

### MTZ-C14. Перестроить `docs/methodology`

- Цель: сделать документацию читаемой по ролям.
- Что делаем:
  - группируем документы по типам:
    - contracts
    - plans
    - run reviews
    - stabilization
    - archive research
- Критерий готовности:
  - новый человек быстро понимает, где норма, где история, где архив.

### MTZ-C15. Довести `.gitignore` и политику рабочего дерева

- Цель: убрать бытовой мусор и зафиксировать правила хранения.
- Что делаем:
  - убираем `.DS_Store`, `__pycache__`, лишние локальные артефакты;
  - пересматриваем `.gitignore`;
  - решаем, должен ли архив исследования быть версионируемым.

### MTZ-C16. Разобрать `requirements-v2.txt`

- Цель: решить, оставляем ли единый манифест или делим зависимости по ролям.
- Что делаем:
  - сравниваем текущий состав с фактическими слоями проекта;
  - фиксируем решение:
    - один файл остается
    - или появляются отдельные dependency groups.
- Статус:
  - закрыто;
  - пакетные зависимости вынесены в `requirements-runtime-v2.txt`;
  - `requirements-v2.txt` оставлен как полный локальный манифест;
  - политика зафиксирована в
    [dependency_policy_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/dependency_policy_ru.md).

## Рекомендуемый Порядок Выполнения

1. `MTZ-C01`
2. `MTZ-C02`
3. `MTZ-C03`
4. `MTZ-C04`
5. `MTZ-C05`
6. `MTZ-C06`
7. `MTZ-C07`
8. `MTZ-C08`
9. `MTZ-C09`
10. `MTZ-C10`
11. `MTZ-C11`
12. `MTZ-C12`
13. `MTZ-C13`
14. `MTZ-C14`
15. `MTZ-C15`
16. `MTZ-C16`

## Почему Такой Порядок

- сначала убираем самые опасные точки связности и крупные активные файлы;
- потом приводим в порядок тестовый слой;
- затем уже оформляем архив, документацию и политику рабочего дерева;
- так мы не ломаем живой контур и не распыляемся.
