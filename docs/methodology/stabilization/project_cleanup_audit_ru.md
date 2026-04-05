# Аудит Уборки Проекта

Дата аудита: `2026-04-04`

Следующие документы по этому аудиту:

- [project_cleanup_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/project_cleanup_tz_ru.md)
- [project_cleanup_micro_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/architecture/project_cleanup_micro_tz_ru.md)

## Контур проверки

Аудит проведен без внесения правок в рабочую логику проекта.

Проверено:

- структура `src/exohost`
- активные крупные модули
- дерево `tests`
- конфигурация `pyproject.toml`
- `.gitignore`
- архивные исследовательские слои
- глобальные проверки `ruff`, `mypy`, `pyright`, `pytest`

## Базовое состояние

Сильные стороны проекта на момент аудита:

- `mypy` проходит на всем `src` и `tests`
- `pyright` проходит на всем `src` и `tests`
- `pytest` проходит полностью: `318 passed`
- в активном коде не найдено `type: ignore`
- в активном коде не найдено `TODO/FIXME/XXX`
- у большинства активных файлов есть краткая верхняя шапка-комментарий

Это важный вывод: проект уже не выглядит как аварийный или разваливающийся. Основные проблемы сейчас не в корректности исполнения, а в поддерживаемости, структуре и понятности слоев.

## Findings

### 1. Пакет `db` превратился в перегруженный фасад

Файл: [src/exohost/db/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/db/__init__.py)

Проблема:

- файл разросся до `243` строк
- в нем очень широкий `re-export` большого числа констант, summary-типов и функций из разных подмодулей
- это уже не легкий пакетный вход, а центральный агрегатор
- именно этот файл сейчас ломает глобальный `ruff check src tests` по `I001`

Риск:

- растет связность между DB-подмодулями
- сложнее понимать, что реально является публичным API DB-слоя
- повышается шанс скрытых циклических зависимостей и неочевидных импортов

Что поправить потом:

- решить, нужен ли вообще такой широкий фасад
- если нужен, сократить его до действительно публичного минимума
- если не нужен, переводить вызовы на прямые импорты из конкретных модулей

### 2. В активном коде есть несколько модулей, которые уже просятся на распил

Файлы:

- [src/exohost/db/bmk_labeled.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/db/bmk_labeled.py) — `582` строк
- [src/exohost/reporting/final_decision_review.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/reporting/final_decision_review.py) — `556` строк
- [src/exohost/features/hierarchical_training_frame.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/features/hierarchical_training_frame.py) — `386` строк
- [src/exohost/reporting/model_pipeline_review.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/reporting/model_pipeline_review.py) — `327` строк
- [src/exohost/db/bmk_parser_sync.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/db/bmk_parser_sync.py) — `325` строк
- [src/exohost/ranking/priority_score.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/ranking/priority_score.py) — `332` строк

Проблема:

- это не аварийные монолиты на тысячу строк, но для вашего стандарта `1 файл = 1 ответственность` они уже слишком тяжелые
- внутри каждого из этих файлов смешано несколько уровней логики

Что именно смешано:

- `bmk_labeled.py`: схема relation, SQL source-query, CSV export, COPY-load, валидация входных колонок, подсчет статистик
- `final_decision_review.py`: загрузка bundle, сводка run-а, распределения, таблицы причин, анализ `priority`, итоговые таблицы по объектам
- `hierarchical_training_frame.py`: подготовка `coarse`, `refinement`, `ID/OOD`, нормализация значений, схлопывание многократной `OOD`-принадлежности
- `model_pipeline_review.py`: benchmark review, stage overview, summary по model artifacts, summary по threshold artifacts
- `bmk_parser_sync.py`: синхронизация parser-derived полей и пересборка трех summary-таблиц
- `priority_score.py`: контракты порогов и весов, scalar coercion, построение объяснения, scoring одной строки и scoring всего frame

Что поправить потом:

- распиливать эти файлы на подмодули по ответственности
- у каждого распила держать отдельный тестовый слой, а не тащить старые крупные тесты целиком

### 3. Шапки в файлах есть, но в большинстве случаев они слишком короткие

Примеры:

- [src/exohost/db/bmk_labeled.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/db/bmk_labeled.py)
- [src/exohost/reporting/final_decision_review.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/reporting/final_decision_review.py)
- [src/exohost/ranking/priority_score.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/ranking/priority_score.py)

Проблема:

- верхний комментарий обычно отвечает только на вопрос “что это за файл”
- но почти не отвечает на вопросы:
  - зачем он нужен в цепочке
  - где его вход
  - что у него на выходе
  - какой модуль по логике программы идет следующим

Это не баг, но это мешает входу в код и поддержке проекта новыми людьми.

Что поправить потом:

- для активных модулей ввести более содержательную шапку:
  - роль файла
  - место в потоке данных
  - вход
  - выход
  - ближайшие соседние файлы по логике программы

### 4. Тестовое дерево работает, но организационно уже перегружено

Состояние:

- `tests/unit`: `109` файлов
- `tests/integration`: `1` файл
- `tests/smoke`: `1` файл
- `tests/contract`: `0` файлов
- `tests/archive_research`: `7` файлов

Проблема:

- почти весь тестовый слой лежит в одном плоском каталоге `tests/unit`
- по именам тестов видно, что там смешаны:
  - DB
  - CLI
  - модели
  - posthoc
  - reporting
  - datasets
  - features
  - notebooks
- `tests/contract` существует, но пустой
- `tests/archive_research` лежит внутри `tests`, но файлы там не начинаются с `test_`, поэтому pytest их не собирает

Риск:

- тесты тяжело искать глазами
- сложно понять, какой слой проекта страхуется, а какой нет
- наличие неисполняемого архива внутри `tests` визуально путает активное покрытие и архив

Что поправить потом:

- ввести подпапки в `tests/unit` хотя бы по доменам:
  - `db`
  - `cli`
  - `reporting`
  - `posthoc`
  - `training`
  - `datasets`
  - `features`
  - `models`
- решить судьбу `tests/contract`:
  - либо наполнить
  - либо удалить как пустой слой
- архивные тесты держать вне активного тестового дерева или явно пометить, что это неисполняемый архив

### 5. Несколько крупных тестовых файлов сами стали кандидатами на распил

Файлы:

- [tests/unit/test_bmk_catalog_parser.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/test_bmk_catalog_parser.py) — `473` строки
- [tests/unit/test_db_bmk_labeled.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/test_db_bmk_labeled.py) — `323` строки
- [tests/unit/test_cli_decide.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/test_cli_decide.py) — `306` строк
- [tests/unit/test_model_pipeline_review.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/test_model_pipeline_review.py) — `246` строк
- [tests/unit/test_host_calibration_review.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/test_host_calibration_review.py) — `220` строк

Проблема:

- часть тестов уже зеркалит разросшиеся рабочие модули
- в таких файлах смешиваются разные сценарии, фикстуры и уровни проверки

Что поправить потом:

- распиливать тесты вместе с рабочими файлами
- не держать один большой “контейнерный” тест на весь модуль

### 6. В части тестов отсутствует верхняя шапка

Файлы:

- [tests/unit/test_coarse_secure_o_reference_comparison.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/test_coarse_secure_o_reference_comparison.py)
- [tests/unit/test_coarse_secure_o_tail_review.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/test_coarse_secure_o_tail_review.py)
- [tests/unit/test_db_bmk_parser_sync.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/test_db_bmk_parser_sync.py)
- [tests/unit/test_db_coarse_ob_boundary_policy.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/test_db_coarse_ob_boundary_policy.py)
- [tests/unit/test_db_coarse_ob_provenance_refresh.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/test_db_coarse_ob_provenance_refresh.py)
- [tests/unit/test_db_coarse_ob_review_pool.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/test_db_coarse_ob_review_pool.py)

Проблема:

- это не ломает тесты, но нарушает выбранный стиль проекта
- на фоне остальных тестов эти файлы выглядят неравномерно оформленными

### 7. Покрытие тестами в целом хорошее, но его труднее читать, чем должно быть

Наблюдение:

- прямое сопоставление “модуль -> test_с тем же stem” не работает для значительной части кода
- это не значит, что покрытия нет
- но значит, что naming scheme неоднородна

Примеры:

- [src/exohost/posthoc/calibration.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/posthoc/calibration.py) покрывается файлом [tests/unit/test_posthoc_calibration.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/test_posthoc_calibration.py)
- [src/exohost/db/bmk_labeled.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/db/bmk_labeled.py) покрывается файлом [tests/unit/test_db_bmk_labeled.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/test_db_bmk_labeled.py)

Проблема:

- покрытие есть, но читаемость связи “код -> тест” для нового человека хуже, чем могла бы быть

Что поправить потом:

- выбрать единый шаблон именования для активного слоя
- особенно для `db`, `posthoc`, `cli`

### 8. Проверка ноутбуков в тестах сейчас очень поверхностная

Файл:

- [tests/unit/test_analysis_notebooks.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/test_analysis_notebooks.py)

Что он делает сейчас:

- проверяет, что notebooks валидны как JSON
- проверяет, что кодовые ячейки компилируются

Что он не делает:

- не проверяет выполнение notebook
- не проверяет expected output shape
- не разделяет активные и архивные notebook по уровню важности

Замечание:

- это полезный smoke
- но не стоит воспринимать его как полноценную гарантию корректности review-ноутбуков

Что поправить потом:

- для ключевых активных notebook оставить `nbclient`-исполнение в scoped QA
- в тестовом слое явно описать, что `test_analysis_notebooks.py` — только syntactic smoke

### 9. Архивный слой отделен, но политика его хранения пока не доведена до конца

Архивные каталоги:

- [src/exohost/reporting/archive_research](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/reporting/archive_research)
- [src/exohost/datasets/archive_research](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/datasets/archive_research)
- [analysis/notebooks/archive_research](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/archive_research)
- [tests/archive_research](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/archive_research)
- [docs/methodology/archive_research](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research)

Плюс:

- архив реально вынесен из активного слоя
- активные notebook и active reporting уже не засорены этими расследованиями

Минус:

- архив все еще живет внутри основных деревьев проекта
- ссылки на архив уже есть в активных документах и индексах
- при желании убрать архив из git это затронет текущую навигацию по docs

Что поправить потом:

- принять явную политику:
  - архив остается версионируемым исследовательским доказательством
  - или архив выносится из основного репозитория
- не добавлять архив в `.gitignore` вслепую, пока активные документы на него ссылаются

### 10. В рабочем дереве есть бытовой файловый мусор

Найдено:

- `.DS_Store` в корне и подкаталогах
- `__pycache__` внутри `tests`

Замечание:

- `.gitignore` уже содержит глобальные правила для `.DS_Store`, `__pycache__`, `.ipynb_checkpoints`
- значит проблема сейчас не в настройке ignore, а в фактической уборке рабочего дерева

Что поправить потом:

- периодически чистить рабочее дерево от системного мусора
- проверить, не попало ли что-то подобное в индекс git в основной истории

### 11. `.gitignore` покрывает базовые случаи, но решение по архиву и исследовательским материалам пока не оформлено

Файл:

- [.gitignore](/Users/evgeniikuznetsov/Desktop/dspro-vkr/.gitignore)

Что уже хорошо:

- игнорируются venv, кэши, артефакты, `.DS_Store`, `.ipynb_checkpoints`, обработанные данные

Что требует решения:

- архивные исследовательские слои сейчас не игнорируются
- при этом пользовательский процесс уже явно отделяет их от активного контура

Что поправить потом:

- не просто дописывать ignore-правила
- сначала решить политику:
  - архив остается частью воспроизводимого исследования
  - или архив не должен жить в git

Если архив решено держать вне git, тогда уже добавлять точные правила для:

- `analysis/notebooks/archive_research/`
- `docs/methodology/archive_research/`
- `src/exohost/reporting/archive_research/`
- `src/exohost/datasets/archive_research/`
- `tests/archive_research/`

### 12. Документация разрослась и уже требует активного/исторического разделения не только по архиву исследований

Каталог:

- [docs/methodology](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology)

Наблюдение:

- в каталоге много полезных документов
- но одновременно там накопилось много round-документов, реестров, планов и логов стабилизации

Риск:

- дальше станет трудно понять, какой документ нормативный, а какой исторический

Что поправить потом:

- ввести внутри `docs/methodology` более явные группы:
  - `contracts`
  - `plans`
  - `run_reviews`
  - `stabilization`
  - `archive_research`

### 13. Зависимости проекта собраны в один манифест, что удобно для старта, но не идеально для поддержки

Файл:

- [requirements-v2.txt](/Users/evgeniikuznetsov/Desktop/dspro-vkr/requirements-v2.txt)

Сейчас в одном файле живут:

- runtime зависимости
- notebook/analysis зависимости
- API-зависимости
- QA-инструменты

Проблема:

- для исследовательской стадии это терпимо
- для дальнейшей поддержки и воспроизводимости окружений это уже не лучший вариант

Что поправить потом:

- подумать о разделении хотя бы на:
  - `runtime`
  - `analysis`
  - `qa`

## Проверки, которые дали положительный результат

Это не задачи на исправление, а сильные стороны текущего состояния:

- глобальный `mypy` проходит
- глобальный `pyright` проходит
- глобальный `pytest` проходит
- в активном коде нет `type: ignore`
- в активном коде нет явных `TODO/FIXME`
- архивные исследования уже вынесены из активного слоя
- `reporting/__init__.py` оставлен минимальным и не превращен в еще один жирный фасад

## Приоритет работ на уборку

### Первый приоритет

- сократить или убрать перегруженный фасад [src/exohost/db/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/db/__init__.py)
- распилить самые тяжелые активные модули:
  - [src/exohost/db/bmk_labeled.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/db/bmk_labeled.py)
  - [src/exohost/reporting/final_decision_review.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/reporting/final_decision_review.py)
  - [src/exohost/features/hierarchical_training_frame.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/features/hierarchical_training_frame.py)
  - [src/exohost/reporting/model_pipeline_review.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/reporting/model_pipeline_review.py)
  - [src/exohost/db/bmk_parser_sync.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/db/bmk_parser_sync.py)
- привести в порядок дерево тестов и отделить архивные тесты от активного слоя

### Второй приоритет

- распилить крупные тестовые файлы
- выровнять шапки в тестах и рабочих модулях
- решить судьбу `tests/contract`
- уточнить политику notebook-smoke против notebook-execution

### Третий приоритет

- навести порядок в `docs/methodology`
- зафиксировать политику хранения архива
- разделить зависимости на более явные группы

## Короткий итог

Проект сейчас уже находится в хорошем техническом состоянии по корректности и типизации. Главный долг сидит не в сломанных модулях, а в поддерживаемости:

- несколько активных файлов стали слишком широкими по ответственности
- тестовый слой перерос плоское дерево
- архивные и исторические материалы уже выделены, но еще не доведены до окончательной политики хранения

Это хороший момент для большой уборки: база уже стабильна, значит можно спокойно наводить структуру без аварийного ремонта логики.
