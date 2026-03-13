# Полный QA-аудит проекта

Дата старта: 13 марта 2026 года

Формат записей:
- `OK` — выглядит нормально, оставляем;
- `TOLERABLE` — неидеально, но пока можно жить;
- `FIX` — стоит править в отдельной волне;
- `REMOVE?` — кандидат на удаление, вынос из git или отдельное решение.

## Шаг 1. Инвентаризация репозитория и file hygiene

### Что проверено

- `git status --short`
- `git status --ignored --short`
- `.gitignore`
- tracked/untracked состав `src`, `analysis`, `tests`, `docs`, `notebooks`, `sql`, `experiments`
- размеры ключевых каталогов артефактов
- наличие служебного мусора `.DS_Store`, `__pycache__`, `*.pyc`

### Findings

#### QA-001

- Статус: `TOLERABLE`
- Зона: `repo state`
- Файл/каталог: весь репозиторий
- Наблюдение: аудит выполняется на грязном рабочем дереве с большим числом `modified` и `untracked` файлов.
- Почему это важно: findings по структуре и необходимости файлов нужно интерпретировать осторожно, потому что часть текущего состояния ещё не оформлена как каноническая версия репозитория.
- Рекомендация: до финального cleanup-решения считать текущий аудит аудитом актуального workspace, а не только последнего зафиксированного git-снимка.

#### QA-002

- Статус: `OK`
- Зона: `.gitignore`
- Файл/каталог: [.gitignore](/Users/evgeniikuznetsov/Desktop/dspro-vkr/.gitignore)
- Наблюдение: базовые ignore-правила для `venv`, `__pycache__`, `.pytest_cache`, `*.pyc`, `.DS_Store`, `.env` уже есть.
- Почему это важно: это снижает риск случайного коммита служебного мусора.
- Рекомендация: оставить как базу, но позже проверить, нужны ли дополнительные правила под новые экспериментальные артефакты.

#### QA-003

- Статус: `TOLERABLE`
- Зона: `workspace hygiene`
- Файл/каталог: `.DS_Store`, `__pycache__`, `*.pyc`, caches
- Наблюдение: в workspace много служебного мусора, но почти весь он корректно находится в ignored-состоянии и не tracked в git.
- Почему это важно: это не ломает репозиторий, но засоряет рабочее дерево и мешает визуальному обзору структуры.
- Рекомендация: позже выполнить отдельную cleanup-волну workspace-мусора, не смешивая её с кодовыми правками.

#### QA-004

- Статус: `FIX`
- Зона: `versioning policy`
- Файл/каталог: `analysis/model_comparison/`
- Наблюдение: целый новый пакет `analysis/model_comparison` сейчас находится в untracked-состоянии.
- Почему это важно: это уже не временный черновик, а существенная часть исследовательского слоя и comparison-контракта проекта.
- Рекомендация: отдельно решить и зафиксировать политику версионирования этого пакета; держать такую зону полностью untracked долго нельзя.

#### QA-005

- Статус: `FIX`
- Зона: `versioning policy`
- Файл/каталог: `tests/test_model_comparison_*.py`, `tests/test_decision_calibration_reporting.py`
- Наблюдение: новые важные тесты comparison-layer и decision calibration сейчас untracked.
- Почему это важно: без этого репозиторий не отражает фактическое покрытие проекта и теряет воспроизводимость QA.
- Рекомендация: после завершения аудита отдельно решить, какие из этих тестов становятся канонической частью репозитория.

#### QA-006

- Статус: `FIX`
- Зона: `reproducibility`
- Файл/каталог: `notebooks/eda/00_data_extraction_and_preprocessing.ipynb`, `notebooks/eda/04_model_comparison_summary.ipynb`
- Наблюдение: два важных notebook-файла untracked, хотя оба имеют методическую ценность для ВКР и воспроизводимости.
- Почему это важно: репозиторий в текущем виде не содержит полный набор реально используемых notebook-артефактов.
- Рекомендация: позже отдельно принять решение по их включению в version control.

#### QA-007

- Статус: `FIX`
- Зона: `reproducibility`
- Файл/каталог: `sql/adql/`, `sql/preprocessing/`, новые SQL-файлы `2026-03-13_*`
- Наблюдение: существенная часть SQL и ADQL-материалов untracked.
- Почему это важно: для data lineage и воспроизводимости preprocessing это важные артефакты, а не побочные файлы.
- Рекомендация: провести отдельное решение, какие SQL/ADQL входят в канонический контур проекта.

#### QA-008

- Статус: `TOLERABLE`
- Зона: `docs policy`
- Файл/каталог: `docs/`
- Наблюдение: в `docs` много новых untracked markdown-документов, включая protocol/findings/traceability/presentation.
- Почему это важно: с точки зрения QA это нормально, если документы ещё в работе, но долго оставлять такие ключевые документы вне version control не стоит.
- Рекомендация: на этапе doc audit отдельно разделить канонические документы, рабочие ТЗ и исторические заметки.

#### QA-009

- Статус: `TOLERABLE`
- Зона: `artifact policy`
- Файл/каталог: `experiments/model_comparison/`
- Наблюдение: каталог артефактов большой (`111M`) и целиком untracked.
- Почему это важно: это уже не мелкий побочный вывод; тут нужна явная policy — что хранится в git, что архивируется вне репозитория, что пересчитывается по команде.
- Рекомендация: не коммитить/не удалять вслепую; решить судьбу артефактов на отдельном шаге ревизии.

#### QA-010

- Статус: `TOLERABLE`
- Зона: `asset policy`
- Файл/каталог: `docs/assets/`, `docs/presentation/`
- Наблюдение: `docs/assets` весит около `9.6M`, `docs/presentation` уже содержит отдельный presentation-контур.
- Почему это важно: нужен критерий, какие медиа-ассеты являются частью документации проекта, а какие временные.
- Рекомендация: на doc audit отдельно проверить каждый asset на необходимость хранения в репозитории.

#### QA-011

- Статус: `OK`
- Зона: `tracked research logs`
- Файл/каталог: `experiments/Логи калибровки decision_layer`, `experiments/Логи работы программы`
- Наблюдение: исторические журналы и QA-отчёты уже частично лежат в репозитории и выглядят как осмысленная исследовательская история, а не мусор.
- Почему это важно: это полезно для traceability и ВКР-контекста.
- Рекомендация: позже проверить только единообразие и необходимость последних untracked итераций, не трогая историческое ядро автоматически.

### Итог шага

- Базовая ignore-гигиена настроена неплохо.
- Главный риск шага — не мусор, а отсутствие чёткой политики version control для новых кодовых, тестовых, notebook, SQL и experiment-артефактов.
- На следующем шаге нужно переходить к статическому QA кода, но помнить, что часть актуального состояния проекта пока находится вне tracked-дерева.

## Шаг 2. Базовый статический QA по всему Python-коду

### Что проверено

- `./venv/bin/python -m ruff check src analysis tests`
- `./venv/bin/python -m mypy src analysis tests`
- `./venv/bin/python -m compileall -q src analysis tests`
- попытка второго статического взгляда через `pyright`
- обзор `Any`, `cast`, `type: ignore`, `noqa`
- обзор самых длинных файлов и самых длинных функций

### Findings

#### QA-012

- Статус: `OK`
- Зона: `style + typing`
- Файл/каталог: `src`, `analysis`, `tests`
- Наблюдение: `ruff` проходит без ошибок, `mypy` проходит по `104` source files, `compileall` завершается успешно.
- Почему это важно: по формальным статическим критериям проект находится в хорошем состоянии и не имеет очевидных синтаксических или type-check regressions.
- Рекомендация: считать это хорошей базой, но не подменять этим более глубокий архитектурный и математический review.

#### QA-013

- Статус: `TOLERABLE`
- Зона: `tooling`
- Файл/каталог: Python QA toolchain
- Наблюдение: `pyright` как дополнительный статический взгляд в текущем `venv` недоступен (`No module named pyright`).
- Почему это важно: сейчас у проекта по факту один основной type gate — `mypy`; второго независимого типового взгляда нет.
- Рекомендация: не считать это проблемой кода, но позже решить, нужен ли `pyright` как постоянный инструмент QA.

#### QA-014

- Статус: `TOLERABLE`
- Зона: `typing boundaries`
- Файл/каталог: `src/priority_pipeline/decision.py`, `src/router_model/*`, `src/host_model/*`, `src/decision_calibration/*`, `analysis/*plots.py`, `analysis/*exports.py`
- Наблюдение: в проекте заметно много `Any` и `cast`, но они в основном сосредоточены на boundary-слоях: plotting, JSON/artifact normalization, pandas/scikit-learn integration и численные helper-ы.
- Почему это важно: это не автоматический баг, но такие зоны часто скрывают слабые контракты и нуждаются в ручном review.
- Рекомендация: в дальнейшем audit-е смотреть эти места как candidate zones для упрощения или усиления контрактов, но не фиксить механически только ради “меньше Any”.

#### QA-015

- Статус: `TOLERABLE`
- Зона: `lint exceptions`
- Файл/каталог: [src/eda.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/eda.py), [src/router_eda.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_eda.py), [src/model_comparison.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/model_comparison.py)
- Наблюдение: есть явные `ruff: noqa: E402`.
- Почему это важно: такие исключения часто оправданы во façade/CLI entrypoints, но их нужно проверить на реальную необходимость, а не принимать по умолчанию.
- Рекомендация: оставить для архитектурного review следующей волны.

#### QA-016

- Статус: `TOLERABLE`
- Зона: `large files`
- Файл/каталог: [src/input_layer.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/input_layer.py), [analysis/model_comparison/snapshot.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/snapshot.py), [analysis/model_comparison/presentation_assets.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/presentation_assets.py), [analysis/model_comparison/mlp_baseline.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/mlp_baseline.py)
- Наблюдение: есть несколько крупных модулей (`692`, `620`, `533`, `400` строк соответственно).
- Почему это важно: размер сам по себе не является дефектом, но такие файлы нужно внимательно читать на смешение ответственностей.
- Рекомендация: не дробить автоматически; проверить на следующих шагах, где длина оправдана, а где действительно скрывает архитектурный долг.

#### QA-017

- Статус: `TOLERABLE`
- Зона: `large functions`
- Файл/каталог: [src/input_layer.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/input_layer.py), [analysis/model_comparison/snapshot.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/snapshot.py), [src/router_model/fit.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/fit.py), [analysis/router_eda/readiness.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/router_eda/readiness.py), [analysis/host_eda/contrastive.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/host_eda/contrastive.py)
- Наблюдение: есть набор функций длиной `100+` строк и ряд функций в диапазоне `70-90` строк.
- Почему это важно: это уже не линтерная проблема, а потенциальный сигнал к перегруженным сценариям и слабой декомпозиции.
- Рекомендация: на file-by-file и architecture review оценивать их по критерию “работает и оправдано” против “можно проще и чище”.

#### QA-018

- Статус: `OK`
- Зона: `static baseline quality`
- Файл/каталог: весь Python-контур
- Наблюдение: на статическом уровне проект выглядит инженерно дисциплинированным; основная дальнейшая работа будет не про исправление банальных ошибок, а про качество решений.
- Почему это важно: QA можно вести прагматично и не тратить время на косметику там, где automation уже всё закрывает.
- Рекомендация: на следующих шагах смещать фокус с style на архитектуру, математику и нужность файлов.

### Итог шага

- Формально статический контур проекта зелёный.
- Основные риски статического уровня — не ошибки линтера, а boundary typing, длинные функции и несколько крупных модулей.
- Следующий шаг логично посвятить dependency audit и среде, чтобы понять, насколько текущая зелёная картина воспроизводима и минимальна по стеку.

## Шаг 3. Dependency audit и среда

### Что проверено

- [requirements.txt](/Users/evgeniikuznetsov/Desktop/dspro-vkr/requirements.txt)
- [pyproject.toml](/Users/evgeniikuznetsov/Desktop/dspro-vkr/pyproject.toml)
- [python_version.txt](/Users/evgeniikuznetsov/Desktop/dspro-vkr/python_version.txt)
- [pyrightconfig.json](/Users/evgeniikuznetsov/Desktop/dspro-vkr/pyrightconfig.json)
- фактические top-level third-party imports в `src`, `analysis`, `tests`

### Findings

#### QA-019

- Статус: `OK`
- Зона: `python target`
- Файл/каталог: [python_version.txt](/Users/evgeniikuznetsov/Desktop/dspro-vkr/python_version.txt), [pyproject.toml](/Users/evgeniikuznetsov/Desktop/dspro-vkr/pyproject.toml), текущее `venv`
- Наблюдение: Python target согласован — `3.13` в toolchain и `3.13.2` в рабочем окружении.
- Почему это важно: нет расхождения между declared target и фактической средой.
- Рекомендация: оставить как сильную сторону проекта.

#### QA-020

- Статус: `FIX`
- Зона: `dependency manifest`
- Файл/каталог: [requirements.txt](/Users/evgeniikuznetsov/Desktop/dspro-vkr/requirements.txt)
- Наблюдение: `requirements.txt` выглядит как полный freeze локального окружения, а не как минимальный манифест проекта. Там есть большой Jupyter-стек, транзитивные пакеты и platform-specific утилиты вроде `appnope`.
- Почему это важно: такой файл хуже объясняет реальные зависимости проекта, усложняет воспроизводимость и затрудняет понимание “что проекту действительно нужно”.
- Рекомендация: позже отдельно решить, нужен ли разделённый dependency policy: runtime / dev / notebook, либо оставить один файл, но осознанно.

#### QA-021

- Статус: `OK`
- Зона: `effective dependency core`
- Файл/каталог: весь Python-код
- Наблюдение: фактическое ядро импортов заметно уже, чем `requirements.txt`: в коде реально доминируют `numpy`, `pandas`, `sqlalchemy`, `scikit-learn`, `matplotlib`, `seaborn`, `pytest`.
- Почему это важно: сам проект не выглядит зависимостно перегруженным по коду; перегружен скорее manifest, а не runtime-архитектура.
- Рекомендация: при будущей чистке не переусложнять dependency management, а просто привести manifest в соответствие реальному использованию.

#### QA-022

- Статус: `TOLERABLE`
- Зона: `tooling consistency`
- Файл/каталог: [pyrightconfig.json](/Users/evgeniikuznetsov/Desktop/dspro-vkr/pyrightconfig.json), `venv`
- Наблюдение: конфиг `pyright` существует, но сам инструмент не установлен в текущем окружении.
- Почему это важно: есть небольшой дрейф между declared tooling и реально доступным tooling.
- Рекомендация: позже решить одно из двух — либо использовать `pyright` по-настоящему, либо не считать его частью активного toolchain.

#### QA-023

- Статус: `TOLERABLE`
- Зона: `mypy strictness`
- Файл/каталог: [pyproject.toml](/Users/evgeniikuznetsov/Desktop/dspro-vkr/pyproject.toml)
- Наблюдение: `mypy` настроен аккуратно, но с `ignore_missing_imports = true`, что делает типовой gate мягче.
- Почему это важно: текущая зелёная типизация не означает максимально строгую типизацию внешних границ.
- Рекомендация: не считать это дефектом само по себе; при дальнейшем audit-е просто помнить, что часть проблем могла быть скрыта этой настройкой.

#### QA-024

- Статус: `OK`
- Зона: `project tooling`
- Файл/каталог: [pyproject.toml](/Users/evgeniikuznetsov/Desktop/dspro-vkr/pyproject.toml)
- Наблюдение: основные repo-tools (`ruff`, `mypy`, `pytest`) описаны просто и без лишней сложности.
- Почему это важно: базовая QA-инфраструктура проекта не выглядит перегруженной или “магической”.
- Рекомендация: оставить текущий подход как прагматичный базовый контур.

### Итог шага

- Ядро зависимостей проекта выглядит разумным.
- Основной вопрос этого шага — не избыток библиотек в коде, а неаккуратный характер `requirements.txt` как полного freeze-списка окружения.
- Следующий шаг логично посвятить архитектурному аудиту production-контура.

## Шаг 4. Архитектурный аудит production-контура

### Что проверено

- публичные API пакетов:
  - [src/router_model/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/__init__.py)
  - [src/host_model/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/host_model/__init__.py)
  - [src/priority_pipeline/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/__init__.py)
  - [src/decision_calibration/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/decision_calibration/__init__.py)
- верхнеуровневые фасады совместимости:
  - [src/gaussian_router.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/gaussian_router.py)
  - [src/model_gaussian.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/model_gaussian.py)
  - [src/decision_layer_calibrator.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/decision_layer_calibrator.py)
  - [src/star_orchestrator.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/star_orchestrator.py)
- внутренние import-зависимости production-пакетов
- спорные модульные границы:
  - [src/priority_pipeline/decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/decision.py)
  - [src/priority_pipeline/input_data.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/input_data.py)
  - [src/router_model/score.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/score.py)
  - [src/host_model/db.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/host_model/db.py)

### Findings

#### QA-025

- Статус: `OK`
- Зона: `package architecture`
- Файл/каталог: `src/router_model`, `src/host_model`, `src/priority_pipeline`, `src/decision_calibration`
- Наблюдение: production-контур уже собран вокруг нормальных пакетов с явным публичным API, а не вокруг россыпи скриптов.
- Почему это важно: архитектура читается, масштабируется и уже отделяет доменные зоны достаточно хорошо.
- Рекомендация: считать это сильной стороной проекта и не ломать без необходимости.

#### QA-026

- Статус: `OK`
- Зона: `compatibility facade policy`
- Файл/каталог: [src/gaussian_router.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/gaussian_router.py), [src/model_gaussian.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/model_gaussian.py), [src/decision_layer_calibrator.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/decision_layer_calibrator.py)
- Наблюдение: фасады совместимости явно подписаны как фасады и не пытаются притворяться канонической реализацией.
- Почему это важно: это снижает архитектурную путаницу и соответствует заявленному стандарту документации.
- Рекомендация: оставить этот подход; он прагматичный и понятный.

#### QA-027

- Статус: `TOLERABLE`
- Зона: `entrypoint surface`
- Файл/каталог: верхний уровень `src/*.py`
- Наблюдение: в корне `src` остаётся довольно много старых entrypoint/facade-файлов.
- Почему это важно: это увеличивает число импортных поверхностей и когнитивную нагрузку, особенно для нового читателя проекта.
- Рекомендация: не убирать автоматически, но позже явно разделить “канонические пакеты” и “legacy convenience entrypoints” в README и архитектурном описании.

#### QA-028

- Статус: `FIX`
- Зона: `layering`
- Файл/каталог: [src/priority_pipeline/decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/decision.py), [src/priority_pipeline/input_data.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/input_data.py)
- Наблюдение: decision-layer helper-ы зависят от `ensure_decision_columns()` из `input_data`.
- Почему это важно: это разворачивает зависимость не в идеальную сторону — чистый scoring/helper слой начинает тянуть кусок IO/data-preparation слоя.
- Рекомендация: позже отдельно оценить, не стоит ли этот контракт вынести в более нейтральный модуль или в `contracts/helpers`.

#### QA-029

- Статус: `FIX`
- Зона: `layering`
- Файл/каталог: [src/host_model/db.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/host_model/db.py)
- Наблюдение: DB-layer использует `prepare_contrastive_training_df()` из `host_model.fit`.
- Почему это важно: загрузка данных оказывается связана с training/preparation-слоем, а не только с нейтральной валидацией/нормализацией.
- Рекомендация: при дальнейшем review оценить, не лучше ли выделить shared preparation/validation helper отдельно от fit-модуля.

#### QA-030

- Статус: `TOLERABLE`
- Зона: `facade purity`
- Файл/каталог: [src/star_orchestrator.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/star_orchestrator.py)
- Наблюдение: это не чистый re-export facade; в нём живёт совместимый wrapper `run_host_similarity()` с собственной рабочей логикой.
- Почему это важно: compatibility layer перестаёт быть полностью прозрачным и сохраняет часть поведения вне канонического пакета.
- Рекомендация: не считать это аварией, но держать в голове как архитектурный долг совместимости.

#### QA-031

- Статус: `TOLERABLE`
- Зона: `package coupling`
- Файл/каталог: [src/router_model/score.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/score.py)
- Наблюдение: scoring-layer использует `FEATURES` из `router_model.db`.
- Почему это важно: для чистой архитектуры scoring обычно хочется отвязать от DB-модуля, даже если это всего лишь shared constant.
- Рекомендация: отдельно оценить, нужен ли вынос таких shared constants в более нейтральный слой; не трогать, если польза окажется минимальной.

#### QA-032

- Статус: `OK`
- Зона: `production vs analysis isolation`
- Файл/каталог: production-пакеты `src/*`
- Наблюдение: прямого протекания `analysis.*` внутрь production-контура по архитектурной карте не видно.
- Почему это важно: это хороший признак зрелой границы между боевым и исследовательским слоями.
- Рекомендация: сохранить это как жёсткий architectural rule.

#### QA-033

- Статус: `OK`
- Зона: `reuse strategy`
- Файл/каталог: [src/decision_calibration/*](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/decision_calibration)
- Наблюдение: пакет калибровки переиспользует production helpers, а не копирует их.
- Почему это важно: это лучше для консистентности и уменьшает риск расхождения логики.
- Рекомендация: оставить текущий подход; он оправдан.

### Итог шага

- Архитектура production-контура в целом сильная и не похожа на хаотичный набор скриптов.
- Основные спорные места — несколько неидеальных направлений зависимостей и неполная “чистота” некоторых compatibility-слоёв.
- Следующий шаг логично посвятить математическому аудиту production scoring.

## Шаг 5. Математический аудит production scoring

### Что проверено

- [src/router_model/score.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/score.py)
- [src/router_model/ood.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/ood.py)
- [src/host_model/contrastive_score.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/host_model/contrastive_score.py)
- [src/host_model/legacy_score.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/host_model/legacy_score.py)
- [src/priority_pipeline/decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/decision.py)
- [src/decision_calibration/scoring.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/decision_calibration/scoring.py)

### Findings

#### QA-034

- Статус: `OK`
- Зона: `router math`
- Файл/каталог: [src/router_model/score.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/score.py)
- Наблюдение: router scoring реально posterior-aware и выбирает победителя по `router_log_posterior`, а не просто по минимальной Mahalanobis distance.
- Почему это важно: это более сильная и более честная математическая постановка для production-классификации.
- Рекомендация: считать это сильной стороной текущего production router.

#### QA-035

- Статус: `OK`
- Зона: `router OOD`
- Файл/каталог: [src/router_model/ood.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/ood.py)
- Наблюдение: open-set reject option вынесен в отдельный слой и сохраняет raw diagnostics для `UNKNOWN`, а не затирает их.
- Почему это важно: это хорошо и для физической интерпретации, и для QA/EDA rejected rows.
- Рекомендация: оставить текущую логику как удачную.

#### QA-036

- Статус: `TOLERABLE`
- Зона: `router assumptions`
- Файл/каталог: [src/router_model/score.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/score.py)
- Наблюдение: router ranking использует `uniform_log_prior(len(classes))`, то есть одинаковые priors по классам внутри router-слоя.
- Почему это важно: это не ошибка, но это модельное допущение; если физические priors по классам позже окажутся важны, именно здесь будет точка развития.
- Рекомендация: пока оставить как допустимое упрощение, не пытаться “улучшать” без отдельной научной причины.

#### QA-037

- Статус: `OK`
- Зона: `host scoring`
- Файл/каталог: [src/host_model/contrastive_score.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/host_model/contrastive_score.py)
- Наблюдение: production host-ветка математически выглядит когерентно: внутри routed class считаются `host` и `field` likelihood, затем `host_log_lr` и bounded `host_posterior`.
- Почему это важно: это соответствует реальной задаче проекта лучше, чем простой distance-only scoring.
- Рекомендация: считать contrastive host-path каноническим production math-контуром.

#### QA-038

- Статус: `OK`
- Зона: `legacy baseline role`
- Файл/каталог: [src/host_model/legacy_score.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/host_model/legacy_score.py)
- Наблюдение: legacy path явно остаётся distance/similarity baseline и не маскируется под основной scoring-контур.
- Почему это важно: архитектурно и методически роль legacy baseline определена честно.
- Рекомендация: оставить его как baseline/diagnostic слой, не смешивая с основной физикой проекта.

#### QA-039

- Статус: `TOLERABLE`
- Зона: `decision-layer formula`
- Файл/каталог: [src/priority_pipeline/decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/decision.py)
- Наблюдение: production `final_score` строится как мультипликативное сочетание `host_posterior` и нескольких soft factors (`class_prior`, `quality`, `metallicity`, `color`, `validation`).
- Почему это важно: сама формула выглядит осмысленной, но мультипликативные схемы всегда чувствительны к скрытой перекалибровке и к тому, насколько факторы действительно независимы.
- Рекомендация: пока не считать это дефектом; держать как зону обязательного sanity-check на top-ranked объектах и при future calibration.

#### QA-040

- Статус: `TOLERABLE`
- Зона: `quality vs distance semantics`
- Файл/каталог: [src/priority_pipeline/decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/decision.py), [src/decision_calibration/scoring.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/decision_calibration/scoring.py)
- Наблюдение: в production distance включён внутрь `quality_factor` как мягкий proxy через параллакс, а в calibration-контуре вынесен в отдельный `distance_factor`.
- Почему это важно: это не баг, но различие между production и offline calibration нужно держать под контролем и хорошо объяснять в документации.
- Рекомендация: позже проверить, достаточно ли это уже ясно отражено в docs и findings.

#### QA-041

- Статус: `OK`
- Зона: `low/unknown handling`
- Файл/каталог: [src/priority_pipeline/decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/decision.py), [src/decision_calibration/scoring.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/decision_calibration/scoring.py)
- Наблюдение: known-low и `UNKNOWN` ветки оформлены явно и не притворяются host-scored объектами; им присваивается нулевой итоговый score и понятный `reason_code`.
- Почему это важно: это хороший контракт и для production, и для QA, и для защиты.
- Рекомендация: оставить как есть.

#### QA-042

- Статус: `OK`
- Зона: `leakage risk`
- Файл/каталог: production scoring path
- Наблюдение: на уровне production scoring-кода не видно утечки train/test логики; это inference-only контур.
- Почему это важно: методически production path не смешивает обучение и инференс.
- Рекомендация: основной leakage review дальше смещать в research comparison-layer, а не в production scoring.

### Итог шага

- Явных математических дефектов в production scoring на этом уровне не видно.
- Основные вопросы здесь не про баги, а про осознанность допущений: uniform priors в router и мультипликативную природу decision layer.
- Следующий шаг логично посвятить архитектурному и методическому audit research-layer.

## Шаг 6. Архитектурный и методический аудит research-layer

### Что проверено

- [analysis/model_comparison/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/__init__.py)
- [analysis/model_comparison/cli.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/cli.py)
- [analysis/model_comparison/snapshot.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/snapshot.py)
- [analysis/host_eda/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/host_eda/__init__.py)
- [analysis/router_eda/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/router_eda/__init__.py)
- [analysis/host_eda/cli.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/host_eda/cli.py)
- [analysis/router_eda/cli.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/router_eda/cli.py)
- карта import-зависимостей `analysis/model_comparison`, `analysis/host_eda`, `analysis/router_eda`

### Findings

#### QA-043

- Статус: `OK`
- Зона: `research architecture`
- Файл/каталог: `analysis/model_comparison`
- Наблюдение: comparison-layer оформлен как отдельный пакет с явными submodules под contracts/data/tuning/manual_search/models/reporting/snapshot.
- Почему это важно: для исследовательского слоя это уже зрелая структура, а не временный набор скриптов.
- Рекомендация: считать его каноническим research-контуром benchmark-сравнения.

#### QA-044

- Статус: `OK`
- Зона: `research vs production reuse`
- Файл/каталог: `analysis/model_comparison/*`
- Наблюдение: comparison-layer переиспользует production math/scoring helpers, а не копирует их.
- Почему это важно: это хорошо для методической консистентности проекта.
- Рекомендация: сохранить этот подход как базовый принцип.

#### QA-045

- Статус: `FIX`
- Зона: `import side effects`
- Файл/каталог: [analysis/host_eda/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/host_eda/__init__.py), [analysis/router_eda/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/router_eda/__init__.py)
- Наблюдение: оба EDA-пакета модифицируют `sys.path` прямо при импорте.
- Почему это важно: это архитектурно хрупкий приём; он работает, но создаёт неявный побочный эффект на уровне package import.
- Рекомендация: не чинить автоматически прямо сейчас, но позже рассмотреть более чистый способ package resolution.

#### QA-046

- Статус: `TOLERABLE`
- Зона: `export surface`
- Файл/каталог: [analysis/model_comparison/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/__init__.py)
- Наблюдение: публичная re-export surface comparison-layer очень широкая.
- Почему это важно: это удобно для фасадного импорта, но увеличивает когнитивную нагрузку и стоимость поддержки package API.
- Рекомендация: пока оставить, если это реально помогает CLI и фасадам; не раздувать её дальше без необходимости.

#### QA-047

- Статус: `TOLERABLE`
- Зона: `snapshot complexity`
- Файл/каталог: [analysis/model_comparison/snapshot.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/snapshot.py)
- Наблюдение: snapshot-layer глубоко связан и с training wrappers, и с production pipeline, и с reporting.
- Почему это важно: это не ломает методику, но делает модуль самым очевидным кандидатом на future refactor внутри research-layer.
- Рекомендация: не дробить автоматически сейчас; держать как осознанный технический долг.

#### QA-048

- Статус: `TOLERABLE`
- Зона: `EDA CLI style`
- Файл/каталог: [analysis/host_eda/cli.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/host_eda/cli.py), [analysis/router_eda/cli.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/router_eda/cli.py)
- Наблюдение: EDA CLI-модули очень script-like и собирают полный end-to-end сценарий внутри одного `main()`.
- Почему это важно: для исследовательского EDA это допустимо, но такие файлы хуже масштабируются, если сценарии начнут резко расти.
- Рекомендация: пока оставить как pragmatic research entrypoints, если они не начинают тянуть доменную логику внутрь CLI.

#### QA-049

- Статус: `OK`
- Зона: `presentation tooling`
- Файл/каталог: [analysis/model_comparison/presentation_assets.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/presentation_assets.py)
- Наблюдение: presentation asset generator изолирован от production и не смешивает свою утилитарную задачу с benchmark math.
- Почему это важно: это хороший пример вспомогательного research-tooling без протекания в боевой контур.
- Рекомендация: оставить так, не превращая его в ещё один “универсальный framework”.

### Итог шага

- Research-layer в целом структурирован лучше, чем обычно бывает в аналитических контурах.
- Главные спорные места здесь — `sys.path`-магия в EDA-пакетах и тяжёлый snapshot-модуль.
- Дальше логично переходить к file-by-file review и тестовому coverage audit.

## Шаг 7. Полный file-by-file review Python-кода

### Что проверено

- построен отдельный пофайловый ledger для `src` и `analysis`:
  [qa_file_ledger_python_2026-03-13_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/QA/qa_file_ledger_python_2026-03-13_ru.md)
- дополнительно дочитаны ключевые модули orchestration/reporting/logbook-слоёв

### Findings

#### QA-050

- Статус: `OK`
- Зона: `overall codebase shape`
- Файл/каталог: `src`, `analysis`
- Наблюдение: пофайловый review подтвердил предыдущую картину: кодовая база в целом собрана осмысленно и не производит впечатление случайного набора файлов.
- Почему это важно: это снижает риск, что дальнейшие findings окажутся следствием тотального структурного хаоса.
- Рекомендация: продолжать аудит прагматично, не искать рефакторинг ради рефакторинга.

#### QA-051

- Статус: `FIX`
- Зона: `most notable code-level candidates`
- Файл/каталог: [src/host_model/db.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/host_model/db.py), [src/priority_pipeline/decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/decision.py), [analysis/host_eda/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/host_eda/__init__.py), [analysis/router_eda/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/router_eda/__init__.py), [analysis/model_comparison/snapshot.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/snapshot.py)
- Наблюдение: именно эти файлы сейчас являются наиболее явными кандидатами на будущие архитектурные правки.
- Почему это важно: это уже не “мелкая эстетика”, а точки, где можно получить реальное упрощение или улучшение границ.
- Рекомендация: использовать этот shortlist как основу будущей волны code fixes, если подтвердится после оставшихся шагов аудита.

#### QA-052

- Статус: `TOLERABLE`
- Зона: `large but acceptable modules`
- Файл/каталог: [src/input_layer.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/input_layer.py), [src/decision_calibration/reporting.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/decision_calibration/reporting.py), [analysis/model_comparison/mlp_baseline.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/mlp_baseline.py), [analysis/model_comparison/presentation_assets.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/presentation_assets.py)
- Наблюдение: есть набор крупных модулей, но не все они требуют рефакторинга прямо сейчас.
- Почему это важно: помогает не тратить время на бессмысленное “дробление ради красоты”.
- Рекомендация: считать эти модули допустимыми до появления реального pain-point.

### Итог шага

- Пофайловый review `src` и `analysis` выполнен и оформлен отдельным ledger.
- Картина остаётся прагматичной: большинство файлов можно оставлять как есть.
- Следующий шаг логично посвятить тестовому coverage audit и качеству проверок.

## Шаг 8. Аудит тестов и покрытие по рискам

### Что проверено

- полная коллекция `pytest`:
  `76 tests collected`
- список test-файлов и количество тестов по каждому файлу
- структура `conftest.py` и DB-backed fixtures
- карта прямых импортов `tests -> src`
- выборочное чтение ключевых тестов по зонам:
  - router/model gaussian
  - priority pipeline
  - input layer
  - decision calibration
  - model comparison
  - runtime artifacts
- отдельная матрица покрытия:
  [qa_test_coverage_matrix_2026-03-13_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/QA/qa_test_coverage_matrix_2026-03-13_ru.md)

### Findings

#### QA-053

- Статус: `OK`
- Зона: `overall test contour`
- Файл/каталог: `tests`
- Наблюдение: тестовый контур небольшой, но не декоративный; в нём есть unit-like проверки, smoke на orchestration, DB-backed интеграции и регрессии на runtime artifacts.
- Почему это важно: проект не выглядит как кодовая база “с тестами для галочки”.
- Рекомендация: сохранить текущий прагматичный стиль тестов без попытки искусственно раздувать suite.

#### QA-054

- Статус: `OK`
- Зона: `DB integration discipline`
- Файл/каталог: [tests/conftest.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/conftest.py), [tests/test_input_layer_db_integration.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/test_input_layer_db_integration.py), [tests/test_priority_pipeline_db_integration.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/test_priority_pipeline_db_integration.py)
- Наблюдение: DB-backed тесты используют отдельную временную схему, безопасную валидацию SQL-идентификаторов и корректно помечены marker-ом `db_integration`.
- Почему это важно: это хороший компромисс между реальной интеграционной проверкой и безопасностью локального окружения.
- Рекомендация: считать этот контур сильной стороной проекта.

#### QA-055

- Статус: `OK`
- Зона: `artifact regression`
- Файл/каталог: [tests/test_runtime_artifacts.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/test_runtime_artifacts.py)
- Наблюдение: есть отдельные регрессии на production JSON artifacts в `data/`, а не только на код вокруг них.
- Почему это важно: это дешёвый, но очень полезный уровень защиты от “тихого” рассинхрона артефактов и runtime-контрактов.
- Рекомендация: оставить и не превращать эти тесты в purely synthetic замену.

#### QA-056

- Статус: `FIX`
- Зона: `production coverage gaps`
- Файл/каталог: [src/priority_pipeline/decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/decision.py), [src/priority_pipeline/input_data.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/input_data.py), [src/priority_pipeline/persist.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/persist.py), [src/priority_pipeline/contracts.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/contracts.py), [src/priority_pipeline/relations.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/relations.py)
- Наблюдение: production pipeline покрыт в основном через `branching`, один mini-batch smoke и один DB-backed e2e, но ряд внутренних модулей почти не имеет собственных направленных тестов.
- Почему это важно: при локальных изменениях в decision/persist/input-contract слоях regression может проявиться поздно и не очень диагностично.
- Рекомендация: позже добавить несколько узких targeted tests именно на decision/persist/input contracts, без раздувания suite.

#### QA-057

- Статус: `FIX`
- Зона: `input validation coverage`
- Файл/каталог: [src/input_layer.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/input_layer.py), [tests/test_input_layer_db_integration.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/test_input_layer_db_integration.py)
- Наблюдение: для большого и критичного `input_layer.py` есть только два DB-backed теста; они ценны, но почти не покрывают быстрые ветки валидации и summary-статистики.
- Почему это важно: валидатор входных данных — чувствительная часть пайплайна, а сейчас он проверяется скорее как integration shell, чем как богатый набор правил.
- Рекомендация: позже рассмотреть компактные unit-like тесты на локальные ветки `validate_dataset(...)`, не заменяя DB-интеграции.

#### QA-058

- Статус: `FIX`
- Зона: `untested infrastructure`
- Файл/каталог: [src/infra/db.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/infra/db.py), [src/infra/relations.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/infra/relations.py), [src/infra/logbook.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/infra/logbook.py), [src/logbooks/decision_layer.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/logbooks/decision_layer.py), [src/logbooks/program_run.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/logbooks/program_run.py)
- Наблюдение: инфраструктурные и logbook-модули практически не имеют прямого тестового покрытия.
- Почему это важно: это не главная математическая зона риска, но именно такие вспомогательные слои часто ломаются от “невинных” изменений путей, relation naming или шаблонов журналов.
- Рекомендация: зафиксировать как реальный, но не первоочередной test gap.

#### QA-059

- Статус: `TOLERABLE`
- Зона: `research test depth`
- Файл/каталог: [tests/test_model_comparison_*.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests)
- Наблюдение: comparison-layer покрыт лучше среднего по контрактам split/CV/reporting/CLI, но сами model runs в основном тестируются smoke-уровнем, а не глубокими regression-сценариями на реальных search outcomes.
- Почему это важно: для исследовательского слоя это пока допустимо, но при дальнейшем усложнении benchmark logic именно здесь быстрее всего появятся хрупкие зоны.
- Рекомендация: не драматизировать сейчас, но не считать comparison tests “полным доказательством корректности” всех search-контуров.

#### QA-060

- Статус: `TOLERABLE`
- Зона: `integration breadth`
- Файл/каталог: `tests`
- Наблюдение: DB-backed интеграций всего три, и они корректно `skip`-аются без доступного Postgres.
- Почему это важно: это прагматично для локальной разработки, но означает, что часть критичных сценариев может оставаться непроверенной в среде без базы.
- Рекомендация: считать такой компромисс нормальным, но помнить о нём при интерпретации “зелёного” `pytest`.

#### QA-061

- Статус: `FIX`
- Зона: `numerical and failure edge cases`
- Файл/каталог: [src/priority_pipeline/decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/decision.py), [src/decision_calibration/scoring.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/decision_calibration/scoring.py), [tests/test_star_orchestrator.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/test_star_orchestrator.py), [tests/test_decision_layer_calibrator.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/test_decision_layer_calibrator.py)
- Наблюдение: decision/calibration слой имеет положительные сценарии, но почти не стрессуется на крайних численных случаях: экстремальные `ruwe`, плохая параллаксная геометрия, missing/negative значения факторов, clipping behaviour на границах.
- Почему это важно: именно такие кейсы чаще всего проявляют ошибки не в архитектуре, а в реальной численной устойчивости scoring-контуров.
- Рекомендация: позже добавить несколько targeted edge-case tests на факторы и clipping, без попытки покрыть всё комбинаторно.

#### QA-062

- Статус: `OK`
- Зона: `test style`
- Файл/каталог: `tests`
- Наблюдение: тесты написаны простым функцональным стилем без лишнего ООП и без чрезмерной meta-магии.
- Почему это важно: suite остаётся читаемым и дешёвым в поддержке.
- Рекомендация: сохранять этот стиль и дальше; не усложнять тестовую архитектуру без необходимости.

#### QA-063

- Статус: `TOLERABLE`
- Зона: `coverage philosophy`
- Файл/каталог: `tests`
- Наблюдение: тестовый набор скорее curated и risk-based, чем исчерпывающий по branch coverage; местами это видно по small числу тестов на большие модули.
- Почему это важно: это не недостаток сам по себе, но важно не переоценивать степень защиты только по зелёному количеству тестов.
- Рекомендация: в дальнейшем добавлять тесты по рискам, а не по формальному coverage number.

### Итог шага

- Тестовый контур проекта рабочий и полезный, но не равномерный по глубине.
- Сильные стороны: router/model regression, DB-backed проверки, artifact regressions, comparison contracts.
- Основные реальные test gaps: `input_layer`, внутренности production `priority_pipeline`, infra/logbook слой и численные edge cases decision/calibration.
- Следующий шаг логично посвятить формированию канонического QA-runbook: что именно и в каком порядке проверять при полном прогоне проекта.

## Шаг 9. План всестороннего тестирования

### Что сделано

- собран отдельный воспроизводимый runbook полного QA-прогона:
  [qa_runbook_2026-03-13_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/QA/qa_runbook_2026-03-13_ru.md)
- в runbook включены:
  - preflight и фиксация состояния worktree
  - статика
  - быстрые и полные тесты
  - DB-backed проверки
  - benchmark/snapshot comparison-layer
  - ручные математические sanity checks
  - notebook и docs validation

### Findings

#### QA-064

- Статус: `OK`
- Зона: `QA reproducibility`
- Файл/каталог: [experiments/QA/qa_runbook_2026-03-13_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/QA/qa_runbook_2026-03-13_ru.md)
- Наблюдение: по проекту уже можно собрать полноценный, воспроизводимый QA runbook без изобретения новой инфраструктуры.
- Почему это важно: зрелость проекта подтверждается не только кодом, но и возможностью повторить проверку пошагово.
- Рекомендация: использовать этот runbook как канонический порядок полного прогона, а не как разовый checklist.

#### QA-065

- Статус: `TOLERABLE`
- Зона: `QA interpretation`
- Файл/каталог: `QA process`
- Наблюдение: для этого проекта зелёные линтеры и `pytest` недостаточны сами по себе; нужен отдельный ручной sanity check benchmark/snapshot-артефактов и score-диапазонов.
- Почему это важно: часть существенных рисков здесь методическая и численная, а не только программная.
- Рекомендация: не упрощать полный QA до одной команды `pytest -q`.

#### QA-066

- Статус: `TOLERABLE`
- Зона: `cost of full QA`
- Файл/каталог: `QA process`
- Наблюдение: полный прогон проекта должен оставаться ступенчатым: дешёвые проверки отдельно, дорогие DB/benchmark/notebook шаги отдельно.
- Почему это важно: иначе QA быстро перестанут реально запускать и начнут обходить.
- Рекомендация: сохранить tiered-подход runbook-а и не прятать дорогие шаги внутри “обычного” smoke-прогона.

### Итог шага

- Канонический QA-runbook для проекта теперь зафиксирован.
- Полный QA здесь понимается как сочетание статики, тестов, benchmark/snapshot-проверки и ручного sanity review.
- Следующий шаг — аудит README и всей документации на актуальность, связность и понятность.

## Шаг 10. Аудит README и всей документации

### Что проверено

- полностью прочитан [README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/README.md)
- инвентаризированы все markdown-документы в `docs/`
- проверены локальные markdown-ссылки:
  missing local links не обнаружены
- прочитаны ключевые документы:
  - [documentation_style_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/documentation_style_ru.md)
  - [model_comparison_protocol_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/model_comparison_protocol_ru.md)
  - [model_comparison_findings_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/model_comparison_findings_ru.md)
  - [preprocessing_pipeline_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/preprocessing_pipeline_ru.md)
  - [vkr_requirements_traceability_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/vkr_requirements_traceability_ru.md)
  - [notebook_review_2026-03-13_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/notebook_review_2026-03-13_ru.md)
  - [ood_unknown_baselines_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/ood_unknown_baselines_tz_ru.md)
  - [ood_unknown_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/ood_unknown_tz_ru.md)
  - [preprocessing_and_comparison_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/preprocessing_and_comparison_tz_ru.md)
  - [vkr_slides_draft_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/presentation/vkr_slides_draft_ru.md)
- оформлен отдельный docs-ledger:
  [qa_docs_ledger_2026-03-13_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/QA/qa_docs_ledger_2026-03-13_ru.md)

### Findings

#### QA-067

- Статус: `OK`
- Зона: `markdown link integrity`
- Файл/каталог: `README.md`, `docs/*.md`
- Наблюдение: битых локальных markdown-ссылок в README и `docs/` не обнаружено.
- Почему это важно: документация действительно связана с repo, а не разваливается на уровне базовой навигации.
- Рекомендация: считать link integrity сильной стороной документационного слоя.

#### QA-068

- Статус: `OK`
- Зона: `README quality`
- Файл/каталог: [README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/README.md)
- Наблюдение: README хорошо объясняет цель проекта, архитектуру, основные сценарии запуска, QA и связь с ВКР-материалами.
- Почему это важно: для верхнеуровневой карты проекта README уже работает как реальный entrypoint, а не как формальный баннер.
- Рекомендация: использовать его как базовую навигацию по проекту и дальше.

#### QA-069

- Статус: `TOLERABLE`
- Зона: `README focus`
- Файл/каталог: [README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/README.md)
- Наблюдение: README местами смешивает несколько ролей сразу: пользовательскую инструкцию, архитектурный обзор, статус V1, ВКР-карту и roadmap следующих этапов.
- Почему это важно: документ остаётся полезным, но менее резким как “быстрый вход” для нового читателя.
- Рекомендация: позже решить, нужно ли слегка развести `overview` и `project status`, но не считать это срочной проблемой.

#### QA-070

- Статус: `FIX`
- Зона: `traceability freshness`
- Файл/каталог: [docs/vkr_requirements_traceability_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/vkr_requirements_traceability_ru.md)
- Наблюдение: карта соответствия частично отстаёт от текущего состояния: в строке `Preprocessing` всё ещё написано, что нет отдельного notebook, хотя [00_data_extraction_and_preprocessing.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/00_data_extraction_and_preprocessing.ipynb) уже существует.
- Почему это важно: это как раз тот документ, по которому удобно ловить формальные рассинхроны перед ВКР.
- Рекомендация: позже обновить как минимум устаревшие статусы и формулировки, чтобы traceability не подставляла на ровном месте.

#### QA-071

- Статус: `FIX`
- Зона: `historical planning docs`
- Файл/каталог: [docs/ood_unknown_baselines_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/ood_unknown_baselines_tz_ru.md)
- Наблюдение: документ уже частично расходится с текущим repo-state: например, упоминает `analysis/model_comparison/protocol.py`, которого нет, и описывает `MLP` как более позднюю волну, хотя она уже реализована.
- Почему это важно: это хороший исторический design doc, но как текущая инструкция он уже опасен своей неактуальностью.
- Рекомендация: позже либо явно пометить его как historical/archival, либо синхронизировать с текущим состоянием.

#### QA-072

- Статус: `TOLERABLE`
- Зона: `docs sprawl`
- Файл/каталог: `docs/`
- Наблюдение: в `docs/` уже накопилось несколько разных типов документов: канонические protocol/findings/style, активные рабочие обзоры и исторические ТЗ.
- Почему это важно: для автора это управляемо, но новому читателю не всегда очевидно, какой документ является “истиной сейчас”, а какой — следом прошлой планировочной волны.
- Рекомендация: позже подумать о более явном разделении на `canonical` и `historical planning`, не вычищая документы автоматически.

#### QA-073

- Статус: `OK`
- Зона: `core docs quality`
- Файл/каталог: [docs/documentation_style_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/documentation_style_ru.md), [docs/model_comparison_protocol_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/model_comparison_protocol_ru.md), [docs/model_comparison_findings_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/model_comparison_findings_ru.md), [docs/preprocessing_pipeline_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/preprocessing_pipeline_ru.md), [docs/notebook_review_2026-03-13_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/notebook_review_2026-03-13_ru.md), [docs/presentation/vkr_slides_draft_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/presentation/vkr_slides_draft_ru.md)
- Наблюдение: основные “живые” документы проекта читаются хорошо, логично связаны с кодом и артефактами и в целом соответствуют реальному состоянию.
- Почему это важно: документационный слой проекта уже полезен не только для ВКР, но и для разработки/поддержки.
- Рекомендация: считать именно эти документы основным каноническим слоем narrative и protocol.

#### QA-074

- Статус: `TOLERABLE`
- Зона: `README timeline language`
- Файл/каталог: [README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/README.md)
- Наблюдение: разделы `Что важно про V1` и `Ближайшие этапы` добавляют roadmap-контекст, но слегка размывают границу между “что уже есть” и “что ещё только планируется”.
- Почему это важно: в README это не критично, но именно там такие смешения быстрее всего читаются как устаревшая информация.
- Рекомендация: позже, при желании, чуть резче отделить current state от future work.

### Итог шага

- Документация проекта в целом сильная и хорошо связана ссылками.
- Главные проблемы здесь не в сломанных ссылках, а в том, что часть старых ТЗ уже живёт рядом с каноническими документами без явного archival-статуса.
- Самые полезные “живые” документы: `README`, `documentation_style`, `preprocessing_pipeline`, `model_comparison_protocol`, `model_comparison_findings`, `notebook_review`, `vkr_slides_draft`.
- Следующий шаг — отдельный аудит ноутбуков и SQL-слоя.

## Шаг 11. Аудит ноутбуков и SQL

### Что проверено

- инвентаризация всех notebooks в `notebooks/eda/`
- инвентаризация всех SQL/ADQL файлов в `sql/`
- метаданные notebooks:
  количество code/markdown cells, наличие outputs, kernelspec
- содержательные сигналы notebooks:
  - старые run-name
  - абсолютные локальные пути
  - warning-output
- структура SQL-слоя:
  - migration pairs
  - preprocessing files
  - ADQL templates
- отдельный ledger:
  [qa_notebooks_sql_ledger_2026-03-13_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/QA/qa_notebooks_sql_ledger_2026-03-13_ru.md)

### Findings

#### QA-075

- Статус: `OK`
- Зона: `notebook scope`
- Файл/каталог: `notebooks/eda/*.ipynb`
- Наблюдение: notebook-слой компактный и понятный: всего пять notebooks, и у каждого есть отдельная роль без явного дублирования.
- Почему это важно: это хороший признак, что notebook-контур не расползся в архив несвязанных черновиков.
- Рекомендация: текущий набор notebooks считать уместным и достаточным.

#### QA-076

- Статус: `OK`
- Зона: `notebook readiness`
- Файл/каталог: [notebooks/eda/00_data_extraction_and_preprocessing.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/00_data_extraction_and_preprocessing.ipynb), [notebooks/eda/01_host_eda_overview.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/01_host_eda_overview.ipynb), [notebooks/eda/02_router_readiness.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/02_router_readiness.ipynb), [notebooks/eda/03_host_vs_field_contrastive.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/03_host_vs_field_contrastive.ipynb), [notebooks/eda/04_model_comparison_summary.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/04_model_comparison_summary.ipynb)
- Наблюдение: все notebooks имеют выполненные code cells, сохранённые outputs и согласованный kernelspec `venv (3.13.2)`.
- Почему это важно: notebook-слой выглядит как реально используемый рабочий материал, а не как пустые контейнеры.
- Рекомендация: считать notebooks живыми артефактами проекта.

#### QA-077

- Статус: `TOLERABLE`
- Зона: `notebook output noise`
- Файл/каталог: [notebooks/eda/00_data_extraction_and_preprocessing.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/00_data_extraction_and_preprocessing.ipynb), [notebooks/eda/01_host_eda_overview.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/01_host_eda_overview.ipynb)
- Наблюдение: в сохранённых outputs есть локальный path noise и warning noise: в `00` сохранён `PosixPath('/Users/.../dspro-vkr')`, а в `01` остались `UserWarning` про glyph/font с абсолютным путём к локальному файлу.
- Почему это важно: это не ломает notebook, но чуть портит portability и презентационную аккуратность.
- Рекомендация: считать это косметическим, а не структурным дефектом.

#### QA-078

- Статус: `OK`
- Зона: `notebook freshness`
- Файл/каталог: [notebooks/eda/04_model_comparison_summary.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/04_model_comparison_summary.ipynb)
- Наблюдение: summary-notebook действительно привязан к новому comparison-контракту; старый run-name встречается только как пояснительный контраст в markdown, а не как живая зависимость.
- Почему это важно: это подтверждает, что comparison-notebook не остался на старом benchmark.
- Рекомендация: считать `04` актуальным.

#### QA-079

- Статус: `OK`
- Зона: `SQL structure`
- Файл/каталог: `sql/`
- Наблюдение: SQL-слой хорошо разложен на понятные группы: forward/rollback migrations, preprocessing SQL и ADQL templates.
- Почему это важно: SQL-контур проекта читается как канонический инженерный слой, а не как dump случайных запросов.
- Рекомендация: сохранить текущую структуру и не сводить её обратно в один DBeaver-монолит.

#### QA-080

- Статус: `OK`
- Зона: `SQL file quality`
- Файл/каталог: [sql/2026-03-11_gaia_results_posterior_host_fields.sql](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/2026-03-11_gaia_results_posterior_host_fields.sql), [sql/2026-03-13_gaia_results_unknown_constraints.sql](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/2026-03-13_gaia_results_unknown_constraints.sql), [sql/preprocessing/01_nasa_gaia_crossmatch.sql](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/preprocessing/01_nasa_gaia_crossmatch.sql), [sql/preprocessing/02_train_classification_views.sql](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/preprocessing/02_train_classification_views.sql), [sql/preprocessing/03_router_reference_layer.sql](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/preprocessing/03_router_reference_layer.sql), [sql/preprocessing/04_data_quality_checks.sql](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/preprocessing/04_data_quality_checks.sql), [sql/adql/01_nasa_hosts_crossmatch_batch_template.adql](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/adql/01_nasa_hosts_crossmatch_batch_template.adql), [sql/adql/02_validation_physics_enrichment.adql](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/adql/02_validation_physics_enrichment.adql), [sql/adql/03_gaia_reference_sampling_examples.adql](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/adql/03_gaia_reference_sampling_examples.adql)
- Наблюдение: SQL и ADQL файлы имеют внятные шапки, понятные роли и читаемую декомпозицию по этапам pipeline.
- Почему это важно: это редкий случай, когда SQL-слой проекта уже реально пригоден для чтения и повторного запуска.
- Рекомендация: считать SQL/ADQL сильной стороной repo.

#### QA-081

- Статус: `FIX`
- Зона: `notebook/sql repo hygiene`
- Файл/каталог: [sql/.DS_Store](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/.DS_Store), `sql/adql`, `sql/preprocessing`, `notebooks/eda/00_data_extraction_and_preprocessing.ipynb`, `notebooks/eda/04_model_comparison_summary.ipynb`
- Наблюдение: внутри notebook/SQL слоя остаётся repo-process шум: `sql/.DS_Store`, а важные SQL/ADQL и часть notebooks всё ещё находятся в untracked состоянии.
- Почему это важно: сам контент здесь хороший, но version-control картина пока не соответствует его фактической важности.
- Рекомендация: не чинить в ходе аудита, но считать это реальным cleanup/versioning issue.

#### QA-082

- Статус: `TOLERABLE`
- Зона: `artifact-heavy notebooks`
- Файл/каталог: `notebooks/eda/*.ipynb`
- Наблюдение: notebooks хранят выполненные outputs прямо в repo.
- Почему это важно: для ВКР и review это полезно, но увеличивает вес repo и делает notebooks более зависимыми от локальной среды.
- Рекомендация: пока это допустимо, потому что notebooks здесь являются presentation-ready артефактами, а не только черновиками.

### Итог шага

- Notebook-слой проекта в целом живой, компактный и пригодный для ВКР.
- SQL/ADQL слой выглядит очень аккуратно и уже тянет на канонический preprocessing-контур.
- Реальные замечания здесь в основном про repo hygiene и немного output-noise, а не про смысловые проблемы.
- Следующий шаг — ревизия артефактов и кандидатов на cleanup/вынос из git.

## Шаг 12. Ревизия артефактов и лишних файлов

### Что проверено

- инвентаризация `data/`
- инвентаризация `experiments/`
- инвентаризация `docs/assets/` и `docs/presentation/assets/`
- размеры ключевых директорий:
  - `data` — `9.4M`
  - `docs/assets` — `9.6M`
  - `docs/presentation/assets` — `508K`
  - `experiments/QA` — `148K`
  - `experiments/model_comparison` — `111M`
  - `experiments/Логи работы программы` — `12K`
  - `experiments/Логи калибровки decision_layer` — `384K`
- проверка ссылок из docs на assets
- оформлен отдельный cleanup-ledger:
  [qa_artifacts_cleanup_2026-03-13_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/QA/qa_artifacts_cleanup_2026-03-13_ru.md)

### Findings

#### QA-083

- Статус: `OK`
- Зона: `artifact usefulness`
- Файл/каталог: `data/`, `experiments/`, `docs/presentation/assets/`
- Наблюдение: большинство крупных артефактных зон выглядят осмысленно и связаны либо с runtime, либо с QA, либо с ВКР-материалами.
- Почему это важно: проблема проекта не в том, что он захламлён артефактами “вообще”, а в отсутствии явной политики хранения поколений этих артефактов.
- Рекомендация: исходить из идеи selective cleanup, а не из агрессивной чистки всего подряд.

#### QA-084

- Статус: `FIX`
- Зона: `filesystem junk`
- Файл/каталог: [data/.DS_Store](/Users/evgeniikuznetsov/Desktop/dspro-vkr/data/.DS_Store), [sql/.DS_Store](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/.DS_Store), [experiments/.DS_Store](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/.DS_Store), [docs/assets/.DS_Store](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/assets/.DS_Store)
- Наблюдение: в артефактных директориях остаются типичные macOS-мусорные файлы.
- Почему это важно: это низкосигнальный шум, но он прямо мешает perception чистоты repo.
- Рекомендация: считать эти файлы прямыми cleanup-кандидатами.

#### QA-085

- Статус: `TOLERABLE`
- Зона: `artifact generations`
- Файл/каталог: [experiments/model_comparison](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/model_comparison)
- Наблюдение: directory `experiments/model_comparison` содержит сразу несколько поколений прогонов одного и того же дня: базовую волну, `mlp`-волну, `snapshot`-волну и каноническую `vkr30_cv10`-волну.
- Почему это важно: сами артефакты не бессмысленны, но без retention policy директорий становится трудно понять, что считать каноническим текущим результатом.
- Рекомендация: позже принять явную политику: что остаётся как canonical run, а что уходит в archive/history.

#### QA-086

- Статус: `FIX`
- Зона: `canonical artifact policy`
- Файл/каталог: [experiments/model_comparison](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/model_comparison)
- Наблюдение: для comparison-артефактов сейчас не зафиксировано, какие run-name являются актуальными, а какие уже только историческими.
- Почему это важно: это уже не просто эстетика, а риск документального рассинхрона между findings, notebook, slides и набором CSV/markdown рядом с ними.
- Рекомендация: позже явно выделить canonical generation (`vkr30_cv10`) и отдельно решить судьбу предыдущих волн.

#### QA-087

- Статус: `TOLERABLE`
- Зона: `unreferenced visual assets`
- Файл/каталог: [docs/assets/Снимок экрана 2026-03-12 в 16.17.24.png](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/assets/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202026-03-12%20%D0%B2%2016.17.24.png), [docs/assets/Снимок экрана 2026-03-12 в 16.28.37.png](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/assets/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202026-03-12%20%D0%B2%2016.28.37.png), [docs/assets/Снимок экрана 2026-03-12 в 16.29.44.png](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/assets/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202026-03-12%20%D0%B2%2016.29.44.png)
- Наблюдение: в `docs/assets` лежат три скриншота с generic именами, и ни один markdown-документ на них не ссылается.
- Почему это важно: это похоже на raw capture residue, а не на канонический repo-asset.
- Рекомендация: позже решить, нужны ли они вообще; если нужны, переименовать и явно привязать к docs.

#### QA-088

- Статус: `OK`
- Зона: `intentional experiment logs`
- Файл/каталог: [experiments/Логи работы программы](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/%D0%9B%D0%BE%D0%B3%D0%B8%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%8B%20%D0%BF%D1%80%D0%BE%D0%B3%D1%80%D0%B0%D0%BC%D0%BC%D1%8B), [experiments/Логи калибровки decision_layer](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/%D0%9B%D0%BE%D0%B3%D0%B8%20%D0%BA%D0%B0%D0%BB%D0%B8%D0%B1%D1%80%D0%BE%D0%B2%D0%BA%D0%B8%20decision_layer)
- Наблюдение: program logs и calibration logs выглядят осмысленными историческими артефактами, а не мусором.
- Почему это важно: их не стоит автоматически включать в cleanup только потому, что они лежат в `experiments/`.
- Рекомендация: сохранять как часть исследовательской истории проекта.

#### QA-089

- Статус: `OK`
- Зона: `presentation assets`
- Файл/каталог: [docs/presentation/assets](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/presentation/assets)
- Наблюдение: presentation-assets лёгкие по размеру, связаны со slide draft и имеют понятную роль.
- Почему это важно: это как раз хороший пример артефактов, которые имеет смысл держать рядом с docs.
- Рекомендация: оставить в repo.

#### QA-090

- Статус: `TOLERABLE`
- Зона: `data policy`
- Файл/каталог: [data/raw](/Users/evgeniikuznetsov/Desktop/dspro-vkr/data/raw), [data/processed](/Users/evgeniikuznetsov/Desktop/dspro-vkr/data/processed), [data/eda](/Users/evgeniikuznetsov/Desktop/dspro-vkr/data/eda), [data/plots](/Users/evgeniikuznetsov/Desktop/dspro-vkr/data/plots)
- Наблюдение: `data/` выглядит содержательно полезным, но для него не очень явно зафиксирована policy: что является production artifact, что demo/raw sample, а что просто локальный EDA residue.
- Почему это важно: пока размеры небольшие, это не блокер, но с ростом проекта именно тут легче всего возникнет неуправляемый data-dump.
- Рекомендация: позже ввести более явное различение `production artifacts / sample data / EDA outputs`.

### Итог шага

- Артефакты проекта в целом осмысленные, но политика хранения поколений и сырых ассетов пока не формализована.
- Главные cleanup-кандидаты: `.DS_Store`, unreferenced screenshots и часть старых волн `experiments/model_comparison`.
- Канонические артефакты, которые точно не выглядят лишними: production JSONs, QA-отчёты, program/calibration logs, preprocessing screenshots с явными ссылками и presentation-assets.
- Дальше остаётся финальный шаг: собрать все findings в единый backlog и карту решений.

## Шаг 13. Сводный backlog и карта решений

### Что сделано

- собран итоговый backlog и decision map:
  [qa_backlog_and_decision_map_2026-03-13_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/QA/qa_backlog_and_decision_map_2026-03-13_ru.md)
- суммарная картина findings:
  - `OK` — `37`
  - `TOLERABLE` — `35`
  - `FIX` — `18`

### Findings

#### QA-091

- Статус: `OK`
- Зона: `overall audit result`
- Файл/каталог: `project-wide`
- Наблюдение: суммарная картина аудита получилась здоровой: сильных красных флагов меньше, чем допускалось на старте, а значительная часть findings относится к hygiene, policy и границам слоёв, а не к фундаментальным багам.
- Почему это важно: это подтверждает, что проект уже находится в фазе точечной доработки, а не тотального спасения архитектуры.
- Рекомендация: будущую волну правок делать узкой и приоритизированной.

#### QA-092

- Статус: `FIX`
- Зона: `highest priority cluster`
- Файл/каталог: `repo-wide`
- Наблюдение: самый опасный реальный риск сейчас — не математический, а организационно-репозиторный: критичные код/тесты/notebooks/SQL/docs и canonical артефакты не до конца формализованы как versioned current state.
- Почему это важно: именно это быстрее всего создаёт проблемы воспроизводимости и защиты, даже если код при этом работает.
- Рекомендация: считать version-control и canonicality первым приоритетом следующей волны.

#### QA-093

- Статус: `FIX`
- Зона: `next priority cluster`
- Файл/каталог: [src/host_model/db.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/host_model/db.py), [src/priority_pipeline/decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/decision.py), [analysis/host_eda/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/host_eda/__init__.py), [analysis/router_eda/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/router_eda/__init__.py)
- Наблюдение: второй по важности кластер — это неидеальные архитектурные сцепки и import side effects в нескольких точках.
- Почему это важно: здесь можно получить реальное упрощение и снижение технического долга без масштабного рефакторинга проекта.
- Рекомендация: после version-control вопросов идти именно в эти точки, а не распыляться на крупные косметические переделки.

#### QA-094

- Статус: `TOLERABLE`
- Зона: `what not to over-fix`
- Файл/каталог: [src/input_layer.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/input_layer.py), [src/decision_calibration/reporting.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/decision_calibration/reporting.py), [analysis/model_comparison/mlp_baseline.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/mlp_baseline.py), [analysis/model_comparison/presentation_assets.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/presentation_assets.py)
- Наблюдение: часть больших модулей действительно крупная, но пока не даёт достаточного сигнала, чтобы дробить её ради эстетики.
- Почему это важно: это предохраняет от рефакторинга “ради порядка”, который не даёт заметной инженерной отдачи.
- Рекомендация: не делать эти файлы приоритетной целью следующей волны.

#### QA-095

- Статус: `OK`
- Зона: `math confidence`
- Файл/каталог: `router / host / decision / comparison`
- Наблюдение: аудит не выявил явного фундаментального математического дефекта в production scoring или comparison protocol.
- Почему это важно: это позволяет не трогать ядро проекта без жёсткой причины.
- Рекомендация: сосредоточиться на boundary conditions, тестах и чистоте контрактов, а не на переписывании математики.

### Итог шага

- Полный QA-аудит проекта завершён.
- Все findings зафиксированы по шагам, а итоговый backlog уже сжат до реально управляемого набора решений.
- Следующая отдельная волна работы должна быть уже не аудитом, а выбором конкретных правок из decision map.
