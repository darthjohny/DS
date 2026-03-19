# Реестр findings по аудиту проекта

Дата открытия: 19 марта 2026 года

## 1. Назначение документа

Этот файл используется как единый реестр наблюдений текущей audit-wave.

Сюда записываются:

- найденные проблемы;
- спорные места;
- сильные стороны;
- evidence из кода, артефактов, notebooks и прогонов;
- предварительная интерпретация.

Отдельно в текущей audit-wave сюда нужно складывать:

- confusing naming файлов, модулей, notebooks и artifacts;
- места, где naming затрудняет onboarding и понимание ownership;
- naming drift между кодом, docs и experiment-layer.

Сюда не записывается:

- подробный план исправлений;
- массовые идеи рефакторинга без evidence;
- правки “на будущее” без привязки к текущей задаче.

Fix-plan будет оформляться отдельно только после завершения основной
audit-wave.

## 2. Формат записи

Каждый finding должен по возможности содержать:

- `ID`
- `Приоритет`
- `Категория`
- `Зона`
- `Факт`
- `Evidence`
- `Почему это важно`
- `Статус`

Рекомендуемые категории:

- `logic`
- `physics`
- `ml`
- `code`
- `tests`
- `notebook`
- `docs`
- `ops`
- `naming`

Статусы:

- `open`
- `investigating`
- `confirmed`
- `accepted_as_is`

## 3. Сводная таблица findings

| ID | Приоритет | Категория | Зона | Кратко | Статус |
| --- | --- | --- | --- | --- | --- |
| A-LOGIC-001 | P1 | logic | orchestrator docs vs runtime | Канонический orchestrator-doc частично устарел относительно production decision-layer | confirmed |
| A-LOGIC-002 | P1 | logic | priority persist contract | Runtime считает `reliability_factor` и `followup_factor`, но persist-layer сохраняет только legacy `quality_factor` | confirmed |
| A-LOGIC-003 | P1 | logic | production vs offline calibration | Offline calibration по-прежнему не совпадает с production decision semantics | confirmed |
| A-LOGIC-004 | P1 | logic | input vs UNKNOWN contract | `missing features -> UNKNOWN` по spec не выполняется в production из-за ранней фильтрации входа | confirmed |
| A-LOGIC-S001 | P2 | logic | pipeline architecture | Основная runtime-цепочка хорошо разделена по ответственности | confirmed |
| A-LOGIC-S002 | P2 | logic | router/OOD boundary | OOD аккуратно вынесен из raw router scoring в отдельный слой | confirmed |
| A-LOGIC-S003 | P2 | logic | input normalization | Входной batch детерминированно схлопывается до одной строки на `source_id` с quality-aware tie-break | confirmed |
| A-LOGIC-S004 | P2 | logic | production vs research split | Production runtime, comparison-layer и heavy validation разведены по отдельным пакетам и protocol-docs | confirmed |
| A-DOC-001 | P1 | docs | public scoring description | README описывает устаревшую боевую формулу | confirmed |
| A-PHYS-001 | P1 | physics | class prioritization | Текущий runtime подтверждает `K` как устойчивый high-priority класс, но не даёт столь же сильной опоры для `G` как отдельной “третьей очереди” | confirmed |
| A-NOTEBOOK-001 | P1 | notebook | shortlist/final narrative | Итоговый порядок `K -> M -> G` в summary-layer задан ручным `priority_map`, а не напрямую извлечён из model output | confirmed |
| A-PHYS-S001 | P2 | physics | class priors vs outcome | Доминирование `K` в current runtime не сводится к ручному завышению class prior | confirmed |
| A-ML-001 | P1 | ml | production vs comparison snapshot | `main_contrastive_v1` в comparison snapshot не равен production artifact и переобучается заново | confirmed |
| A-ML-002 | P1 | ml | snapshot reproducibility | Канонический versioned snapshot-артефакт от 13 марта больше не воспроизводится current comparison code-path | confirmed |
| A-ML-003 | P1 | ml | production reproducibility | Текущее production runtime-поведение на `limit=5000` резко расходится с versioned orchestrator baseline артефактом | confirmed |
| A-ML-004 | P2 | ml | class-wise quality | `baseline_mlp_small` при глобальном threshold практически выключает класс `F` | confirmed |
| A-ML-005 | P2 | ml | overfitting risk | `baseline_random_forest` лучший по benchmark, но и самый напряжённый по validation gaps | confirmed |
| A-ML-006 | P2 | ml | threshold behavior | `main_contrastive_v1` работает как recall-heavy модель с большим числом false positive | confirmed |
| A-NOTEBOOK-002 | P1 | notebook | shortlist source semantics | Итоговый shortlist в summary notebook строится по comparison snapshot, а не по боевому production run | confirmed |
| A-ML-S001 | P2 | ml | validation discipline | Validation-layer реально ловит overfit/instability сигналы, а не маскирует их | confirmed |
| A-ML-S002 | P2 | ml | benchmark reproducibility | Current benchmark-probe воспроизводит versioned `vkr30_cv10` артефакты | confirmed |
| A-CODE-001 | P1 | code | typing gate | Полный `mypy` по проекту сейчас уже не зелёный из-за нового quality-layer | confirmed |
| A-CODE-002 | P2 | code | input layer ownership | `src/input_layer/__init__.py` перегружен ответственностями и прячет реальную реализацию в package `__init__` | confirmed |
| A-CODE-003 | P2 | code | snapshot layer size | `analysis/model_comparison/snapshot.py` совмещает training, scoring, assembly и artifact-writing в одном модуле | confirmed |
| A-CODE-004 | P2 | code | reporting layer size | `analysis/model_comparison/reporting.py` совмещает агрегацию, quality, generalization и persistence-логику | confirmed |
| A-TEST-001 | P2 | tests | orchestration tests | Часть comparison/model-validation тестов стала крупной и mock-heavy, что ухудшает читаемость и локализацию поломок | confirmed |
| A-NAMING-001 | P2 | naming | package/file naming | Смешение package-реализации в `__init__.py`, новых `__main__.py` и legacy facade-имен ухудшает onboarding | confirmed |
| A-TEST-S001 | P2 | tests | test feedback loop | Полный `pytest` остаётся быстрым и зелёным, что даёт хороший базовый feedback loop | confirmed |
| A-TEST-S002 | P2 | tests | test runtime concentration | Время full-suite в основном съедают несколько реальных model-fit smoke-тестов, а не системная тяжесть всего дерева | confirmed |
| A-DOC-S001 | P2 | docs | docstring coverage | В `src/` и `analysis/` не найдено публичных top-level функций и классов без docstring | confirmed |
| A-NOTEBOOK-003 | P2 | notebook | notebook hygiene | В Jupyter notebooks отсутствуют `cell id`, из-за чего `nbformat` уже выдаёт warning о будущей несовместимости | confirmed |
| A-NOTEBOOK-004 | P2 | notebook | summary notebook portability | Итоговый comparison notebook жёстко привязан к фиксированным run-name и содержит очень крупные setup/presentation cells | confirmed |
| A-DOC-002 | P2 | docs | presentation narrative | Slide-draft смешивает “верхушку contrastive ranking” и общий `K -> M -> G` operational priority без явного разведения уровней вывода | confirmed |
| A-NAMING-S001 | P2 | naming | stage naming | Именование основных EDA notebooks и analysis-пакетов в целом хорошо отражает стадии проекта | confirmed |

## 4. Inventory / Current State

### 4.1 Подтверждённый current state

Зафиксировано:

- корень проекта:
  [dspro-vkr](/Users/evgeniikuznetsov/Desktop/dspro-vkr)
- основной обзор:
  [README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/README.md)
- policy current state:
  [repository_state_policy_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/repository_state_policy_ru.md)
- канонический comparison-run:
  [baseline_comparison_2026-03-13_vkr30_cv10.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10.md)
- канонический snapshot preview:
  [baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot.md)
- текущий summary notebook:
  [04_model_comparison_summary.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/04_model_comparison_summary.ipynb)

### 4.2 Audit-оговорка по рабочему дереву

На момент старта audit-wave рабочее дерево `dirty`.

Это нужно учитывать при интерпретации findings:

- часть файлов уже находится в незакоммиченном состоянии;
- findings должны ссылаться и на file-path, и на текущий observed state;
- при необходимости отдельно отмечаем:
  `канонический контракт` vs `локальное текущее состояние`.

## 5. Логика и архитектура

### Findings

#### A-LOGIC-001

- `ID`: `A-LOGIC-001`
- `Приоритет`: `P1`
- `Категория`: `logic`
- `Зона`: `orchestrator docs vs runtime`
- `Факт`:
  Канонический документ по orchestrator уже частично не соответствует
  текущему production-коду. В документе всё ещё написано, что:
  - production использует `host_posterior × class_prior × quality_factor × metallicity_factor × color_factor × validation_factor`;
  - `distance_factor` скрыт внутри `quality_factor`;
  - выравнивание production/offline ещё только целевое.

  В текущем production-коде decision-layer уже реально разделён на:
  - `host_score`
  - `reliability_factor`
  - `followup_factor`
  - `color_factor`
  - `validation_factor`
- `Evidence`:
  - [orchestrator_host_prioritization_canon_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/orchestrator_host_prioritization_canon_ru.md)
  - [decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/decision.py)
- `Почему это важно`:
  На защите и в последующих audit-выводах легко перепутать:
  - что является текущим runtime fact;
  - что является target canon;
  - что уже выровнено, а что ещё нет.

  Это не просто cosmetic docs drift, а риск логической путаницы между
  production, research и explanation-layer.
- `Статус`: `confirmed`

#### A-LOGIC-002

- `ID`: `A-LOGIC-002`
- `Приоритет`: `P1`
- `Категория`: `logic`
- `Зона`: `priority persist contract`
- `Факт`:
  Production runtime уже считает раздельно:
  - `reliability_factor`
  - `followup_factor`

  Но persist-контракт итоговой relation всё ещё сохраняет только legacy
  `quality_factor` и не сохраняет ни `reliability_factor`, ни
  `followup_factor`.
- `Evidence`:
  - [decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/decision.py)
  - [constants.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/constants.py)
  - [persist.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/persist.py)
- `Почему это важно`:
  Декомпозиция итогового `final_score` частично теряется при записи в БД.
  Это осложняет:
  - sensitivity-аудит observability-факторов;
  - верификацию вклада distance/follow-up;
  - объяснение результата по persisted artifacts, если анализ идёт уже
    не по runtime frame, а по таблице результатов.
- `Статус`: `confirmed`

#### A-LOGIC-003

- `ID`: `A-LOGIC-003`
- `Приоритет`: `P1`
- `Категория`: `logic`
- `Зона`: `production vs offline calibration`
- `Факт`:
  Offline calibration по-прежнему живёт на существенно другой decision
  semantics:
  - `distance_factor` и `quality_factor` считаются явно;
  - `color_factor` отсутствует;
  - `validation_factor` отсутствует;
  - итоговая формула отличается от current production.
- `Evidence`:
  - [decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/decision.py)
  - [scoring.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/decision_calibration/scoring.py)
  - [orchestrator_host_prioritization_canon_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/orchestrator_host_prioritization_canon_ru.md)
- `Почему это важно`:
  Offline calibration и production ranking нельзя интерпретировать как
  строго один и тот же decision-layer. Это значит, что:
  - calibration artifacts полезны, но не являются прямой прокси для
    production score;
  - findings по offline quality нельзя автоматически переносить на
    runtime без оговорок.
- `Статус`: `confirmed`

#### A-LOGIC-004

- `ID`: `A-LOGIC-004`
- `Приоритет`: `P1`
- `Категория`: `logic`
- `Зона`: `input vs UNKNOWN contract`
- `Факт`:
  В OOD-spec зафиксировано, что один из допустимых сценариев —
  `missing features -> UNKNOWN`. Но production input-layer сейчас
  фильтрует строки без `teff_gspphot`, `logg_gspphot` или
  `radius_gspphot` ещё до router scoring.

  В результате объект с неполной базовой физикой не получает canonical
  `UNKNOWN`, а просто исчезает из runtime batch.
- `Evidence`:
  - [ood_unknown_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/ood_unknown_tz_ru.md)
  - [input_data.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/input_data.py)
  - [score.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/score.py)
- `Почему это важно`:
  Реальная семантика `UNKNOWN` в production уже сейчас уже, чем в spec:
  она покрывает low-confidence / low-support router cases, но не
  structural incompleteness входной relation.

  Это влияет на:
  - трактовку `unknown_share`;
  - полноту audit-а по open-set поведению;
  - формулировки в docs и на защите.
- `Статус`: `confirmed`

## 6. Математика и физика

### Findings

#### A-PHYS-001

- `ID`: `A-PHYS-001`
- `Приоритет`: `P1`
- `Категория`: `physics`
- `Зона`: `class prioritization`
- `Факт`:
  Текущий runtime действительно поддерживает `K` как главный устойчивый
  класс high-priority кандидатов, но значительно слабее поддерживает
  тезис о `G` как отдельной полноценной “третьей очереди”.

  На живом safe-run `limit=5000` текущий production runtime дал:
  - `HIGH`: `K=9`, `M=6`, `G=0`, `F=0`
  - `MEDIUM`: `K=971`, `M=297`, `G=128`, `F=27`

  При этом:
  - по числу `HIGH` и `MEDIUM` объектов доминирует `K`;
  - `M` даёт часть самых сильных пиковых кандидатов;
  - `G` в current runtime почти не выходит в верхнюю приоритетную зону.
- `Evidence`:
  - live safe-run `run_pipeline(engine, limit=5000, persist=False)` на
    19 марта 2026 года
  - [decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/decision.py)
  - [baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot.md)
- `Почему это важно`:
  Для текущей `V1` корректнее говорить так:
  - `K` — массово устойчивый основной слой;
  - `M` — класс с сильными верхними кандидатами;
  - `G` — допустимый резервный слой, но не столь же уверенно
    подтверждённый как отдельная приоритетная очередь.

  Иначе есть риск выдать presentation-friendly ordering за прямой
  вывод модели.
- `Статус`: `confirmed`

## 7. ML-качество и боевые прогоны

### Findings

#### A-ML-001

- `ID`: `A-ML-001`
- `Приоритет`: `P1`
- `Категория`: `ml`
- `Зона`: `production vs comparison snapshot`
- `Факт`:
  `main_contrastive_v1` внутри comparison snapshot не использует боевой
  production artifact напрямую. Вместо этого comparison-layer заново
  обучает модель на benchmark train split через
  `fit_main_contrastive_model_with_search(...)`, а уже потом применяет её
  к live Gaia batch.
- `Evidence`:
  - [pipeline.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/pipeline.py)
  - [input_data.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/input_data.py)
  - [snapshot.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/snapshot.py)
- `Почему это важно`:
  `main_contrastive_v1` в summary-layer и `main_contrastive_v1` в
  production runtime — это сейчас не один и тот же model instance.

  Следовательно:
  - comparison snapshot нельзя автоматически трактовать как поведение
    production artifact;
  - прямое сравнение `run_pipeline(...)` и snapshot-артефактов требует
    оговорки;
  - narrative “основная модель показала на живом батче ...” может быть
    двусмысленным, если не уточнять источник.
- `Статус`: `confirmed`

#### A-ML-002

- `ID`: `A-ML-002`
- `Приоритет`: `P1`
- `Категория`: `ml`
- `Зона`: `snapshot reproducibility`
- `Факт`:
  Канонический versioned snapshot-артефакт
  `baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot.md`
  больше не воспроизводится current comparison code-path в текущей
  среде.

  На 19 марта 2026 года текущий rerun дал:

  - `main_contrastive_v1`: `HIGH=56`, `MEDIUM=1386`, `LOW=3558`
  - `baseline_legacy_gaussian`: `HIGH=68`, `MEDIUM=1182`, `LOW=3750`
  - `baseline_random_forest`: `HIGH=602`, `MEDIUM=936`, `LOW=3462`
  - `baseline_mlp_small`: `HIGH=291`, `MEDIUM=976`, `LOW=3733`

  Тогда как versioned artifact от 13 марта зафиксировал:

  - `main_contrastive_v1`: `HIGH=811`, `MEDIUM=1379`, `LOW=2810`
  - `baseline_legacy_gaussian`: `HIGH=18`, `MEDIUM=868`, `LOW=4114`
  - `baseline_random_forest`: `HIGH=439`, `MEDIUM=927`, `LOW=3634`
  - `baseline_mlp_small`: `HIGH=168`, `MEDIUM=898`, `LOW=3934`
- `Evidence`:
  - live current rerun `run_snapshot_comparison(limit=5000)` на
    19 марта 2026 года
  - [baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot.md)
- `Почему это важно`:
  Это сильный reproducibility-signal. На текущем этапе нельзя без
  оговорок считать versioned snapshot от 13 марта точным представлением
  current code-path.

  Источник расхождения ещё не разложен до конца:
  - code drift;
  - model-selection drift;
  - data drift;
  - комбинация факторов.

  Но сам факт расхождения уже подтверждён.
- `Статус`: `confirmed`

#### A-ML-003

- `ID`: `A-ML-003`
- `Приоритет`: `P1`
- `Категория`: `ml`
- `Зона`: `production reproducibility`
- `Факт`:
  Текущее production runtime-поведение на `public.gaia_dr3_training`
  c `limit=5000` резко расходится с versioned orchestrator baseline
  артефактом, который уже лежит в репозитории.

  Versioned production-like артефакт от 14 марта фиксирует:
  - `high_count = 777`
  - `medium_count = 1487`
  - `low_count = 2736`

  Текущий safe-run production pipeline на 19 марта даёт:
  - `HIGH = 15`
  - `MEDIUM = 1423`
  - `LOW = 3562`
- `Evidence`:
  - live safe-run `run_pipeline(engine, limit=5000, persist=False)` на
    19 марта 2026 года
  - [orchestrator_baseline_2026-03-14_limit5000_summary.csv](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/model_validation/orchestrator_baseline_2026-03-14_limit5000_summary.csv)
- `Почему это важно`:
  Это уже не “небольшой drift”, а сильное расхождение в operational
  поведении текущей production-версии относительно versioned
  reproducibility-артефактов.

  Пока причина не локализована окончательно, но для защиты и Git это
  значит: historical production-like артефакты нельзя автоматически
  выдавать за current state.
- `Статус`: `confirmed`

#### A-ML-004

- `ID`: `A-ML-004`
- `Приоритет`: `P2`
- `Категория`: `ml`
- `Зона`: `class-wise quality`
- `Факт`:
  `baseline_mlp_small` при выбранном глобальном train-threshold
  фактически перестаёт выдавать положительные решения для класса `F`.

  В versioned quality-артефактах для `F` на test split:
  - `precision = 0.0`
  - `recall = 0.0`
  - `f1 = 0.0`
- `Evidence`:
  - [baseline_comparison_2026-03-13_vkr30_cv10.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10.md)
  - [baseline_comparison_2026-03-13_vkr30_cv10_quality_classwise.csv](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10_quality_classwise.csv)
- `Почему это важно`:
  Это хороший пример того, почему одной общей aggregate-метрики мало:
  overall quality у MLP сильная, но class-wise поведение уже показывает
  явную слепую зону.

  Для `V1` это не ломает проект, потому что MLP — baseline, а не
  production core. Но как findings для честной ВКР это обязательно стоит
  держать в поле зрения.
- `Статус`: `confirmed`

#### A-ML-005

- `ID`: `A-ML-005`
- `Приоритет`: `P2`
- `Категория`: `ml`
- `Зона`: `overfitting risk`
- `Факт`:
  `baseline_random_forest` остаётся численно лучшей моделью на
  benchmark, но одновременно показывает самый напряжённый профиль по
  train/test gap и получает наибольший risk-level в heavy validation.

  Примеры из validation-layer:
  - `avg train/test gap for pr_auc = 0.185`
  - `avg train/test gap for precision_at_k = 0.113`
  - `risk_level = HIGH`
- `Evidence`:
  - [baseline_comparison_2026-03-13_vkr30_cv10.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10.md)
  - [model_validation_2026-03-14_fast_v1_validation_report.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/model_validation/model_validation_2026-03-14_fast_v1_validation_report.md)
- `Почему это важно`:
  Это не отменяет победу `RandomForest` по benchmark, но не даёт
  трактовать эту победу как безусловную замену основной production
  модели без оговорок по устойчивости.
- `Статус`: `confirmed`

#### A-ML-006

- `ID`: `A-ML-006`
- `Приоритет`: `P2`
- `Категория`: `ml`
- `Зона`: `threshold behavior`
- `Факт`:
  `main_contrastive_v1` на threshold-based quality ведёт себя как
  recall-heavy модель:
  - `recall = 0.8665`
  - `precision = 0.4823`
  - `FP = 948` на test split

  То есть она редко пропускает host-объекты, но заметно чаще даёт
  ложноположительные решения, чем `RF` и `MLP`.
- `Evidence`:
  - [baseline_comparison_2026-03-13_vkr30_cv10.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10.md)
  - [audit_probe_2026-03-19_current.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/model_comparison/audit_probe_2026-03-19_current.md)
- `Почему это важно`:
  Это объясняет, почему contrastive может быть привлекательной для
  retrieval-like задачи follow-up shortlist, но одновременно хуже как
  строгий бинарный классификатор.
- `Статус`: `confirmed`

## 8. Код

### Findings

Пока пусто.

## 9. Тесты

### Findings

Пока пусто.

## 10. Ноутбуки и выводы

### Findings

#### A-NOTEBOOK-001

- `ID`: `A-NOTEBOOK-001`
- `Приоритет`: `P1`
- `Категория`: `notebook`
- `Зона`: `shortlist/final narrative`
- `Факт`:
  Итоговый shortlist и финальная формулировка в summary-layer используют
  явный ручной порядок:

  `priority_map = {"K": 1, "M": 2, "G": 3}`

  То есть финальный порядок `K -> M -> G` в notebook задаётся как
  presentation policy, а не вычисляется напрямую из runtime-ranking.
- `Evidence`:
  - [04_model_comparison_summary.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/04_model_comparison_summary.ipynb)
  - [model_comparison_findings_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/model_comparison_findings_ru.md)
  - live safe-run `run_pipeline(engine, limit=5000, persist=False)` на
    19 марта 2026 года
- `Почему это важно`:
  Само по себе такое presentation-layer решение допустимо для `V1`, но
  его нужно честно интерпретировать как policy overlay.

  Если оставить формулировку без оговорки, можно создать впечатление,
  что модель “сама доказала” именно такой строгий порядок, хотя в
  текущем runtime это не так.
- `Статус`: `confirmed`

#### A-NOTEBOOK-002

- `ID`: `A-NOTEBOOK-002`
- `Приоритет`: `P1`
- `Категория`: `notebook`
- `Зона`: `shortlist source semantics`
- `Факт`:
  Итоговый shortlist в summary notebook строится по
  `main_contrastive_v1` из comparison snapshot-артефакта, а не по
  результату боевого `run_pipeline(...)` с production artifact.
- `Evidence`:
  - [04_model_comparison_summary.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/04_model_comparison_summary.ipynb)
  - [snapshot.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/snapshot.py)
  - [pipeline.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/pipeline.py)
- `Почему это важно`:
  Notebook может выглядеть так, будто показывает “боевой список звёзд”,
  хотя на самом деле показывает shortlist от retrained comparison head.

  Для исследовательского слоя это допустимо, но для защиты и для Git
  narrative нужна явная оговорка, иначе смешиваются:
  - production runtime;
  - benchmark/snapshot comparison;
  - прикладной shortlist.
- `Статус`: `confirmed`

## 11. Документация и presentation-layer

### Findings

#### A-DOC-001

- `ID`: `A-DOC-001`
- `Приоритет`: `P1`
- `Категория`: `docs`
- `Зона`: `public scoring description`
- `Факт`:
  Public-facing описание scoring-контракта в `README` уже не
  соответствует текущему production decision-layer.

  В `README` всё ещё написано, что боевая формула:

  `host_posterior × class_prior × quality_factor × metallicity_factor × color_factor × validation_factor`

  тогда как runtime-код уже реально считает:
  - `host_score = host_posterior × class_prior × metallicity_factor`
  - `final_score = host_score × reliability_factor × followup_factor × color_factor × validation_factor`
- `Evidence`:
  - [README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/README.md)
  - [decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/decision.py)
- `Почему это важно`:
  README — это первая точка входа для нового человека и для быстрой
  защиты архитектуры. Если он описывает старую формулу, то:
  - onboarding становится менее надёжным;
  - техническое объяснение проекта расходится с runtime;
  - findings по observability/follow-up факторам сложнее объяснять.
- `Статус`: `confirmed`

## 12. Явно подтверждённые сильные стороны

Сюда складываются не проблемы, а подтверждённые сильные стороны проекта,
если они подтверждены evidence и помогают защитной позиции.

#### A-LOGIC-S001

- `ID`: `A-LOGIC-S001`
- `Приоритет`: `P2`
- `Категория`: `logic`
- `Зона`: `pipeline architecture`
- `Факт`:
  Основная production-цепочка хорошо разделена по ответственности:
  - [input_data.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/input_data.py) отвечает за relation loading, dedup и model loading;
  - [pipeline.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/pipeline.py) оркестрирует шаги;
  - [branching.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/branching.py) маршрутизирует ветки;
  - [decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/decision.py) считает factors и final score;
  - [persist.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/persist.py) пишет в БД.
- `Evidence`:
  - [pipeline.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/pipeline.py)
  - [input_data.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/input_data.py)
  - [branching.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/branching.py)
  - [decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/decision.py)
  - [persist.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/persist.py)
- `Почему это важно`:
  Это снижает риск монолитности, облегчает audit и делает архитектуру
  защитопригодной: границы между input/router/host-score/decision/persist
  реально существуют в коде, а не только в презентации.
- `Статус`: `confirmed`

#### A-LOGIC-S002

- `ID`: `A-LOGIC-S002`
- `Приоритет`: `P2`
- `Категория`: `logic`
- `Зона`: `router/OOD boundary`
- `Факт`:
  Open-set поведение router не смешано с raw scoring и живёт отдельным
  подслоем.
- `Evidence`:
  - [score.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/score.py)
  - [ood.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/ood.py)
  - [artifacts.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/artifacts.py)
- `Почему это важно`:
  Это делает open-set режим наблюдаемым и управляемым:
  - raw ranking не загрязняется пороговой логикой;
  - OOD policy хранится в artifact metadata;
  - analysis и QA могут отдельно смотреть raw router diagnostics и
    reject-option.
- `Статус`: `confirmed`

#### A-LOGIC-S003

- `ID`: `A-LOGIC-S003`
- `Приоритет`: `P2`
- `Категория`: `logic`
- `Зона`: `input normalization`
- `Факт`:
  В production input-layer есть явная детерминированная нормализация:
  из relation выбирается одна строка на `source_id` через
  `ROW_NUMBER() OVER (...)` и quality-aware порядок:
  - `parallax_over_error DESC`
  - `ruwe ASC`
  - `validation_factor DESC`
  - затем стабильные физические tie-break поля.
- `Evidence`:
  - [input_data.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/input_data.py)
- `Почему это важно`:
  Это снижает случайность входного batch и делает поведение pipeline
  воспроизводимее, чем простое `SELECT DISTINCT` или произвольный выбор
  одной строки на `source_id`.
- `Статус`: `confirmed`

#### A-LOGIC-S004

- `ID`: `A-LOGIC-S004`
- `Приоритет`: `P2`
- `Категория`: `logic`
- `Зона`: `production vs research split`
- `Факт`:
  Production runtime, comparative benchmark и heavy generalization
  validation разведены по отдельным слоям:
  - production: [src/priority_pipeline](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline)
  - benchmark/snapshot: [analysis/model_comparison](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison)
  - heavy validation: [analysis/model_validation](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_validation)
- `Evidence`:
  - [repository_state_policy_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/repository_state_policy_ru.md)
  - [model_validation_protocol_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/model_validation_protocol_ru.md)
  - [analysis/model_comparison/cli.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/cli.py)
- `Почему это важно`:
  Это уменьшает риск того, что heavy validation или research-search
  случайно прорастут в production runtime. Для ВКР и защиты это сильная
  архитектурная позиция: benchmarking и anti-overfitting checks реально
  выделены в отдельные контуры.
- `Статус`: `confirmed`

#### A-PHYS-S001

- `ID`: `A-PHYS-S001`
- `Приоритет`: `P2`
- `Категория`: `physics`
- `Зона`: `class priors vs outcome`
- `Факт`:
  Доминирование `K` в current runtime нельзя объяснить простым “ручным
  завышением” class prior.

  В production-коде priors заданы так:
  - `M = 1.00`
  - `K = 0.95`
  - `G = 0.80`
  - `F = 0.65`

  То есть формально `M` получает даже более сильный prior, чем `K`.
  Несмотря на это, в safe-run `limit=5000` именно `K` остаётся
  доминирующим массовым классом по `HIGH+MEDIUM` зоне.
- `Evidence`:
  - [decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/decision.py)
  - live safe-run `run_pipeline(engine, limit=5000, persist=False)` на
    19 марта 2026 года
- `Почему это важно`:
  Это хороший научный сигнал для `V1`: итоговое положение `K` идёт не
  из грубого хардкода одного коэффициента, а из совместного действия:
  - host-model signal;
  - astrophysical priors;
  - observability/follow-up факторов.
- `Статус`: `confirmed`

#### A-ML-S001

- `ID`: `A-ML-S001`
- `Приоритет`: `P2`
- `Категория`: `ml`
- `Зона`: `validation discipline`
- `Факт`:
  В проекте уже есть отдельный validation-layer, который реально
  подсвечивает рискованные модели, а не просто повторяет benchmark.

  Например:
  - `baseline_random_forest` получает `HIGH` risk в heavy validation из-за
    больших gap по `pr_auc` и `precision_at_k`;
  - `main_contrastive_v1` и `legacy` получают более умеренные risk-level;
  - отдельные repeated split diagnostics сохранены как versioned
    артефакты.
- `Evidence`:
  - [model_validation_2026-03-14_fast_v1_validation_report.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/model_validation/model_validation_2026-03-14_fast_v1_validation_report.md)
  - [baseline_comparison_2026-03-13_vkr30_cv10_generalization_audit.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10_generalization_audit.md)
  - [model_validation_protocol_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/model_validation_protocol_ru.md)
- `Почему это важно`:
  Это сильная сторона проекта: риск переобучения и split-instability не
  замалчивается, а уже вынесен в отдельный observable layer.
- `Статус`: `confirmed`

#### A-ML-S002

- `ID`: `A-ML-S002`
- `Приоритет`: `P2`
- `Категория`: `ml`
- `Зона`: `benchmark reproducibility`
- `Факт`:
  Свежий current benchmark-probe на 19 марта воспроизвёл versioned
  benchmark-артефакты `vkr30_cv10` по supervised метрикам и quality
  summary без заметного расхождения.
- `Evidence`:
  - current run:
    `python -m analysis.model_comparison --run-name audit_probe_2026-03-19_current --skip-snapshot`
  - [baseline_comparison_2026-03-13_vkr30_cv10.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10.md)
- `Почему это важно`:
  Это локализует текущий drift:
  supervised benchmark воспроизводится стабильно, а значит основная
  проблема лежит не “везде сразу”, а ближе к runtime/snapshot/shortlist
  semantics и к их versioned артефактам.
- `Статус`: `confirmed`

## 8. Код, типизация, naming и тесты

### Findings

#### A-CODE-001

- `ID`: `A-CODE-001`
- `Приоритет`: `P1`
- `Категория`: `code`
- `Зона`: `typing gate`
- `Факт`:
  Полный type-gate по проекту сейчас уже не зелёный. На свежем полном
  запуске `mypy src tests analysis` упал на новом quality-блоке:
  `analysis/model_comparison/quality.py` передаёт обычные `str` туда,
  где typed contracts требуют `Literal['train', 'test']` и
  `Literal['overall', 'classwise']`.
- `Evidence`:
  - [quality.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/quality.py)
  - full run 19 марта 2026 года:
    `./venv/bin/mypy src tests analysis`
- `Почему это важно`:
  Проект декларирует строгую типизацию как часть инженерного качества.
  Пока `ruff` и `pytest` зелёные, но full `mypy` уже нет. Это не ломает
  runtime напрямую, но означает, что типовой quality-gate больше нельзя
  считать надёжно зелёным после последних изменений comparison-layer.
- `Статус`: `confirmed`

#### A-CODE-002

- `ID`: `A-CODE-002`
- `Приоритет`: `P2`
- `Категория`: `code`
- `Зона`: `input layer ownership`
- `Факт`:
  Основная реализация input-layer живёт прямо в
  [src/input_layer/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/input_layer/__init__.py),
  и этот модуль уже совмещает:
  - environment bootstrap;
  - relation/schema validation;
  - SQL aggregation;
  - registry DDL;
  - registry upsert;
  - CLI parsing;
  - `main()`.

  При этом package уже имеет отдельный
  [src/input_layer/__main__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/input_layer/__main__.py),
  так что реализация в `__init__` выглядит как перегруз и смешение
  публичного package API с внутренней логикой.
- `Evidence`:
  - [src/input_layer/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/input_layer/__init__.py) — `692` строки
  - дерево файлов `src/input_layer/*`
- `Почему это важно`:
  Это уже не про “любимый стиль”, а про ownership и onboarding clarity:
  новому человеку трудно понять, где у package public surface, а где
  реальная бизнес-логика input validation.
- `Статус`: `confirmed`

#### A-CODE-003

- `ID`: `A-CODE-003`
- `Приоритет`: `P2`
- `Категория`: `code`
- `Зона`: `snapshot layer size`
- `Факт`:
  [analysis/model_comparison/snapshot.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/snapshot.py)
  уже совмещает слишком много обязанностей в одном модуле:
  - retrain benchmark-моделей;
  - scoring host-ветки;
  - metadata attachment;
  - summary/top-table assembly;
  - markdown generation;
  - artifact persistence;
  - orchestration `run_snapshot_comparison(...)`.
- `Evidence`:
  - [snapshot.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/snapshot.py) — `620` строк
- `Почему это важно`:
  Это снижает локальную читаемость comparison-layer и усложняет аудит:
  reproducibility drift, training semantics и artifact-writing живут в
  одном месте, вместо более узких модулей по ответственности.
- `Статус`: `confirmed`

#### A-CODE-004

- `ID`: `A-CODE-004`
- `Приоритет`: `P2`
- `Категория`: `code`
- `Зона`: `reporting layer size`
- `Факт`:
  [analysis/model_comparison/reporting.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/reporting.py)
  уже не только “reporting”. В одном файле собраны:
  - benchmark metrics aggregation;
  - threshold quality aggregation;
  - search summary;
  - generalization diagnostics;
  - markdown rendering;
  - artifact persistence.
- `Evidence`:
  - [reporting.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/reporting.py) — `615` строк
- `Почему это важно`:
  По смыслу это уже не один слой ответственности. Модуль остаётся
  функционально полезным, но начинает расти в “сервис-комбайн”, что
  осложняет точечную проверку и сопровождение.
- `Статус`: `confirmed`

#### A-TEST-001

- `ID`: `A-TEST-001`
- `Приоритет`: `P2`
- `Категория`: `tests`
- `Зона`: `orchestration tests`
- `Факт`:
  Часть тестового дерева comparison/model-validation уже ушла в крупные
  orchestration-heavy сценарии с большим числом dummy objects и
  monkeypatch-заглушек.

  Наиболее заметные примеры:
  - [tests/test_model_comparison_cli.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/test_model_comparison_cli.py) — `535` строк
  - [tests/test_model_validation.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/test_model_validation.py) — `452` строки
  - [tests/test_priority_pipeline_db_integration.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/test_priority_pipeline_db_integration.py) — `428` строк
- `Evidence`:
  - размер файлов из текущего inventory
  - структура тестов в перечисленных модулях
- `Почему это важно`:
  Полный `pytest` сейчас зелёный (`155 passed`), но такие тесты уже
  ухудшают:
  - читаемость;
  - локализацию поломок;
  - onboarding для нового человека;
  - поддержку тестов при изменении orchestration-контрактов.
- `Статус`: `confirmed`

#### A-NAMING-001

- `ID`: `A-NAMING-001`
- `Приоритет`: `P2`
- `Категория`: `naming`
- `Зона`: `package/file naming`
- `Факт`:
  В проекте появился naming drift между:
  - package-based entrypoints (`__main__.py`);
  - реализацией в `__init__.py`;
  - legacy facade-именами верхнего уровня;
  - research-layer и production-layer.

  Самый яркий текущий пример —
  [src/input_layer/__init__.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/input_layer/__init__.py),
  где `__init__` выглядит не как package surface, а как основной рабочий
  модуль.

  Дополнительно рабочее дерево показывает живой переход от старых
  верхнеуровневых фасадов (`src/input_layer.py`, `src/model_comparison.py`,
  `src/star_orchestrator.py` и др.) к package entrypoints, что повышает
  риск путаницы в текущем состоянии.
- `Evidence`:
  - дерево `src/*`
  - текущее `git status --short` на 19 марта 2026 года
- `Почему это важно`:
  Это не ломает correctness напрямую, но заметно ухудшает onboarding:
  новичку трудно быстро понять, что является canonical import path,
  что является CLI entrypoint, а что legacy facade или transitional state.
- `Статус`: `confirmed`

#### A-TEST-S001

- `ID`: `A-TEST-S001`
- `Приоритет`: `P2`
- `Категория`: `tests`
- `Зона`: `test feedback loop`
- `Факт`:
  Несмотря на рост отдельных test-модулей, полный test-suite проекта
  остаётся быстрым и зелёным.
- `Evidence`:
  - full run 19 марта 2026 года:
    `./venv/bin/pytest -q`
  - результат: `155 passed in 15.17s`
- `Почему это важно`:
  Это сильная инженерная сторона текущего состояния: тесты не только
  существуют, но и дают достаточно быстрый feedback loop для локальной
  разработки и аудита.
- `Статус`: `confirmed`

#### A-TEST-S002

- `ID`: `A-TEST-S002`
- `Приоритет`: `P2`
- `Категория`: `tests`
- `Зона`: `test runtime concentration`
- `Факт`:
  Полный runtime test-suite в основном концентрируется в нескольких
  smoke-тестах, которые реально прогоняют обучение baseline-моделей:
  - `test_run_random_forest_baseline_returns_common_score_contract`
  - `test_run_mlp_baseline_returns_common_score_contract`
  - `test_run_mlp_baseline_handles_small_split_without_early_stopping_failure`

  Остальная масса тестов заметно легче.
- `Evidence`:
  - full run 19 марта 2026 года:
    `./venv/bin/pytest -q --durations=15`
  - slowest durations:
    `6.50s`, `2.70s`, `2.19s` у перечисленных тестов
- `Почему это важно`:
  Это скорее хороший сигнал, чем проблема: дерево тестов пока не стало
  “тяжёлым по всему фронту”. Основная цена полного прогона связана с
  немногими реалистичными fit-smoke сценариями, а не с общей
  неуправляемостью test-suite.
- `Статус`: `confirmed`

#### A-DOC-S001

- `ID`: `A-DOC-S001`
- `Приоритет`: `P2`
- `Категория`: `docs`
- `Зона`: `docstring coverage`
- `Факт`:
  В `src/` и `analysis/` не найдено публичных top-level классов и
  функций без docstring.
- `Evidence`:
  - AST-проверка 19 марта 2026 года по деревьям
    [src](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src) и
    [analysis](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis)
  - результат: `MISSING_DOCSTRINGS 0`
- `Почему это важно`:
  Это хороший сигнал по readability и handoff quality: проект уже
  выглядит как кодовая база, где публичный слой не оставлен без
  пояснений.
- `Статус`: `confirmed`

#### A-NOTEBOOK-003

- `ID`: `A-NOTEBOOK-003`
- `Приоритет`: `P2`
- `Категория`: `notebook`
- `Зона`: `notebook hygiene`
- `Факт`:
  Текущие notebooks открываются и читаются, но при programmatic scan
  `nbformat` выдаёт предупреждение:
  `MissingIDFieldWarning: Cell is missing an id field`.
- `Evidence`:
  - programmatic read 19 марта 2026 года через `nbformat`
    по [notebooks/eda](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda)
- `Почему это важно`:
  Это пока не runtime-проблема, но уже технический debt presentation
  layer: в будущих версиях `nbformat` отсутствие `cell id` станет более
  жёсткой incompatibility.
- `Статус`: `confirmed`

#### A-NOTEBOOK-004

- `ID`: `A-NOTEBOOK-004`
- `Приоритет`: `P2`
- `Категория`: `notebook`
- `Зона`: `summary notebook portability`
- `Факт`:
  Итоговый notebook
  [04_model_comparison_summary.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/04_model_comparison_summary.ipynb)
  уже хорошо играет роль summary-layer, но остаётся жёстко привязанным к
  фиксированным versioned run-name и содержит очень крупные setup /
  presentation code-cells:
  - первая code-cell — `229` строк;
  - вторая code-cell — `146` строк;
  - shortlist/build logic также сосредоточена в крупной отдельной cell.
- `Evidence`:
  - programmatic notebook scan 19 марта 2026 года
  - markers:
    `BENCHMARK_RUN_NAME = "baseline_comparison_2026-03-13_vkr30_cv10"`
    и
    `SNAPSHOT_RUN_NAME = "baseline_comparison_2026-03-13_vkr30_cv10_limit5000"`
- `Почему это важно`:
  Notebook остаётся полезным для защиты, но его переносимость и
  сопровождаемость ниже, чем могла бы быть у более thin presentation
  layer, который меньше завязан на фиксированные artifact names и
  крупные setup-cells.
- `Статус`: `confirmed`

#### A-DOC-002

- `ID`: `A-DOC-002`
- `Приоритет`: `P2`
- `Категория`: `docs`
- `Зона`: `presentation narrative`
- `Факт`:
  В presentation-layer уже появилась тонкая narrative-неоднозначность.
  В [docs/presentation/vkr_slides_draft_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/presentation/vkr_slides_draft_ru.md)
  одновременно присутствуют тезисы:
  - `Top shortlist основной модели состоит из компактных M dwarf...`
  - финальный operational вывод проекта: `K dwarf -> M dwarf -> G dwarf`

  Оба тезиса могут быть совместимы, но только если явно развести:
  - верхушку contrastive-ranking;
  - массовый class-level operational priority.

  В текущем draft это разведение проговорено недостаточно явно.
- `Evidence`:
  - [docs/presentation/vkr_slides_draft_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/presentation/vkr_slides_draft_ru.md)
  - [docs/model_comparison_findings_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/model_comparison_findings_ru.md)
- `Почему это важно`:
  На защите это легко воспринимается как противоречие, хотя по сути речь
  идёт о двух разных уровнях интерпретации.
- `Статус`: `confirmed`

#### A-NAMING-S001

- `ID`: `A-NAMING-S001`
- `Приоритет`: `P2`
- `Категория`: `naming`
- `Зона`: `stage naming`
- `Факт`:
  На верхнем уровне naming проекта в целом помогает ориентироваться:
  - EDA notebooks выстроены последовательно как
    `00_data_extraction_and_preprocessing` →
    `01_host_eda_overview` →
    `02_router_readiness` →
    `03_host_vs_field_contrastive` →
    `04_model_comparison_summary`;
  - analysis-layer тоже разделён по смыслу на
    `host_eda`, `router_eda`, `model_comparison`, `model_validation`.
- `Evidence`:
  - [notebooks/eda](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda)
  - [analysis](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis)
- `Почему это важно`:
  Это смягчает общий onboarding-cost: несмотря на отдельные naming drift
  в runtime/package-слоях, high-level карта проекта уже читается
  достаточно логично.
- `Статус`: `confirmed`
