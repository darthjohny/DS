# План всестороннего аудита проекта

Дата фиксации: 19 марта 2026 года

## 1. Назначение документа

Этот документ фиксирует стартовый план полного аудита проекта перед
следующей волной правок и передзащитной стабилизацией.

Главная цель текущей волны:

- не переписывать проект немедленно;
- сначала пошагово изучить код, логику, математику, физику, тесты,
  notebooks и документацию;
- зафиксировать наблюдения, сильные стороны, слабые места и evidence;
- только после завершения audit-wave перейти к отдельному плану правок.

Иными словами, это документ именно про `изучение и фиксацию`, а не про
рефакторинг.

Operational карта покрытия аудита вынесена отдельно в:

- [project_audit_mapping_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/project_audit_mapping_ru.md)

Этот документ задаёт:

- surfaces, которые мы обязаны проверить;
- ownership ключевых файлов и слоёв;
- risk map;
- audit coverage map.

## 2. Принцип текущей волны

До завершения audit-wave мы придерживаемся жёсткого правила:

1. Сначала смотрим.
2. Потом записываем findings.
3. Потом сверяем findings с артефактами и прогонами.
4. И только после этого составляем отдельный fix-plan.

Что сейчас не делаем:

- не устраиваем массовый рефакторинг;
- не правим “на всякий случай”;
- не оптимизируем код без доказанного смысла;
- не смешиваем findings и будущие исправления в одном документе.

### 2.1 Внешние ориентиры, на которые опирается план

Перед запуском audit-wave план был дополнительно сверен не только с
внутренней логикой проекта, но и с внешними инженерными ориентирами:

- [Martin Fowler — Test Pyramid](https://martinfowler.com/bliki/TestPyramid.html)
- [Martin Fowler / Ham Vocke — The Practical Test Pyramid](https://martinfowler.com/articles/practical-test-pyramid.html)
- [pytest docs — Good Integration Practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html)
- [Software Engineering at Google — Test Size and Scope](https://abseil.io/resources/swe-book/html/ch11.html)
- [SEI CMU — Common Testing Problems](https://www.sei.cmu.edu/blog/common-testing-problems-pitfalls-to-prevent-and-mitigate/)
- [CMU 17-214 — Testing and Object Methods](https://www.cs.cmu.edu/~charlie/courses/17-214/2018-fall/slides/20180906-testing-and-object-methods.pdf)
- [UC Berkeley CS W169A](https://www2.eecs.berkeley.edu/Courses/CSW169A/)

Из этого в audit-plan добавлены следующие практические акценты:

- перед deep-review нужно выделять critical paths и рискованные зоны;
- при аудите тестов надо смотреть не только на `unit/integration`, но и
  на `size/scope` тестов;
- важно проверять не только correctness, но и reproducibility:
  test data, environments и artifacts должны быть под конфигурационным
  контролем;
- нужен отдельный взгляд на `fast smoke suite` vs `full suite`, а не
  только на общее количество тестов;
- большое число broad-stack tests само по себе не является плюсом, если
  они дублируют более дешёвые и быстрые проверки.

## 3. Текущее audit-основание

### 3.1 Канонический проектный контур

Текущим источником истины для current state считаются:

- [README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/README.md)
- [repository_state_policy_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/repository_state_policy_ru.md)
- [orchestrator_host_prioritization_canon_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/orchestrator_host_prioritization_canon_ru.md)
- [model_comparison_protocol_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/model_comparison_protocol_ru.md)
- [model_comparison_findings_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/model_comparison_findings_ru.md)
- [model_validation_protocol_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/model_validation_protocol_ru.md)

### 3.2 Канонические кодовые зоны

Production и runtime:

- [src/router_model](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model)
- [src/host_model](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/host_model)
- [src/priority_pipeline](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline)
- [src/decision_calibration](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/decision_calibration)
- [src/input_layer](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/input_layer)
- [src/infra](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/infra)

Research и validation:

- [analysis/model_comparison](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison)
- [analysis/model_validation](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_validation)
- [analysis/host_eda](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/host_eda)
- [analysis/router_eda](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/router_eda)

Полная surface/ownership map для этих зон вынесена в:

- [project_audit_mapping_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/project_audit_mapping_ru.md)

### 3.3 Канонические notebooks

- [00_data_extraction_and_preprocessing.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/00_data_extraction_and_preprocessing.ipynb)
- [01_host_eda_overview.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/01_host_eda_overview.ipynb)
- [02_router_readiness.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/02_router_readiness.ipynb)
- [03_host_vs_field_contrastive.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/03_host_vs_field_contrastive.ipynb)
- [04_model_comparison_summary.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/04_model_comparison_summary.ipynb)

### 3.4 Канонические experiment-артефакты

Сравнение моделей:

- [experiments/model_comparison/README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/model_comparison/README.md)
- каноническая wave:
  [baseline_comparison_2026-03-13_vkr30_cv10.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10.md)
- associated CSV-артефакты той же волны:
  `summary`, `classwise`, `search_summary`, `thresholds`,
  `quality_summary`, `quality_classwise`, `confusion_matrices`,
  `generalization`, `dataset_validation`
- operational snapshot:
  [baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/model_comparison/baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot.md)

QA:

- [experiments/QA/README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/QA/README.md)
- current full QA wave dated `2026-03-13`

Validation:

- [model_validation_2026-03-14_fast_v1_validation_report.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/model_validation/model_validation_2026-03-14_fast_v1_validation_report.md)
- [orchestrator_baseline_2026-03-14_limit5000_summary.csv](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/model_validation/orchestrator_baseline_2026-03-14_limit5000_summary.csv)

### 3.5 Важная operational оговорка

На момент старта аудита рабочее дерево проекта уже является `dirty`.
Это значит:

- findings нужно привязывать к текущему состоянию файлов в workspace;
- при этом канонические версии логики всё равно определяются через
  README, policy-docs и канонические experiment artifacts;
- в audit-findings нужно отделять:
  `current state behavior`,
  `канонический runtime-contract`,
  `локальные незакоммиченные изменения`.

### 3.6 Critical paths и surfaces, которые обязательно покрываем аудитом

До начала file-by-file review фиксируем поверхности, которые считаются
наиболее важными для correctness и защитной позиции:

- preprocessing lineage и канонические training relations;
- router scoring и posterior-aware routing;
- OOD / Unknown gating;
- host-model scoring и contrastive vs baseline comparison;
- decision layer и факторы ранжирования;
- DB-backed persist и runtime results contract;
- comparison artifacts и model-validation artifacts;
- summary notebooks и финальные выводы для ВКР.

Если в ходе audit-wave найдётся проблема в одном из этих critical paths,
она по умолчанию приоритизируется выше локальной косметики.

Каноническая детализация этих critical paths дана в:

- [project_audit_mapping_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/project_audit_mapping_ru.md)

## 4. Блоки аудита

### Блок 1. Логика и архитектура

Главный вопрос:

- соответствует ли текущий pipeline реальной задаче ВКР?

Проверяем:

- preprocessing -> router -> OOD -> host-model -> decision layer -> output
- comparison-layer и validation-layer
- связь notebooks с runtime-кодом
- отсутствие логических разрывов и дублирования responsibility

Выход:

- карта архитектурной логики
- findings по логическим несоответствиям

### Блок 2. Математика и физика

Главный вопрос:

- физически и математически адекватно ли текущая `V1` ранжирует объекты?

Проверяем:

- порядок `K / M / G`
- class priors и quality modifiers
- роль `OOD`
- thresholds и quality-модуль
- общую согласованность ranking-а с формулировкой задачи

Выход:

- findings по физике и математике
- перечень сильных и спорных мест

### Блок 3. ML-качество, воспроизводимость и боевые прогоны

Главный вопрос:

- устойчиво ли ведут себя модели и насколько они адекватны на текущих
  данных и текущих артефактах?

Проверяем:

- benchmark
- snapshot
- threshold-based quality
- generalization
- несколько боевых прогонов production-контура
- reproducibility артефактов и config control для data/test environments
- соответствие runtime-результатов каноническим experiment artifacts

Выход:

- findings по переобучению / недообучению / stability
- findings по reproducibility и environment control
- список обязательных оговорок для защиты

### Блок 4. Код

Главный вопрос:

- чистый ли код, читаемый ли он и соответствует ли целевой задаче?

Проверяем:

- modularity
- явное дублирование
- типизацию
- линтеры
- naming файлов, модулей, функций и exported API
- названия сущностей
- уместность комментариев
- сложность функций и модулей

Выход:

- code-review findings
- список потенциальных упрощений

### Блок 5. Тесты

Главный вопрос:

- корректны ли тесты и не перегружено ли дерево тестирования?

Проверяем:

- актуальность тестов
- соответствие текущему контракту
- дублирование тестов
- недопокрытие critical paths
- избыточность и хрупкость
- распределение тестов по `size/scope`
- не превращается ли suite в inverted pyramid / ice-cream cone
- есть ли быстрый smoke-контур и отдельно более тяжёлый full-контур
- соответствуют ли pytest-конфигурация, import-режим и package layout
  нормальным практикам reproducible test execution

Выход:

- test-audit findings
- карта пробелов и перегрузок
- отдельная оценка `fast feedback` vs `heavy coverage`

### Блок 6. Ноутбуки и выводы

Главный вопрос:

- соответствуют ли notebooks коду, логике и итоговой задаче?

Проверяем:

- что действительно считается в notebooks
- нет ли ручных логических обходов
- не расходятся ли notebooks с каноническими артефактами
- корректны ли краткие выводы для ВКР

Выход:

- notebook findings
- список trustworthy notebooks и risky cells

### Блок 7. Документация и presentation-layer

Главный вопрос:

- готов ли проект по документации и оформлению к Git и защите?

Проверяем:

- README
- protocol/findings/state-policy docs
- docstring policy
- согласованность naming в docs, notebooks и artifacts
- согласованность документации с кодом и артефактами

Выход:

- docs findings
- список дыр в documentation layer

## 5. Формат findings

Для фиксации наблюдений используется отдельный документ:

- [project_audit_findings_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/project_audit_findings_ru.md)

Пока audit-wave не завершена, туда складываются:

- факты;
- evidence;
- ссылки на файлы и артефакты;
- уровень серьёзности;
- предварительная интерпретация.

План исправлений в этот документ не вносится.

## 6. Приоритеты findings

- `P0` — ломает корректность результата или защитную позицию.
- `P1` — важная логическая, физическая, ML- или кодовая проблема.
- `P2` — полезная правка, но не блокер.
- `P3` — косметика.

Категории:

- `logic`
- `physics`
- `ml`
- `code`
- `tests`
- `notebook`
- `docs`
- `ops`

## 7. Порядок прохождения audit-wave

Порядок намеренно фиксируется таким:

1. inventory и current state
2. critical paths и risk mapping
3. логика и архитектура
4. математика и физика
5. ML, reproducibility и боевые прогоны
6. код
7. тесты
8. notebooks
9. документация
10. отдельный fix-plan

Это нужно, чтобы:

- не полировать код до проверки логики;
- не пропустить рискованные зоны из-за одинакового отношения ко всем
  файлам;
- не делать fixes до появления evidence;
- не смешивать ревью и рефакторинг.

## 8. Что будет считаться завершением audit-wave

Audit-wave можно считать завершённой, когда:

- по каждому из 7 блоков есть findings или явная отметка “критичных
  проблем не обнаружено”;
- findings опираются на реальные файлы, артефакты и прогоны;
- отдельно составлен fix-plan;
- fix-plan не содержит работ “ради красоты”, если для них нет evidence.

До этого момента данный документ считается рабочим runbook для аудита.
