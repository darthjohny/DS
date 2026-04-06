# Политика QA Для Notebook

Дата фиксации: `2026-04-05`

Связанные документы:

- [project_cleanup_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/project_cleanup_tz_ru.md)
- [analysis/notebooks/README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/README.md)
- [test_analysis_notebooks.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/notebooks/test_analysis_notebooks.py)

Внутренний микро-план notebook cleanup ведется вне публичного контура
репозитория.

## Цель

Разделить две разные задачи:

- быстрый smoke-контроль notebook как файлов проекта;
- осознанное исполнение важных notebook через `nbclient` в scoped QA.

Это нужно, чтобы:

- не делать вид, что обычный `pytest` выполняет тяжелые notebook;
- не терять повторяемую проверку тех notebook, которые реально участвуют в
  анализе результатов проекта;
- не смешивать активный обзорный слой и исследовательский архив.

## Что Считается Smoke-Проверкой

Файл [test_analysis_notebooks.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/notebooks/test_analysis_notebooks.py)
проверяет только активные notebook из каталогов
[eda](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/eda),
[research](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/research)
и
[technical](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical):

- файл является валидным JSON;
- все code-cell синтаксически компилируются;
- notebook не пустой.

Этот тест:

- не запускает notebook;
- не проверяет доступность внешних данных;
- не заменяет исполнение через `nbclient`.

## Какие Notebook Идут В Scoped `nbclient` QA

Следующие notebook считаются активными рабочими notebook, которые должны
исполняться через `nbclient`, когда затронут их связанный слой:

- `scoring_review.ipynb`
  из каталога `technical`,
  при изменениях scoring/reporting слоя;
- `model_pipeline_review.ipynb`
  из каталога `technical`,
  при изменениях pipeline review, benchmark/reporting и model artifact слоя;
- `final_decision_review.ipynb`
  из каталога `technical`,
  при изменениях final decision review, final decision artifacts и posthoc;
- `quality_gate_calibration.ipynb`
  из каталога `research`,
  при изменениях quality gate review и calibration policy;
- `host_priority_calibration_review.ipynb`
  из каталога `technical`,
  при изменениях host calibration review и host priority слоя;
- `priority_threshold_review.ipynb`
  из каталога `technical`,
  при изменениях ranking/priority threshold review;
- `coarse_ob_domain_shift.ipynb`
  из каталога `research`,
  при изменениях active `O/B` domain-shift review;
- `secure_o_tail.ipynb`
  из каталога `research`,
  при изменениях active review маленького надежного хвоста `O`.

## Какие Notebook По Умолчанию Остаются Только В Smoke-Контуре

Следующие notebook не считаются обязательными для регулярного `nbclient` QA
на каждом шаге уборки:

- `router_training.ipynb`
- `host_training.ipynb`
- `label_coverage.ipynb`

Их исполняют отдельно только тогда, когда меняются соответствующие source
relations или логика их обзорного слоя.

## Архивные Notebook

Notebook из
[analysis/notebooks/archive_research](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/archive_research)
не входят в активный notebook QA.

Они считаются исследовательским архивом и исполняются только по отдельному
запросу или при возвращении конкретной архивной ветки в активную работу.

## Правило Для Следующих Шагов

Во всех следующих микро-ТЗ:

- обычный `pytest` покрывает только smoke-проверку notebook;
- `nbclient` прогоняется адресно, когда меняется соответствующий notebook-слой;
- факт исполнения notebook через `nbclient` фиксируется в ответе и, при
  необходимости, в рабочей документации шага.
