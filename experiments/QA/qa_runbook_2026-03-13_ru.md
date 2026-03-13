# QA Runbook

Дата фиксации: 13 марта 2026 года

Цель:
- иметь один канонический порядок полного QA-прогона проекта;
- не сводить проверку только к `pytest`;
- разделять дешёвые, средние и дорогие шаги, чтобы QA был воспроизводимым.

Принцип:
- шаги идут строго по порядку;
- после каждого шага фиксируются выводы;
- если что-то падает, это не “чинится на лету” в рамках аудита, а записывается в backlog.

## Предусловия

- рабочая директория: `/Users/evgeniikuznetsov/Desktop/dspro-vkr`
- Python/venv проекта доступны
- для DB-backed шагов нужен рабочий Postgres через `.env` или переменные окружения
- для notebook validation нужен установленный Jupyter stack из текущего окружения

## Полный порядок прогона

1. Зафиксировать текущее состояние workspace.
Команды:
```bash
git status --short
git status --ignored --short
```
Что проверить руками:
- dirty worktree;
- есть ли untracked код/тесты/docs, которые участвуют в текущем состоянии проекта;
- нет ли неожиданного мусора в корне.

2. Проверить статическое качество кода.
Команды:
```bash
./venv/bin/python -m ruff check src analysis tests
./venv/bin/python -m mypy src analysis tests
./venv/bin/python -m compileall -q src analysis tests
```
Что считается нормой:
- `ruff` зелёный;
- `mypy` без новых ошибок;
- все Python-модули компилируются.

3. Прогнать полный `pytest` как базовую regression-проверку.
Команда:
```bash
./venv/bin/python -m pytest -q
```
Что проверить руками:
- количество `passed/skipped`;
- не появилось ли новых `xfail/skip`-подобных обходов;
- не спрятались ли важные DB-шаги за массовым skip.

4. Отдельно проверить DB-backed сценарии.
Команда:
```bash
./venv/bin/python -m pytest -q -m db_integration
```
Что считается нормой:
- либо тесты реально проходят на временной схеме Postgres;
- либо явно и ожидаемо `skip`-аются из-за отсутствия доступной БД.
Что смотреть дополнительно:
- нет ли неожиданного SQL-падения;
- не оставляет ли тест мусорных схем и таблиц.

5. Прогнать регрессии на production artifacts.
Команда:
```bash
./venv/bin/python -m pytest -q tests/test_runtime_artifacts.py
```
Что считается нормой:
- `router_gaussian_params.json` соответствует posterior-aware contract;
- `model_gaussian_params.json` соответствует contrastive host-vs-field contract.

6. Прогнать targeted smoke по математическому ядру.
Команда:
```bash
./venv/bin/python -m pytest -q \
  tests/test_gaussian_router.py \
  tests/test_model_gaussian.py \
  tests/test_star_orchestrator.py \
  tests/test_decision_layer_calibrator.py
```
Что проверить руками:
- нет ли регресса в router posterior/OOD;
- нет ли регресса в host-vs-field scoring;
- не изменился ли decision-layer contract.

7. Прогнать targeted smoke по production pipeline и input-layer.
Команда:
```bash
./venv/bin/python -m pytest -q \
  tests/test_priority_pipeline.py \
  tests/test_priority_pipeline_db_integration.py \
  tests/test_input_layer_db_integration.py
```
Что проверить руками:
- mini-batch orchestration собирается;
- persist-контур жив;
- input registry не ломается на валидном и сломанном relation.

8. Прогнать targeted smoke по comparison-layer.
Команда:
```bash
./venv/bin/python -m pytest -q tests/test_model_comparison_*.py
```
Что проверить руками:
- split/CV/search/reporting contract остаётся целым;
- CLI не расходится с каноническим protocol;
- snapshot/reporting артефакты продолжают собираться.

9. Запустить канонический benchmark comparison-layer.
Команда:
```bash
./venv/bin/python src/model_comparison.py --skip-snapshot
```
Что проверить руками:
- в `experiments/model_comparison/` появились/обновились markdown и CSV;
- есть `summary`, `classwise`, `search_summary`;
- run-name и protocol summary в markdown соответствуют актуальному контракту.

10. Запустить operational snapshot preview.
Команда:
```bash
./venv/bin/python src/model_comparison.py --snapshot-limit 5000 --snapshot-top-k 25
```
Что проверить руками:
- собран snapshot markdown;
- собраны `*_snapshot_summary.csv`, `*_priority.csv`, `*_top.csv`;
- результаты не выглядят явно сломанными по распределению tier-ов и top candidates.

11. Сделать ручной математический sanity check benchmark-артефактов.
Что проверить руками:
- все вероятности и нормированные score-колонки лежат в разумных диапазонах;
- `roc_auc`, `pr_auc`, `precision@k`, `brier` не содержат `NaN/inf`;
- `best_params` и `best_cv_score` сохранены в search summary;
- ranking benchmark и snapshot не противоречат друг другу качественно.

12. Сделать ручной sanity check production scoring semantics.
Что проверить руками:
- `HIGH/MEDIUM/LOW` соответствуют `final_score`;
- `UNKNOWN` не попадает в host-scoring;
- low-known ветка не маскируется под валидный host ranking;
- `host_posterior`, `final_score`, `reason_code` согласованы между собой.

13. Проверить исполняемость summary notebook.
Команда:
```bash
./venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/eda/04_model_comparison_summary.ipynb
```
Что считается нормой:
- notebook исполняется без ошибок;
- ссылки на актуальные benchmark/snapshot run-name не сломаны.

14. Проверить README-команды и основные пути к артефактам.
Что проверить руками:
- команды из README существуют в текущем дереве проекта;
- ссылки на docs/notebooks/experiments не устарели;
- пользователь может понять базовый workflow без скрытых договорённостей.

15. Зафиксировать findings в QA-лог.
Что записать:
- что прошло;
- что упало;
- что выглядит спорно, но терпимо;
- какие артефакты или файлы требуют отдельного решения.

## Интерпретация результатов

- Зелёный `ruff + mypy + pytest` — это базовый барьер качества, но не полный аудит проекта.
- Полный QA считается завершённым только если дополнительно просмотрены benchmark/snapshot-артефакты и подтверждена актуальность README/notebooks/docs.
- Если дорогие шаги `9-13` пропущены, такой прогон нужно считать сокращённым, а не полным.

## Минимально обязательный набор перед важной поставкой

1. `ruff`
2. `mypy`
3. `pytest -q`
4. `pytest -q -m db_integration`
5. benchmark `src/model_comparison.py --skip-snapshot`
6. snapshot preview `src/model_comparison.py --snapshot-limit 5000 --snapshot-top-k 25`
7. ручной просмотр summary/search/snapshot CSV и markdown

## Краткий вывод

- Для этого проекта правильный QA — ступенчатый.
- Автоматические тесты покрывают важную часть рисков, но не заменяют проверку артефактов и методической интерпретации.
- Канонический полный прогон должен сочетать статику, тесты, benchmark, snapshot и ручной sanity review.
