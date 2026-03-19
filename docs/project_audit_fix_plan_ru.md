# План правок по итогам audit-wave

Дата фиксации: 19 марта 2026 года

## 1. Назначение документа

Этот документ переводит findings текущей audit-wave в **практический
план правок**.

Он отвечает на вопросы:

- что именно нужно исправлять;
- в каком порядке;
- зачем это нужно;
- какие зоны проекта будут затронуты;
- какой результат мы ожидаем;
- что в `V1` сознательно **не трогаем**, чтобы не раздувать объём работ.

Важно:

- это уже **не аудит**, а follow-up plan;
- здесь допустимы решения и рекомендуемые направления изменений;
- но правки всё ещё должны быть точечными, без массового
  рефакторинга “ради красоты”.

Аудит, на котором основан этот документ:

- [project_audit_plan_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/project_audit_plan_ru.md)
- [project_audit_mapping_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/project_audit_mapping_ru.md)
- [project_audit_findings_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/project_audit_findings_ru.md)
- [project_audit_synthesis_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/project_audit_synthesis_ru.md)

## 2. Рабочие принципы правок

### 2.1 Что считаем правильной стратегией

1. Сначала выравниваем **смыслы и контракты**.
2. Потом восстанавливаем **воспроизводимость и артефакты**.
3. Потом синхронизируем **notebooks / docs / защитный narrative**.
4. И только после этого делаем **точечную code hygiene-полировку**.

### 2.2 Что сознательно не делаем

В этой волне не нужно:

- переписывать весь `comparison-layer`;
- делать новый большой рефакторинг `src/*`;
- перерабатывать всю математику `V1` с нуля;
- внедрять `GMM`, новую all-star физику или следующий большой research layer;
- бороться со всеми `P2` сразу, если они не мешают correctness,
  explainability или defense readiness.

### 2.3 Базовая гипотеза этой волны

С учётом недавних изменений это нормально, что часть артефактов “поехала”:

- добавлялся новый comparison/quality слой;
- менялись некоторые семантики и контракты;
- оркестратор и его значения ещё не были доведены до полностью
  канонического production state.

Это не отменяет findings, но влияет на стратегию:

- задача не “доказать, что всё сломано”;
- задача — **правильно выровнять текущую V1**.

## 3. Приоритетная карта работ

### P1-волна: обязательно

Это то, что реально нужно выровнять, чтобы проект было безопасно
объяснять, воспроизводить и защищать:

1. восстановить full quality-gate (`mypy`);
2. зафиксировать каноническую current production semantics;
3. синхронизировать persist contract с runtime-факторами;
4. определить и закрепить текущую семантику `UNKNOWN`;
5. настроить оркестратор как канонический operational layer;
6. развести production outputs и comparison snapshot;
7. пересобрать канонические артефакты после выравнивания;
8. обновить notebooks/docs/slides под новые канонические артефакты.

### P2-волна: после стабилизации

Это полезные улучшения, но не первые по очереди:

- разукрупнение тяжёлых comparison-модулей;
- naming cleanup;
- notebook hygiene (`cell ids`, тонкий setup-layer);
- снижение размеров отдельных orchestration-heavy тестов.

## 4. План по волнам

## Волна 0. Вернуть инженерную опору

### Цель

Перед смысловыми правками вернуть проект в состояние, где quality-gates
снова дают чёткий сигнал.

### Что делаем

1. Починить текущий `mypy` regression из quality-layer.
2. Убедиться, что:
   - `ruff` зелёный;
   - `mypy` зелёный;
   - `pytest` зелёный.
3. Зафиксировать, что именно считаем канонической командой full-check.

### Что затронем

- [analysis/model_comparison/quality.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/quality.py)
- [analysis/model_comparison/contracts.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/contracts.py)
- при необходимости точечно tests, если проблема окажется в contract drift

### Почему это важно

Пока full `mypy` красный, любые следующие изменения уже хуже
контролируются.

### Ожидаемый результат

- full quality-gate снова зелёный;
- у команды есть единый “базовый сигнал”: проект хотя бы типово и
  тестово стабилен перед следующими правками.

## Волна 1. Зафиксировать каноническую current semantics

### Цель

Перестать жить в смеси:

- старого explanation-layer;
- нового runtime;
- частично обновлённых notebooks;
- частично обновлённых comparison-артефактов.

### Что делаем

1. Явно зафиксировать **current production formula** как источник истины.
2. Привести docs к одному состоянию:
   - что считает production;
   - что считает offline calibration;
   - что считает comparison-layer;
   - где эти слои совпадают, а где нет.
3. Отдельно зафиксировать текущую роль:
   - `quality_factor`
   - `reliability_factor`
   - `followup_factor`

### Что затронем

- [README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/README.md)
- [orchestrator_host_prioritization_canon_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/orchestrator_host_prioritization_canon_ru.md)
- [model_comparison_protocol_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/model_comparison_protocol_ru.md)
- [model_comparison_findings_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/model_comparison_findings_ru.md)
- при необходимости краткие комментарии в
  [decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/decision.py)

### Почему это важно

Сейчас самая опасная проблема проекта — не “падает код”, а смысловой
дрейф между runtime и explanation-layer.

### Ожидаемый результат

- один текущий канон production semantics;
- больше нет документа, который описывает старую формулу как current runtime.

## Волна 2. Синхронизировать runtime contract и persist

### Цель

Сделать так, чтобы persisted result-layer не терял важную часть
explainability.

### Что делаем

1. Решить судьбу `quality_factor`:
   - оставить как alias совместимости;
   - но явно хранить и `reliability_factor`, и `followup_factor`.
2. Обновить:
   - constants;
   - persist mapping;
   - SQL schema/migration;
   - docs по result tables.
3. Проверить, чтобы comparison / notebooks / exports не ломались от
   расширения result contract.

### Что затронем

- [src/priority_pipeline/constants.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/constants.py)
- [src/priority_pipeline/persist.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/persist.py)
- [src/priority_pipeline/decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/decision.py)
- SQL миграции в [sql](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql)
- docs result-schema

### Почему это важно

Иначе production runtime уже считает больше, чем умеет объяснить по
persisted artifacts.

### Ожидаемый результат

- итоговые result tables отражают реальную структуру runtime scoring;
- decomposition `final_score` больше не теряется после записи в БД.

## Волна 3. Зафиксировать V1-семантику `UNKNOWN`

### Цель

Убрать двусмысленность между spec и реальным runtime.

### Рекомендуемое решение для V1

Для этой волны **не расширять open-set глубже в input-layer**, а
сделать более прагматичное выравнивание:

1. считать, что current `UNKNOWN` — это router/OOD reject для
   **scoreable rows**;
2. structural missing-feature cases трактовать как отдельный input
   filtering case, а не как `UNKNOWN`;
3. отразить это в docs, QA и protective narrative.

### Почему такой выбор лучше для V1

- он меньше ломает текущую архитектуру;
- не тянет лишний redesign input runtime;
- честно описывает реальное поведение системы.

### Что затронем

- [docs/ood_unknown_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/ood_unknown_tz_ru.md)
- [src/priority_pipeline/input_data.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/input_data.py) — только если потребуется уточняющий comment/contract
- QA/docs/notebooks, где сейчас `UNKNOWN` объясняется слишком широко

### Ожидаемый результат

- open-set narrative перестаёт противоречить production runtime;
- `unknown_share` интерпретируется корректно.

## Волна 4. Настроить оркестратор как канонический operational слой

### Цель

Сделать оркестратор не “рабочим по умолчанию”, а **осмысленно
настроенным** под текущую V1.

### Что делаем

1. Перепроверить class priors, tier thresholds и factor contributions.
2. Явно решить, что именно в `V1` мы считаем:
   - retrieval-oriented ranking;
   - conservative positive classification;
   - final operational shortlist semantics.
3. Прогнать несколько safe production runs:
   - `limit=5000`
   - возможно один полный controlled run
4. Сопоставить:
   - class distribution;
   - `HIGH/MEDIUM/LOW`;
   - вклад факторов;
   - shortlist shape.
5. Зафиксировать канонические operational thresholds.

### Что затронем

- [src/priority_pipeline/decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/decision.py)
- возможно [src/priority_pipeline/constants.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/constants.py)
- comparison notebook и findings docs
- production QA artifacts

### Почему это важно

Ты уже правильно отметил: часть drift сейчас вполне объяснима тем, что
оркестратор ещё не доведён до окончательно канонической настройки.

### Ожидаемый результат

- у `V1` появляется осмысленный и воспроизводимый operational profile;
- high/medium/low tiers перестают быть “случайным следствием текущих
  коэффициентов”.

## Волна 5. Развести production outputs и comparison outputs

### Цель

Перестать смешивать два разных слоя:

- research/comparison snapshot;
- production operational result.

### Рекомендуемое решение

Не пытаться насильно сделать comparison snapshot равным production run.
Вместо этого:

1. оставить comparison snapshot как research-layer;
2. завести канонический production artifact/export для реального
   `run_pipeline(...)`;
3. использовать:
   - comparison snapshot — для сравнения моделей;
   - production result — для реального shortlist и финального ответа.

### Что делаем

1. Сделать отдельный versioned production-run artifact.
2. Если нужно, добавить export для production shortlist.
3. Перестроить summary notebook так, чтобы:
   - benchmark и snapshot оставались comparative;
   - итоговый shortlist шёл из production result.
4. Явно пометить в docs, какие артефакты research-only, а какие
   operational.

### Что затронем

- [analysis/model_comparison/snapshot.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/snapshot.py) — возможно только narrative/metadata side
- [notebooks/eda/04_model_comparison_summary.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/04_model_comparison_summary.ipynb)
- [docs/model_comparison_findings_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/model_comparison_findings_ru.md)
- [docs/presentation/vkr_slides_draft_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/presentation/vkr_slides_draft_ru.md)
- `experiments/model_validation/*` и/или `experiments/QA/*` как канон production layer

### Почему это важно

Именно здесь сейчас рождается самая неприятная путаница:
comparison artifact выглядит как боевой shortlist, хотя им не является.

### Ожидаемый результат

- production и comparison больше не маскируются друг под друга;
- shortlist можно показывать на защите без semantic caveat в каждой фразе.

## Волна 6. Пересобрать канонические артефакты

### Цель

После выравнивания смысла и контрактов зафиксировать **новую каноническую
версию артефактов**.

### Что делаем

1. Перезапускаем:
   - benchmark-wave;
   - quality artifacts;
   - snapshot-wave;
   - production safe-run / baseline;
   - нужные QA/model-validation прогоны.
2. Фиксируем новые versioned run-names.
3. Старые артефакты не удаляем без причины, но перестаём выдавать их за
   current canon.

### Что затронем

- [experiments/model_comparison](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/model_comparison)
- [experiments/model_validation](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/model_validation)
- [experiments/QA](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/QA)

### Почему это важно

Нельзя продолжать полировать docs поверх артефактов, которые уже не
соответствуют current behavior.

### Ожидаемый результат

- одна новая текущая каноническая wave;
- reproducibility drift становится либо устранённым, либо явно
  локализованным.

## Волна 7. Синхронизировать notebooks, docs и защитный narrative

### Цель

Сделать так, чтобы итоговые materials говорили ровно то, что реально
делает и показывает текущая V1.

### Что делаем

1. Обновить:
   - [04_model_comparison_summary.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/04_model_comparison_summary.ipynb)
   - [docs/model_comparison_findings_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/model_comparison_findings_ru.md)
   - [docs/presentation/vkr_slides_draft_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/presentation/vkr_slides_draft_ru.md)
   - [README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/README.md)
2. Явно развести в narrative:
   - benchmark winner;
   - production ranking winner;
   - top-of-ranking behavior;
   - class-level operational priority.
3. Аккуратно переписать физический вывод так, чтобы:
   - он был честным;
   - он не переутверждал `G`;
   - он не путал contrastive top candidates и массовый class-level слой.

### Почему это важно

Сейчас защита может “споткнуться” не о код, а о narrative inconsistency.

### Ожидаемый результат

- итоговые materials можно показывать без постоянных устных оговорок.

## Волна 8. Точечная hygiene-полировка

### Цель

Убрать то, что реально мешает сопровождаемости, но **не превращать эту
волну в рефакторинг всей кодовой базы**.

### Что делаем

1. При необходимости разукрупняем только **те модули, которые всё равно
   трогаем в текущей волне**:
   - `analysis/model_comparison/reporting.py`
   - `analysis/model_comparison/snapshot.py`
   - возможно `src/input_layer/__init__.py`
2. Нормализуем package/file naming там, где это реально путает
   ownership.
3. Делаем notebook hygiene:
   - `cell ids`
   - тоньше setup/config
4. Проверяем, нужен ли лёгкий cleanup наиболее тяжёлых mock-heavy tests.

### Что не делаем

- не дробим всё подряд на микрофайлы;
- не переписываем working modules только из-за длины;
- не трогаем fast/green test-tree без причины.

### Ожидаемый результат

- читаемость лучше;
- onboarding легче;
- но объём изменений остаётся разумным для V1.

## 5. Таблица внедрения

| Волна | Что делаем | Зачем | Что затронем | Ожидаемый результат |
| --- | --- | --- | --- | --- |
| 0 | Вернуть зелёный `mypy` и quality-gates | Без этого дальнейшие изменения плохо контролируются | `analysis/model_comparison/quality.py`, contracts, tests | Снова зелёные `ruff + mypy + pytest` |
| 1 | Выравниваем current semantics | Убираем drift между runtime и docs | `README`, orchestrator docs, comparison docs | Один текущий смысл production scoring |
| 2 | Синхронизируем persist contract | Не терять explainability факторов | `priority_pipeline/*`, SQL, docs | Persist отражает реальный runtime |
| 3 | Фиксируем `UNKNOWN` как V1-contract | Убираем двусмысленность spec vs runtime | `ood docs`, input/runtime narrative | Понятная open-set semantics |
| 4 | Настраиваем оркестратор | Приводим operational ranking к осмысленному канону | `decision.py`, constants, QA runs | Воспроизводимый operational profile |
| 5 | Разводим production и comparison outputs | Убираем semantic путаницу shortlist/snapshot | notebooks, comparison docs, production exports | Shortlist идёт из production, snapshot остаётся research-layer |
| 6 | Пересобираем канонические артефакты | Старые артефакты уже не current canon | `experiments/*` | Новая versioned canonical wave |
| 7 | Синхронизируем notebooks/docs/slides | Защита должна опираться на current state | notebooks, docs, README, slides | Narrative совпадает с кодом и артефактами |
| 8 | Точечная hygiene-полировка | Улучшаем читаемость без лишнего рефакторинга | touched modules/tests/notebooks | Кодовая база чище, но без “Титанабоа-рефакторинга” |

## 6. Что лучше исправлять первым на практике

Если идти не по теории, а по реальному рабочему порядку, я бы делал так:

1. `mypy` / quality-gate
2. current semantics docs
3. persist contract
4. `UNKNOWN` contract
5. orchestrator tuning
6. reruns / canonical artifacts
7. notebook + docs + slides
8. только потом hygiene cleanup

Это даёт самый здоровый порядок:

- сначала возвращаем инженерный контроль;
- потом выравниваем смысл;
- потом настраиваем runtime;
- потом переиздаём evidence;
- и только потом полируем структуру.

## 7. Definition of Done для этой волны

Волну правок можно считать закрытой, когда:

1. `ruff`, `mypy`, `pytest` снова зелёные.
2. Current production formula и docs совпадают.
3. Persist contract отражает реальный scoring decomposition.
4. `UNKNOWN` semantics зафиксирована честно и без противоречий.
5. Оркестратор имеет осмысленный и воспроизводимый operational profile.
6. Comparison snapshot и production shortlist разведены по смыслу.
7. Выпущена новая каноническая versioned wave артефактов.
8. Notebooks и slides опираются на current canon, а не на исторический drift.

## 8. Итог одной фразой

Главная цель следующей волны не “сделать проект красивее”, а
**превратить хороший, но частично разъехавшийся V1 в согласованную,
воспроизводимую и защитопригодную систему**.
