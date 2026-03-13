# ТЗ по OOD/Unknown для production router

Дата: 12 марта 2026 года

Статус документа:
- historical planning document;
- фиксирует исходное ТЗ перед внедрением `UNKNOWN` как first-class
  router outcome;
- не должен читаться как главный current-state документ.

Где смотреть текущее состояние:
- `README.md` — пользовательское описание текущего runtime-контракта;
- `src/router_model/ood.py` и `src/router_model/score.py` — каноническая
  реализация reject/unknown logic;
- `tests/test_gaussian_router.py`,
  `tests/test_priority_pipeline.py`,
  `tests/test_priority_pipeline_db_integration.py` — актуальные проверки
  runtime-контракта.

## 1. Цель документа

Зафиксировать канонический open-set контракт для `OOD/Unknown` до начала
кодовых изменений в `router_model`, `priority_pipeline`, persist-слое и
ноутбуках.

Документ отвечает на три вопроса:

- когда router имеет право не присваивать известный физический класс;
- какие значения при этом должны появиться в runtime и persist;
- какие артефакты проекта необходимо изменить после включения
  `UNKNOWN` как first-class значения.

## 2. Текущее состояние

Сейчас в проекте уже есть технический stub для неполной строки:

- в [src/router_model/score.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/score.py)
  `_empty_router_score()` возвращает
  `predicted_spec_class='UNKNOWN'`,
  `predicted_evolution_stage='unknown'`,
  `router_label='UNKNOWN'`;
- этот сценарий срабатывает только для missing features или пустого artifact;
- production pipeline и persist-слой пока не считают это официальным
  open-set режимом;
- текущие DB constraints не разрешают `UNKNOWN` и `unknown` в результатах.

Итог: `UNKNOWN` уже существует как техническая заглушка, но ещё не
зафиксирован как официальный production-contract.

## 3. Цель первой волны OOD/Unknown

Первая волна должна быть минимальной и прагматичной:

- добавить reject-option поверх текущего posterior-aware router;
- не внедрять отдельный OOD-модуль машинного обучения;
- не добавлять новые DB-колонки без прямой необходимости;
- использовать уже существующие diagnostics:
  `router_log_posterior`, `posterior_margin`, `router_similarity`,
  `second_best_label`, missing features.

В этой итерации `UNKNOWN` означает:

- объект с неполным набором ключевых признаков;
- объект с недостаточной уверенностью router;
- объект, который лучше не пускать в closed-set ветку, чем насильно
  относить к одному из известных классов.

## 4. Канонический runtime-контракт

### 4.1 Значения полей router result

При reject в `UNKNOWN` router должен возвращать такие значения:

| Поле | Значение |
| --- | --- |
| `predicted_spec_class` | `UNKNOWN` |
| `predicted_evolution_stage` | `unknown` |
| `router_label` | `UNKNOWN` |
| `router_model_version` | текущая версия router artifact |

Дополнительные поля diagnostics ведут себя так:

- если reject произошёл из-за missing features, допускаются текущие
  fallback-значения из `_empty_router_score()`:
  `d_mahal_router=NaN`, `router_similarity=0.0`,
  `router_log_likelihood=NaN`, `router_log_posterior=NaN`,
  `second_best_label='UNKNOWN'`, `margin=NaN`,
  `posterior_margin=NaN`;
- если reject произошёл после успешного raw scoring, численные
  diagnostics должны сохраняться и не затираться, чтобы их можно было
  использовать в QA и EDA.

### 4.2 Значения полей priority result

При попадании в unknown-ветку pipeline должен писать:

| Поле | Значение |
| --- | --- |
| `predicted_spec_class` | `UNKNOWN` |
| `predicted_evolution_stage` | `unknown` |
| `router_label` | `UNKNOWN` |
| `final_score` | `0.0` |
| `priority_tier` | `LOW` |
| `reason_code` | `ROUTER_UNKNOWN` |
| `host_model_version` | `NULL` |

Host-specific поля при этом не заполняются:

- `gauss_label=NULL`
- `host_log_likelihood=NULL`
- `field_log_likelihood=NULL`
- `host_log_lr=NULL`
- `host_posterior=NULL`
- `d_mahal=NULL`
- `similarity=NULL`

Общие quality-факторы можно сохранять так же, как для обычной
low-priority ветки, чтобы unknown-объекты не выпадали из общей
диагностической картины.

## 5. Правило reject-option

### 5.1 Порядок принятия решения

Reject должен применяться после raw scoring и иметь явный порядок:

1. `Structural reject`
   Любой из ключевых признаков `teff_gspphot`, `logg_gspphot`,
   `radius_gspphot` отсутствует.
2. `Artifact reject`
   В artifact нет классов, либо победившая метка не нормализуется к
   поддерживаемому router label.
3. `Confidence reject`
   Raw scoring завершился успешно, но объект не проходит пороги
   уверенности.

### 5.2 Сигналы уверенности

В первой волне для confidence reject используются только уже существующие
поля:

- `router_log_posterior`
- `posterior_margin`
- `router_similarity`

### 5.3 Каноническая логика confidence reject

Первая версия open-set правила фиксируется так:

- reject, если `router_log_posterior < min_router_log_posterior`;
- reject, если одновременно
  `posterior_margin < min_posterior_margin`
  и `router_similarity < min_router_similarity`;
- иначе объект считается accepted и остаётся в своём best-known классе.

Эта схема нужна, чтобы:

- отсеивать совсем слабые объекты по posterior;
- не выбрасывать хороший объект только из-за одного неидеального сигнала;
- сохранить rule-form простой и интерпретируемой.

### 5.4 Как хранятся пороги

Численные значения порогов не должны быть захардкожены внутри
`score.py`. Они должны жить в router artifact metadata.

Ожидаемые ключи в `meta`:

- `allow_unknown: bool`
- `ood_policy_version: str`
- `min_router_log_posterior: float`
- `min_posterior_margin: float`
- `min_router_similarity: float`

В этом документе фиксируется форма порогов, а не их окончательные
numeric values. Конкретные числа задаются и проверяются на этапе
обновления artifact contract и QA.

## 6. Контракт ветвления в pipeline

После внедрения `UNKNOWN` production pipeline должен разделяться на три
явные ветки:

1. `host`
   Только `M/K/G/F` и только `dwarf`.
2. `low_known`
   Известные, но нецелевые для host-scoring объекты:
   `A/B/O`, `evolved`, другие filtered known cases.
3. `unknown`
   Объекты с `predicted_spec_class='UNKNOWN'`.

Канонические правила:

- unknown-объект никогда не должен попадать в host-scoring;
- unknown-ветка не должна жить как частный случай внутри
  `build_low_priority_stub()` без явного reason-code;
- branching лучше вынести в отдельный модуль
  [src/priority_pipeline/branching.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/branching.py),
  чтобы [src/priority_pipeline/decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/decision.py)
  остался модулем про факторы и scoring.

## 7. Persist и DB-schema

### 7.1 Что обязательно меняется

Нужна отдельная SQL migration, потому что текущие result-таблицы не
принимают новые значения.

Минимум, который должен поддержать persist:

- `predicted_spec_class='UNKNOWN'`
- `predicted_evolution_stage='unknown'`
- `router_label='UNKNOWN'`
- `reason_code='ROUTER_UNKNOWN'`

Текущая migration этого шага:

- [2026-03-13_gaia_results_unknown_constraints.sql](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/2026-03-13_gaia_results_unknown_constraints.sql)
- [2026-03-13_gaia_results_unknown_constraints.rollback.sql](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/2026-03-13_gaia_results_unknown_constraints.rollback.sql)

### 7.2 Что сохраняем без изменений

В первой волне не добавляем новые DB-колонки, если хватает текущих
полей для диагностики.

Это означает:

- используем существующие `router_similarity`,
  `router_log_posterior`, `posterior_margin`;
- не вводим отдельные `is_unknown` или `reject_reason` до тех пор,
  пока это реально не понадобится.

## 8. Отчёты, notebook и QA

После внедрения `UNKNOWN` нужно синхронизировать все артефакты, где
сейчас по умолчанию предполагается closed-set мир.

Обязательные изменения:

- [README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/README.md)
  должен явно описывать open-set поведение router;
- [experiments/QA/qa_mvp_report_2026-03-11.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/QA/qa_mvp_report_2026-03-11.md)
  и будущие QA markdown должны содержать `unknown_count` и
  `unknown_share`;
- [notebooks/eda/00_data_extraction_and_preprocessing.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/00_data_extraction_and_preprocessing.ipynb)
  и будущие router notebooks должны не смешивать unknown с известными
  классами в distribution-плотах;
- `top-k` отчёты не должны включать `UNKNOWN` рядом с кандидатами из
  host-ветки.

## 9. Целевая файловая карта

| Файл | Что меняется |
| --- | --- |
| `docs/ood_unknown_tz_ru.md` | Канонический spec по `UNKNOWN` |
| `src/router_model/ood.py` | Reject logic и helper-функции |
| `src/router_model/artifacts.py` | Metadata contract для OOD thresholds |
| `src/router_model/labels.py` | Константы `UNKNOWN`/`unknown` и helper-функции |
| `src/router_model/score.py` | Явное разделение raw scoring и reject decision |
| `src/priority_pipeline/branching.py` | Новое трёхветочное разбиение |
| `src/priority_pipeline/constants.py` | `ROUTER_UNKNOWN` и related constants |
| `src/priority_pipeline/pipeline.py` | Unknown branch в runtime |
| `src/priority_pipeline/persist.py` | Совместимость persist с новыми значениями |
| `src/decision_calibration/runtime.py` | Учёт unknown в preview |
| `src/decision_calibration/reporting.py` | Учёт unknown в отчётах |
| `analysis/router_eda/open_set.py` | Open-set diagnostics |
| `sql/*_router_unknown_constraints.sql` | Migration для DB constraints |

## 10. Минимальный тестовый контур

### 10.1 Unit tests

- missing features -> `UNKNOWN`;
- low confidence -> `UNKNOWN`;
- уверенный in-domain sample -> известный class/stage;
- split logic не пускает unknown в host-ветку.

### 10.2 DB / integration tests

- persist принимает `UNKNOWN` после migration;
- `priority_pipeline` корректно сохраняет unknown-result в временную
  схему;
- unknown-объект не получает host-specific поля.

### 10.3 Reporting smoke tests

- markdown summary умеет считать `unknown_count`;
- comparative/QA отчёты не падают на `UNKNOWN`;
- ordering/top-k не поднимает unknown выше обычных scored-кандидатов.

## 11. Что не входит в первую волну

На этом этапе не делаем:

- отдельный ML OOD-detector;
- `GMM` как механизм open-set rejection;
- новый большой subsystems вокруг unknown;
- смешивание open-set benchmark с baseline-сравнениями в один отчёт;
- полную переработку всех исторических логов задним числом.

## 12. Критерий готовности микро-ТЗ 1

`Микро-ТЗ 1` считается закрытым, если:

- описан официальный `UNKNOWN`-контракт;
- зафиксирован порядок reject decision;
- зафиксированы runtime и persist значения;
- перечислены обязательные изменения в pipeline, БД, QA и notebook;
- дальнейшая кодовая реализация может идти уже по этому документу.
