# Карта покрытия аудита проекта

Дата фиксации: 19 марта 2026 года

## 1. Назначение документа

Этот документ является operational mapping-слоем для audit-wave.

Он нужен, чтобы:

- явно зафиксировать, какие зоны проекта считаются ключевыми;
- не потерять ни один важный слой при пошаговом аудите;
- не дублировать одни и те же проверки в разных блоках;
- быстро понимать, где искать canonical logic, артефакты и основные риски.

Если [project_audit_plan_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/project_audit_plan_ru.md)
задаёт последовательность работ, то этот документ задаёт
`что именно покрываем`.

## 2. Карта слоёв проекта

| Слой | Назначение | Канонические зоны | Основные выходы |
| --- | --- | --- | --- |
| `preprocessing` | подготовка train/reference data | [sql/preprocessing](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/preprocessing), [sql/adql](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/adql), [00_data_extraction_and_preprocessing.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/00_data_extraction_and_preprocessing.ipynb) | training relations, views, reproducible SQL/ADQL |
| `router` | физическая классификация звезды | [src/router_model](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model), [analysis/router_eda](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/router_eda), [02_router_readiness.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/02_router_readiness.ipynb) | predicted class, evolution stage, router scores |
| `OOD` | reject/open-set gating | [src/router_model/ood.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/ood.py), [docs/ood_unknown_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/ood_unknown_tz_ru.md) | `UNKNOWN`, reject policy, unknown share |
| `host-model` | host-vs-field scoring внутри MKGF dwarf | [src/host_model](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/host_model), [03_host_vs_field_contrastive.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/03_host_vs_field_contrastive.ipynb) | `host_posterior`, host-like profile |
| `decision layer` | итоговое ранжирование | [src/priority_pipeline/decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/decision.py), [src/priority_pipeline/pipeline.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/pipeline.py), [src/decision_calibration](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/decision_calibration) | `final_score`, `priority_tier`, `reason_code` |
| `comparison` | benchmark и baseline-сравнение | [analysis/model_comparison](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison), [04_model_comparison_summary.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/04_model_comparison_summary.ipynb) | benchmark metrics, snapshot, shortlist |
| `validation` | repeated split, generalization, risk audit | [analysis/model_validation](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_validation), [experiments/model_validation](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/model_validation) | generalization reports, gap diagnostics |
| `qa/docs/notebooks` | narrative, evidence, защитный слой | [docs](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs), [experiments/QA](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/QA), [notebooks/eda](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda) | README, findings, notebooks, QA logs |

## 3. Ownership map

### 3.1 Production runtime

| Зона | Главные файлы |
| --- | --- |
| router training/artifacts/scoring | [artifacts.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/artifacts.py), [fit.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/fit.py), [score.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/score.py), [ood.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model/ood.py) |
| host scoring | [fit.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/host_model/fit.py), [contrastive_score.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/host_model/contrastive_score.py), [legacy_score.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/host_model/legacy_score.py) |
| pipeline branching | [branching.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/branching.py), [pipeline.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/pipeline.py) |
| decision factors | [decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/decision.py) |
| persist/runtime IO | [input_data.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/input_data.py), [persist.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/persist.py), [src/infra/db.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/infra/db.py) |

### 3.2 Research and evidence

| Зона | Главные файлы |
| --- | --- |
| benchmark dataset and split | [data.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/data.py) |
| baseline wrappers | [contrastive.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/contrastive.py), [legacy_gaussian.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/legacy_gaussian.py), [random_forest.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/random_forest.py), [mlp_baseline.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/mlp_baseline.py) |
| benchmark metrics | [metrics.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/metrics.py), [quality.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/quality.py) |
| artifact/report generation | [reporting.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/reporting.py), [snapshot.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/snapshot.py), [generalization_audit.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_comparison/generalization_audit.py) |
| heavy validation | [analysis/model_validation](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/model_validation) |

## 4. Risk map

| Поверхность | Основной риск | Почему важно |
| --- | --- | --- |
| preprocessing lineage | train leakage, неверные relations, historical drift | если здесь ошибка, вся математика ниже теряет смысл |
| router | неверная физическая классификация | ломает вход в host-ветку и shortlist |
| OOD | слишком агрессивный или слишком слабый reject | искажает candidate pool и защитную позицию |
| host-model | переобучение, class bias, плохая separation `host vs field` | влияет на ranking core |
| decision layer | неоправданные modifiers, перекос итогового score | можно “испортить” хороший model score неправильной логикой |
| comparison-layer | нечестное сравнение, mismatch metrics/thresholds | можно сделать ложный вывод о лучшей модели |
| validation-layer | ложное чувство generalization stability | создаёт неправильную уверенность в модели |
| notebooks | ручная логика, расхождение с артефактами | опасно для защиты и репозитория |
| docs | narrative drift | код и выводы начинают говорить разное |
| tests | inverted pyramid, устаревшие контракты, хрупкость | замедляют работу и перестают защищать систему |
| naming | naming drift, исторически накопленные имена, confusing artifacts | новому человеку трудно понять ownership, current state и назначение файлов |

## 5. Audit coverage map

| Audit блок | Что покрывает | Основные evidence |
| --- | --- | --- |
| `critical paths / risk mapping` | inventory, surfaces, канонические зоны | README, state policy, experiment READMEs |
| `логика и архитектура` | routing chain, responsibilities, module boundaries | `src/*`, `analysis/*`, orchestration docs |
| `математика и физика` | priors, factors, class ranking, host-like interpretation | decision-layer code, model comparison findings, notebooks |
| `ML / reproducibility / runs` | benchmark, quality, snapshot, validation | `experiments/model_comparison/*`, `experiments/model_validation/*`, live runs |
| `код` | readability, duplication, typing, complexity | Python modules, ruff/mypy, file-by-file review |
| `тесты` | correctness, актуальность, size/scope, coverage balance | `tests/*`, pytest markers, suite structure |
| `notebooks` | correctness of narrative and calculations | `notebooks/eda/*`, linked CSV artifacts |
| `docs` | consistency and readiness for defense | `docs/*`, README, findings/protocol/state policy |

## 6. Critical-path checklist

Ниже перечислены зоны, которые обязательно должны получить findings или
явную отметку `critical issues not found`.

### 6.1 Data and preprocessing

- relations и views для train/reference
- SQL/ADQL reproducibility
- соответствие notebooks и SQL

### 6.2 Runtime chain

- router -> OOD -> host-model -> decision layer
- `UNKNOWN` ветка
- persist results

### 6.3 Benchmark and validation

- benchmark split
- search protocol
- threshold-based quality
- generalization audit
- snapshot consistency

### 6.4 Presentation and defense

- summary notebook
- shortlist
- финальные physical conclusions
- docs consistency

## 7. Практическое правило использования mapping

При каждом новом audit-блоке:

1. сначала отмечаем, какой surface мы проверяем;
2. сверяемся с ownership map;
3. фиксируем evidence;
4. записываем finding в реестр;
5. не переходим к плану правок до завершения audit-wave.
