# Post-Run Stabilization TZ

## Цель

Этот документ фиксирует следующий пакет работ после закрытия основного
implementation-плана:

- довести текущий `end-to-end` контур до устойчивого и объяснимого состояния;
- разложить по полкам реальные баги, тормоза, сомнительные места и научные
  ограничения;
- опираться на official documentation языка, библиотек, Gaia и NASA, а не на
  ad hoc-интуицию;
- улучшать только доказанные проблемные места, не ломая уже собранные
  contracts.

Текущая фаза уже не про сбор новой архитектуры, а про:

- `debugging`
- `profiling`
- `traceability`
- `scientific interpretation`
- `VKR-ready reporting`

## Текущее Базовое Состояние

К моменту старта stabilization-пакета уже есть:

- DB layers для `coarse / refinement / OOD / quality_gate / unknown_review`;
- artifact-backed `decide`;
- clean host enrichment с `radius_flame`;
- clean host retrain и `priority` в mainline;
- notebooks:
  - `model_pipeline_review.ipynb`
  - `final_decision_review.ipynb`
  - `quality_gate_calibration.ipynb`;
- успешный live run:
  - `artifacts/decisions/hierarchical_final_decision_2026_03_29_073129_772461`

Это означает, что дальнейшая работа идет поверх уже работающего контура, а не
вместо него.

Базовые run-артефакты и живой issue register вынесены отдельно:

- [baseline_run_registry_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/baseline_run_registry_ru.md)
- [stabilization_issue_ledger_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/stabilization_issue_ledger_ru.md)
- [star_level_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/star_level_review_round1_ru.md)
- [star_level_review_round2_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/star_level_review_round2_ru.md)
- [priority_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/priority_review_round1_ru.md)
- [coarse_o_tail_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_tail_review_round1_ru.md)
- [coarse_o_hot_subset_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_hot_subset_review_round1_ru.md)
- [coarse_ob_boundary_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_ob_boundary_review_round1_ru.md)
- [coarse_o_train_support_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_train_support_review_round1_ru.md)
- [coarse_ob_feature_separability_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_ob_feature_separability_review_round1_ru.md)
- [coarse_ob_provenance_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/coarse_ob_provenance_review_round1_ru.md)
- [coarse_ob_provenance_review_round2_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/coarse_ob_provenance_review_round2_ru.md)
- [coarse_ob_boundary_policy_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/coarse_ob_boundary_policy_ru.md)
- [secure_o_tail_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/secure_o_tail_review_round1_ru.md)
- [pre_battle_tuning_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/pre_battle_tuning_tz_ru.md)

Текущее решение по hot `O/B` boundary уже operational:

- ambiguous `OB...` больше не учатся как clean `O`;
- `lab.gaia_ob_boundary_review` используется как explicit review-pool;
- без external spectroscopy forced automatic `O/B` split для boundary-пула
  сейчас не делаем.

## Инженерный Инвариант

Для stabilization-пакета сохраняются те же правила:

- `1 файл = 1 ответственность`;
- без монолитов и “универсальных” helper-файлов;
- `PEP 8`;
- явная типизация;
- простые решения раньше сложных;
- без скрытой магии совместимости;
- каждый bugfix должен закрывать корень проблемы, а не только symptom;
- сначала narrow repro, потом fix, потом regression test;
- после каждого небольшого куска:
  - micro-QA
  - `ruff`
  - точечный `mypy/pyright`
  - targeted `pytest`;
- после завершения модуля:
  - scoped big-QA только по затронутому слою.

## Official Опора

### Python / Typing / Profiling

- [typing — Python 3.14 docs](https://docs.python.org/3/library/typing.html)
- [collections.abc — Python docs](https://docs.python.org/3/library/collections.abc.html)
- [The Python Profilers](https://docs.python.org/3/library/profile.html)

### pandas / Jupyter / SQLAlchemy

- [pandas: Working with missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [pandas: Enhancing performance](https://pandas.pydata.org/docs/user_guide/enhancingperf.html)
- [Jupyter documentation](https://docs.jupyter.org/en/latest/)
- [Jupyter execution / nbclient](https://docs.jupyter.org/en/latest/projects/execution.html)
- [SQLAlchemy Engine Configuration](https://docs.sqlalchemy.org/20/core/engines.html)

### scikit-learn / joblib

- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [Decision threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
- [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
- [joblib Parallel](https://joblib.readthedocs.io/en/stable/generated/joblib.Parallel.html)

### Gaia / NASA

- [Gaia DR3 documentation index](https://gea.esac.esa.int/archive/documentation/GDR3/)
- [Gaia DR3 gaia_source semantics](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html)
- [Gaia DR3 astrophysical_parameters semantics](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_astrophysical_parameter_tables/ssec_dm_astrophysical_parameters.html)
- [Gaia DR3 GSP-Phot](https://gea.esac.esa.int/archive/documentation/GDR3/Data_analysis/chap_cu8par/sec_cu8par_apsis/ssec_cu8par_apsis_gspphot.html)
- [Gaia DR3 Apsis overview](https://gea.esac.esa.int/archive/documentation/GDR3/Data_analysis/chap_cu8par/sec_cu8par_intro/ssec_cu8par_intro_apsis.html)
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [NASA TAP guide](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html)
- [PSCompPars semantics](https://exoplanetarchive.ipac.caltech.edu/docs/pscp_about.html)

## Полезные Похожие Работы

- [Quality flags for GSP-Phot Gaia DR3 astrophysical parameters with machine learning](https://academic.oup.com/mnras/article-abstract/527/3/7382/7442087)
- [A classifier for spurious astrometric solutions in Gaia EDR3](https://arxiv.org/abs/2101.11641)
- [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500)
- [SelectiveNet](https://arxiv.org/abs/1901.09192)
- [The Gaia-Kepler-TESS-Host Stellar Properties Catalog](https://arxiv.org/abs/2301.11338)

## Стартовый Реестр Известных Проблем

### KI-01. Artifact/Relation Compatibility

Уже пойманный класс проблем:

- model artifacts требуют feature names, которых нет в DB relation один-в-один;
- пример: `radius_feature` в `decide` при live relation с `radius_flame`.

Текущее решение:

- compatibility alias строится в loader/frame слое;
- relation остается чистым и не хранит artifact-specific дубли.

### KI-02. Overfiltering На Input Loader

Уже пойманный класс проблем:

- `final decision` loader фильтровал relation по union-признакам всех стадий;
- это обнуляло dataset до inference, хотя stage-level pipelines имеют свои
  imputer/preprocessor contracts.

Текущее решение:

- global non-null filtering для feature union убран;
- stage models обрабатывают missing values в своих пайплайнах.

### KI-03. Notebook Drift

Уже пойманный класс проблем:

- notebook смотрит на устаревший run dir;
- notebook показывает служебный путь вместо итогового review;
- roles notebook-ов смешиваются.

Текущее решение:

- `06` оставлен pipeline/benchmark notebook;
- `07` оставлен final decision / star-level notebook;
- run dirs привязаны к живым artifacts.

### KI-04. Высокая Доля `unknown`

Это пока не bug, а открытый scientific/operational вопрос.

Уже известно:

- главный драйвер `unknown/reject` — не `RUWE`, а `missing_core_features`;
- часть `unknown` может быть корректным selective outcome.

### KI-05. Host/Priority Interpretation Risk

Priority уже подключен, но теперь нужно отдельно проверить:

- не переоцениваем ли `host_similarity_score`;
- не получаем ли слишком много `high` без достаточного explainability;
- не расходится ли ranking с astrophysical intuition и withhold-case review.

## Новый Пакет Микро-ТЗ

### MTZ-S01. Зафиксировать Baseline Run Registry

- Цель: заморозить текущие живые artifact dirs как baseline для stabilization.
- Что фиксируем:
  - benchmark runs
  - model runs
  - threshold runs
  - final decision runs
  - notebooks, которые на них смотрят
- Результат:
  - doc/table с `latest stable run dirs`
- Критерий готовности:
  - дальше не спорим, какой run считать baseline.

### MTZ-S02. Собрать Issue Ledger По Реальным Прогонам

- Цель: вести bugs, тормоза, странности и сомнительные места в одном месте.
- Что фиксируем:
  - repro
  - expected behaviour
  - observed behaviour
  - probable root cause
  - layer owner
  - статус
- Формат:
  - короткий issue register в docs
  - без GitHub-style process overhead
- Критерий готовности:
  - каждый баг имеет owner и гипотезу причины.

### MTZ-S03. Провести Star-Level Review Текущего `decide` Run

- Цель: перейти от метрик к объектам.
- Что смотрим:
  - `final_domain_state`
  - `final_coarse_class`
  - `final_refinement_label`
  - `priority_score`
  - `priority_label`
  - `priority_reason`
  - `mh_gspphot`
  - `parallax`
  - `parallax_over_error`
  - `ruwe`
  - `radius_flame`
  - `quality_reason`
  - `review_bucket`
- Результат:
  - список типовых хороших кейсов
  - список сомнительных кейсов
  - список явно плохих кейсов
- Критерий готовности:
  - можем предметно говорить “что модель решила и почему”.

### MTZ-S04. Провести Traceability Audit Для Final Outputs

- Цель: убедиться, что все важные финальные поля семантически согласованы.
- Проверяем:
  - `coarse -> refinement` handoff
  - `unknown / ood / id`
  - `priority` presence only where allowed
  - canonical radius semantics
  - отсутствие скрытых legacy fallbacks
- Official опора:
  - Python typing docs
  - pandas missing-data semantics
  - Gaia/NASA contracts
- Критерий готовности:
  - final outputs объяснимы и не содержат скрытых semantic jumps.

Текущее состояние:

- первый traceability defect уже найден и закрыт:
  - explainability fields теперь сохраняются в `decision_input` artifacts;
- следующий открытый подтрек внутри `S04`:
  - semantic saturation в `priority` уже зафиксирован отдельным review;
  - дальше нужно проверить host-score calibration;
  - проверить отсутствие скрытых legacy jumps в star-level outputs

### MTZ-S05. Провести Performance And Bottleneck Profiling

- Цель: не гадать про тормоза, а измерить их.
- Что профилируем:
  - `decide`
  - notebook execution
  - DB read-heavy loaders
  - artifact loading
  - scoring / ranking hotspots
- Official опора:
  - Python `cProfile` / `pstats`
  - pandas performance guide
  - joblib docs
- Критерий готовности:
  - список measured bottlenecks с приоритетом.

### MTZ-S06. Провести Precision Bugfix Cycle

- Цель: исправлять только доказанные дефекты.
- Правило:
  - сначала repro
  - потом narrow fix
  - потом regression test
  - потом scoped QA
- Не делаем:
  - speculative refactors
  - массовое переписывание working modules
- Критерий готовности:
  - каждый bugfix закрыт тестом и не ломает соседние слои.

### MTZ-S07. Подготовить Scientific Findings Pack

- Цель: перевести результат из engineering-state в VKR-ready summary.
- Что собираем:
  - baseline metrics
  - распределения `id / unknown / ood`
  - примеры `high / medium / low` priority
  - ограничения
  - где система сознательно осторожна
- Результат:
  - табличный и narrative summary для текста работы.

### MTZ-S08. Определить Следующую Iteration Boundary

- Цель: понять, что реально стоит делать в новой итерации.
- Возможные направления:
  - quality-gate fine tuning
  - host-priority recalibration
  - refinement family uplift
  - performance optimization
  - reporting/VKR polish
- Критерий готовности:
  - есть короткий и честный план следующего цикла.

## Порядок Работы

1. `MTZ-S01`
2. `MTZ-S02`
3. `MTZ-S03`
4. `MTZ-S04`
5. `MTZ-S05`
6. `MTZ-S06`
7. `MTZ-S07`
8. `MTZ-S08`

## Ближайший Практический Следующий Шаг

Сейчас лучший следующий шаг:

- не трогать `priority` и `quality_gate`, пока не закрыт кейс `O/B`;
- не ретрейнить coarse-модель вслепую;
- после подтвержденного domain shift на `O/B` перейти к provenance/source-alignment review
  downstream true `O`;
- только после этого решать, нужен ли narrow retrain, class weighting или relabel/source-alignment.

Перед этим уже закрыт alignment sanity-check:

- [coarse_ob_alignment_audit_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/coarse_ob_alignment_audit_round1_ru.md)

Он подтверждает, что gross train/inference mismatch не найден, а радиусный
`radius_feature` nuance не объясняет `O -> B` collapse сам по себе.
