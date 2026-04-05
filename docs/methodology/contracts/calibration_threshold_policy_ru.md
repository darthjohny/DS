# Calibration And Threshold Policy

## Цель

Этот документ фиксирует second-wave policy для:

- probability calibration;
- threshold tuning;
- handoff между `OOD`, `coarse`, `refinement` и `unknown/review`.

Документ нужен, чтобы:

- не прятать confidence logic внутри inference-кода;
- не смешивать calibration с base model training;
- не подбирать пороги на test split;
- не превращать decision layer в набор ad hoc `if` по месту.

## Инженерный Инвариант

Для кода, который будет реализовывать этот policy later, действует тот же стандарт:

- `1 файл = 1 ответственность`
- calibration modules отдельно от base model runners
- threshold tuning отдельно от final decision mapping
- без giant inference-policy file
- `PEP 8`
- явная типизация
- простая логика раньше сложной
- без лишних зависимостей
- после каждого небольшого куска:
  - micro-QA
  - `ruff`
  - точечный `mypy/pyright`
  - целевые tests
- после завершения микро-ТЗ:
  - scoped big-QA только по написанному слою

## Official Опора

### Probability Calibration

Official scikit-learn docs фиксируют:

- calibration переводит `decision_function` или `predict_proba` output в calibrated
  probability;
- calibrator должен учиться на данных, независимых от train subset базового
  classifier;
- `CalibratedClassifierCV` делает это через internal cross-validation;
- для multiclass calibrator fit-ится отдельно по классам;
- при `ensemble=True` calibration идет по folds и averaged на predict-time;
- `isotonic` не рекомендуется при слишком малом числе calibration samples;
- `sigmoid` safer как first default на ограниченных данных.

Официальные источники:

- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV)

### Threshold Tuning

Official scikit-learn docs фиксируют:

- threshold tuning — это отдельный post-hoc step поверх classifier score/probability;
- `TunedThresholdClassifierCV` пост-оптимизирует decision threshold через CV;
- default metric у него — `balanced_accuracy`;
- `cv="prefit"` нельзя использовать на том же датасете, на котором estimator
  обучался, иначе будет overfitting;
- default internal CV — `5-fold stratified`.

Официальные источники:

- [Tuning the decision threshold for class prediction](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [TunedThresholdClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html)

### Metrics

Official scikit-learn docs фиксируют:

- `balanced_accuracy` подходит для binary и multiclass задач при class imbalance;
- `classification_report` удобен как structured report по precision / recall / F1
  per class и macro/weighted averages.

Официальные источники:

- [balanced_accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)
- [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)

## Общие Правила Проекта

### Rule 1. Calibration Не Живет Внутри Base Model

Base classifier и calibrator — разные сущности.

Project policy:

- сначала train base estimator;
- потом отдельно calibration stage;
- потом отдельно decision threshold stage;
- потом отдельно final decision mapping.

### Rule 2. Test Split Не Используется Для Calibration/Threshold Tuning

Calibration и threshold tuning разрешены только на:

- внутреннем CV;
- отдельном validation split;
- специальном holdout, который не является final test.

Запрещено:

- fit classifier на set A и tune threshold на том же set A;
- fit classifier, calibrator и final threshold на test split.

### Rule 3. Thresholds Versioned

Порог — это не магическое число в коде.

Для каждого tuned/manual threshold должны быть явные поля:

- `threshold_name`
- `threshold_value`
- `threshold_metric`
- `threshold_fit_scope`
- `threshold_policy_version`

## Stage-Specific Policy

## Stage A. `ID/OOD`

### Почему Этот Stage Тюним Первым

`ID/OOD` у нас binary и уже имеет сильный baseline:

- test balanced_accuracy `0.926215`
- test macro_f1 `0.944521`

Значит именно этот stage first candidate для official threshold tuning.

### Base Policy

First second-wave design:

- base estimator family пока не меняем;
- используем existing `HistGradientBoostingClassifier` baseline;
- тюнинг threshold делаем отдельно от fit.

### Calibration Policy

Для `ID/OOD` допускается:

- calibrate probabilities через `CalibratedClassifierCV`
- default calibration method: `sigmoid`

Причина:

- `sigmoid` safer default;
- не требует большого calibration mass как `isotonic`;
- лучше соответствует conservative first second-wave.

`isotonic`:

- оставляем как later experiment;
- не делаем default без отдельного evidence-run.

### Threshold Policy

Для `ID/OOD` threshold tuning проектируем через:

- `TunedThresholdClassifierCV`

Default scoring:

- `balanced_accuracy`

Пояснение:

- это official default scorer для `TunedThresholdClassifierCV`;
- он согласуется с нашей задачей, где и false `OOD`, и missed `OOD` важны;
- это лучше, чем hard-coded `0.5`.

### Output Contract

`ID/OOD` stage должен уметь отдавать:

- `ood_probability`
- `id_probability`
- `ood_threshold`
- `ood_threshold_policy_version`
- `ood_decision`
  - `in_domain`
  - `candidate_ood`
  - `ood`

### First Decision Policy

Этот документ не фиксирует конкретные numeric thresholds.

Фиксируется только процесс:

- tuned threshold определяем на validation/CV;
- `ood` и `candidate_ood` разделяем в отдельном manual policy layer;
- `candidate_ood` не схлопываем с clean `in_domain`.

## Stage B. `Coarse`

### Base Policy

`coarse` baseline уже достаточно силен:

- test balanced_accuracy `0.992379`

Поэтому:

- coarse stage не становится primary candidate для threshold tuning first;
- сначала его используем как stable classifier с probability output.

### Calibration Policy

Second-wave default:

- calibration для `coarse` не является обязательным первым шагом;
- сначала сохраняем raw `predict_proba` и confidence margin;
- отдельный calibration experiment допустим позже.

Причина:

- coarse metrics уже очень высокие;
- biggest next-wave gain ожидается не от re-calibration coarse,
  а от better handoff logic между stages.

### Confidence Policy

Для `coarse` фиксируем decision contract, а не numeric threshold:

- `coarse_probability_max`
- `coarse_probability_margin`
- `coarse_policy_version`

Project policy:

- numeric threshold для `coarse -> refinement` подбирается на validation;
- threshold не тюним на test;
- threshold не живет внутри base model code.

## Stage C. `Refinement`

### Base Policy

`refinement` во второй волне идет через family-based decomposition:

- `A`
- `B`
- `F`
- `G`
- `K`
- `M`

`O` остается coarse-only.

### Calibration Policy

Calibration для refinement families допускается, но не должна быть mandatory
в первом кодовом шаге.

Причина:

- сначала нужно materialize-ить family tasks и проверить их baseline;
- только потом решать, где calibration реально улучшает reliability.

Default future policy:

- per-family calibration module, если и когда включаем;
- default method: `sigmoid`;
- `isotonic` только как targeted experiment на достаточно больших family slices.

### Output Contract

Каждая family model должна уметь отдавать:

- `refinement_probability_max`
- `refinement_probability_margin`
- `refinement_threshold`
- `refinement_threshold_policy_version`
- `refinement_decision`
  - `accepted`
  - `rejected_to_unknown`
  - `not_attempted`

### Handoff Policy

Refinement запускается только если одновременно:

- `OOD` stage не перевел объект в `ood`
- `coarse` class входит в refinement-enabled list
- coarse confidence проходит agreed threshold
- family-specific subclass policy допускает target

Если это не выполнено:

- refinement не forced-run-ится;
- объект остается на coarse-level или уходит в `unknown/review`.

## Unknown / Review Policy

`unknown/review` — это отдельный outcome, не ошибка метрики.

Object должен попадать в `unknown/review`, если:

- `quality_state <> 'pass'`
- `ood_state = 'candidate_ood'`
- `ood_state = 'ood'`, но policy требует ручного review bucket
- coarse confidence ниже agreed threshold
- coarse class coarse-only для refinement
- refinement family отказалась от confident decision

## Что Не Делаем

- не вшиваем numeric thresholds в training runner;
- не смешиваем calibration, threshold tuning и final decision mapping в одном модуле;
- не используем `cv="prefit"` на том же наборе, где fit-ился estimator;
- не вводим multiclass threshold tuning ad hoc без validation contract;
- не делаем `isotonic` default без отдельного justification.

## MTZ-M51 Deliverable

`MTZ-M51` считается закрытым, когда:

- calibration policy зафиксирована отдельно от model-code;
- threshold policy зафиксирована отдельно от decision mapping;
- official scikit-learn опора и project policy разведены явно;
- следующий шаг может проектировать final decision layer без архитектурной
  импровизации.

## Current Code Status (`2026-03-28`)

Policy уже переведен в отдельные code-side модули:

- [calibration.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/posthoc/calibration.py)
- [threshold_tuning.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/posthoc/threshold_tuning.py)
- [id_ood_gate.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/posthoc/id_ood_gate.py)
- [run_id_ood_posthoc_gate.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/training/run_id_ood_posthoc_gate.py)
- [id_ood_threshold_artifacts.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/reporting/id_ood_threshold_artifacts.py)

Инженерно важный low-level фикс:

- custom model wrappers приведены к корректной sklearn classifier semantics через
  `ClassifierMixin` слева от `BaseEstimator`;
- это потребовалось, чтобы official `CalibratedClassifierCV` и
  `TunedThresholdClassifierCV` работали без ad hoc adapters.

Дополнительный data-slice фикс:

- `ID/OOD` loader больше не режет `limit` по `domain_target ASC`;
- ограниченные выборки теперь упорядочиваются по `random_index`, чтобы CV не
  получал одноклассовый slice.

Artifact contract теперь тоже закрыт:

- tuned threshold policy сохраняется отдельно от model artifact;
- decision CLI читает не только saved estimator, но и saved threshold artifact;
- это исключает скрытый manual threshold drift между notebook, CLI и end-to-end run.

Live smoke-run для `gaia_id_ood_classification` (`HistGradientBoosting`,
`limit=5000`):

- train accuracy: `0.985677`
- train balanced_accuracy: `0.990211`
- train macro_f1: `0.982064`
- test accuracy: `0.969272`
- test balanced_accuracy: `0.968762`
- test macro_f1: `0.961488`
- tuned threshold: `0.041898`

Scoped QA по post-hoc slice:

- `ruff` ok
- `mypy` ok
- `pyright` ok
- targeted `pytest` ok
