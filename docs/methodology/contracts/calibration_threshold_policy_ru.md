# Политика калибровки и порогов

## Цель

Этот документ фиксирует политику второй рабочей версии для:

- калибровки вероятностей;
- настройки порогов;
- handoff между `OOD`, `coarse`, `refinement` и `unknown/review`.

Документ нужен, чтобы:

- не прятать логику уверенности внутри кода обработки;
- не смешивать калибровку с обучением базовой модели;
- не подбирать пороги на test split;
- не превращать слой решения в набор разрозненных `if`.

## Общий принцип

Документ задает правила для кода, но не описывает нашу внутреннюю организацию
работы по файлам и шагам.

Здесь фиксируются только:

- роль калибровки;
- роль настройки порогов;
- место этих шагов в общем контуре.

## Документационная опора

### Probability Calibration

Документация `scikit-learn` фиксирует:

- калибровка переводит выход `decision_function` или `predict_proba` в
  откалиброванную вероятность;
- калибратор должен обучаться на данных, независимых от обучающего поднабора
  базового классификатора;
- `CalibratedClassifierCV` делает это через внутреннюю кросс-валидацию;
- для многоклассовой задачи калибратор подгоняется отдельно по классам;
- при `ensemble=True` calibration идет по folds и averaged на predict-time;
- `isotonic` не рекомендуется при слишком малом числе calibration samples;
- `sigmoid` безопаснее как первый вариант по умолчанию на ограниченных данных.

Официальные источники:

- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV)

### Threshold Tuning

Документация `scikit-learn` фиксирует:

- настройка порога — это отдельный шаг постобработки поверх оценки или
  вероятности классификатора;
- `TunedThresholdClassifierCV` пост-оптимизирует decision threshold через CV;
- default metric у него — `balanced_accuracy`;
- `cv="prefit"` нельзя использовать на том же датасете, на котором estimator
  обучался, иначе будет overfitting;
- default internal CV — `5-fold stratified`.

Официальные источники:

- [Tuning the decision threshold for class prediction](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [TunedThresholdClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html)

### Metrics

Документация `scikit-learn` фиксирует:

- `balanced_accuracy` подходит для binary и multiclass задач при class imbalance;
- `classification_report` удобен как structured report по precision / recall / F1
  per class и macro/weighted averages.

Официальные источники:

- [balanced_accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)
- [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)

## Общие правила проекта

### Правило 1. Калибровка не живет внутри базовой модели

Базовый классификатор и калибратор — разные сущности.

Правило проекта:

- сначала train base estimator;
- потом отдельно calibration stage;
- потом отдельно decision threshold stage;
- потом отдельно final decision mapping.

### Правило 2. Test split не используется для калибровки и настройки порогов

Calibration и threshold tuning разрешены только на:

- внутреннем CV;
- отдельном validation split;
- специальном holdout, который не является final test.

Запрещено:

- fit classifier на set A и tune threshold на том же set A;
- fit classifier, calibrator и final threshold на test split.

### Правило 3. Пороги должны версионироваться

Порог — это не магическое число в коде.

Для каждого tuned/manual threshold должны быть явные поля:

- `threshold_name`
- `threshold_value`
- `threshold_metric`
- `threshold_fit_scope`
- `threshold_policy_version`

## Политика по этапам

## Этап A. `ID/OOD`

### Почему этот этап настраиваем первым

`ID/OOD` у нас binary и уже имеет сильный baseline:

- test balanced_accuracy `0.926215`
- test macro_f1 `0.944521`

Значит именно этот этап является первым кандидатом для настройки порогов.

### Базовая политика

Для второй рабочей версии:

- base estimator family пока не меняем;
- используем существующий базовый `HistGradientBoostingClassifier`;
- тюнинг threshold делаем отдельно от fit.

### Политика калибровки

Для `ID/OOD` допускается:

- калибровка вероятностей через `CalibratedClassifierCV`
- default calibration method: `sigmoid`

Причина:

- `sigmoid` безопаснее как вариант по умолчанию;
- не требует большого calibration mass как `isotonic`;
- лучше соответствует осторожной первой рабочей версии.

`isotonic`:

- оставляем как отдельный следующий эксперимент;
- не делаем вариантом по умолчанию без отдельного подтверждающего прогона.

### Политика порогов

Для `ID/OOD` threshold tuning проектируем через:

- `TunedThresholdClassifierCV`

Default scoring:

- `balanced_accuracy`

Пояснение:

- это стандартная метрика по умолчанию для `TunedThresholdClassifierCV`;
- он согласуется с нашей задачей, где и false `OOD`, и missed `OOD` важны;
- это лучше, чем жестко прописанное `0.5`.

### Выходной контракт

`ID/OOD` stage должен уметь отдавать:

- `ood_probability`
- `id_probability`
- `ood_threshold`
- `ood_threshold_policy_version`
- `ood_decision`
  - `in_domain`
  - `candidate_ood`
  - `ood`

### Решение для первой рабочей версии

Этот документ не фиксирует конкретные numeric thresholds.

Фиксируется только процесс:

- tuned threshold определяем на validation/CV;
- `ood` и `candidate_ood` разделяем в отдельном manual policy layer;
- `candidate_ood` не схлопываем с clean `in_domain`.

## Этап B. `Coarse`

### Базовая политика

`coarse` baseline уже достаточно силен:

- test balanced_accuracy `0.992379`

Поэтому:

- coarse stage не становится primary candidate для threshold tuning first;
- сначала используем его как стабильный классификатор с вероятностным выходом.

### Политика калибровки

Для второй рабочей версии:

- calibration для `coarse` не является обязательным первым шагом;
- сначала сохраняем raw `predict_proba` и confidence margin;
- отдельный calibration experiment допустим позже.

Причина:

- coarse metrics уже очень высокие;
- основной следующий выигрыш ожидается не от повторной калибровки coarse,
  а от более качественной логики передачи между этапами.

### Политика уверенности

Для `coarse` фиксируем decision contract, а не numeric threshold:

- `coarse_probability_max`
- `coarse_probability_margin`
- `coarse_policy_version`

Правило проекта:

- numeric threshold для `coarse -> refinement` подбирается на validation;
- threshold не тюним на test;
- threshold не живет внутри base model code.

## Этап C. `Refinement`

### Базовая политика

`refinement` во второй рабочей версии идет через разбиение по семействам:

- `A`
- `B`
- `F`
- `G`
- `K`
- `M`

`O` остается coarse-only.

### Политика калибровки

Калибровка для семейств refinement допускается, но не должна быть обязательной
в первом рабочем шаге.

Причина:

- сначала нужно материализовать задачи по семействам и проверить их базовое
  качество;
- только потом решать, где calibration реально улучшает reliability.

Default future policy:

- per-family calibration module, если и когда включаем;
- default method: `sigmoid`;
- `isotonic` только как targeted experiment на достаточно больших family slices.

### Выходной контракт

Каждая family model должна уметь отдавать:

- `refinement_probability_max`
- `refinement_probability_margin`
- `refinement_threshold`
- `refinement_threshold_policy_version`
- `refinement_decision`
  - `accepted`
  - `rejected_to_unknown`
  - `not_attempted`

### Политика передачи

Refinement запускается только если одновременно:

- `OOD` stage не перевел объект в `ood`
- `coarse` class входит в refinement-enabled list
- coarse confidence проходит agreed threshold
- family-specific subclass policy допускает target

Если это не выполнено:

- refinement не запускается принудительно;
- объект остается на coarse-level или уходит в `unknown/review`.

## Политика `unknown/review`

`unknown/review` — это отдельный исход, а не ошибка метрики.

Объект должен попадать в `unknown/review`, если:

- `quality_state <> 'pass'`
- `ood_state = 'candidate_ood'`
- `ood_state = 'ood'`, но policy требует ручного review bucket
- coarse confidence ниже agreed threshold
- coarse class coarse-only для refinement
- refinement family отказалась от confident decision

## Чего не делаем

- не вшиваем numeric thresholds в training runner;
- не смешиваем calibration, threshold tuning и final decision mapping в одном модуле;
- не используем `cv="prefit"` на том же наборе, где fit-ился estimator;
- не вводим multiclass threshold tuning ad hoc без validation contract;
- не делаем `isotonic` вариантом по умолчанию без отдельного обоснования.

## Критерий готовности документа

Документ считается зафиксированным, когда:

- политика калибровки зафиксирована отдельно от кода моделей;
- threshold policy зафиксирована отдельно от decision mapping;
- документация `scikit-learn` и правила проекта разведены явно;
- следующий шаг может проектировать final decision layer без архитектурной
  импровизации.

## Состояние реализации (`2026-03-28`)

Политика уже переведена в отдельные модули:

- [calibration.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/posthoc/calibration.py)
- [threshold_tuning.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/posthoc/threshold_tuning.py)
- [id_ood_gate.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/posthoc/id_ood_gate.py)
- [run_id_ood_posthoc_gate.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/training/run_id_ood_posthoc_gate.py)
- [id_ood_threshold_artifacts.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/reporting/id_ood_threshold_artifacts.py)

Важный технический фикс:

- custom model wrappers приведены к корректной sklearn classifier semantics через
  `ClassifierMixin` слева от `BaseEstimator`;
- это потребовалось, чтобы official `CalibratedClassifierCV` и
  `TunedThresholdClassifierCV` работали без дополнительных обходных адаптеров.

Дополнительный фикс на уровне выборки:

- `ID/OOD` loader больше не режет `limit` по `domain_target ASC`;
- ограниченные выборки теперь упорядочиваются по `random_index`, чтобы CV не
  получал одноклассовый slice.

Контракт артефактов теперь тоже закрыт:

- tuned threshold policy сохраняется отдельно от model artifact;
- `decision CLI` читает не только сохраненную модель, но и сохраненный артефакт
  порогов;
- это исключает скрытый ручной дрейф порогов между ноутбуком, `CLI` и
  сквозным прогоном.

Проверочный прогон для `gaia_id_ood_classification` (`HistGradientBoosting`,
`limit=5000`):

- train accuracy: `0.985677`
- train balanced_accuracy: `0.990211`
- train macro_f1: `0.982064`
- test accuracy: `0.969272`
- test balanced_accuracy: `0.968762`
- test macro_f1: `0.961488`
- tuned threshold: `0.041898`

Проверка по срезу постобработки:

- `ruff` ok
- `mypy` ok
- `pyright` ok
- targeted `pytest` ok
