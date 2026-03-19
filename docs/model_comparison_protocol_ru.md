# Протокол сравнения моделей для baseline-блока

Дата: 13 марта 2026 года

## 1. Назначение документа

Этот документ фиксирует канонический protocol сравнения моделей для ВКР.

Его задача:

- определить, что именно сравнивается;
- зафиксировать dataset и split;
- зафиксировать метрики и итоговые артефакты;
- зафиксировать место validation-layer относительно benchmark;
- заранее отделить baseline-слой от production-контура.

Документ намеренно не содержит кода. Он задаёт стабильный контракт, на
который дальше будут опираться модули `analysis/model_comparison/`.

Связанный документ:

- `docs/model_validation_protocol_ru.md`
- `docs/orchestrator_host_prioritization_canon_ru.md`

## 2. Объект сравнения

В первой волне сравнивается не весь production pipeline целиком, а
`host-scoring head` внутри уже существующей физической маршрутизации.

Это означает:

- `router` и его `OOD/Unknown`-контракт считаются общими и фиксированными;
- preprocessing и input-layer не меняются;
- baseline-модели заменяют только слой оценки `host vs field` внутри
  ветки `M/K/G/F dwarf`.

Такой scope позволяет:

- не ломать production `src/`;
- не смешивать научный benchmark и боевой код;
- сравнивать модели честно на одном и том же наборе признаков.

## 3. Модели первой волны

В baseline-блок текущего этапа входят четыре модели:

1. `main_contrastive_v1`
   Текущая основная модель проекта: contrastive `host-vs-field` Gaussian.
2. `baseline_legacy_gaussian`
   Legacy Gaussian similarity по host-популяции.
3. `baseline_random_forest`
   Классический ML baseline на задаче `host vs field`.
4. `baseline_mlp_small`
   Компактный нейросетевой baseline на задаче `host vs field`.

Важно:

- `RandomForest` в первой волне делается class-specific для
  `M/K/G/F`, а не как один глобальный классификатор на все классы;
- `baseline_mlp_small` в первой волне тоже делается class-specific для
  `M/K/G/F`;
- `MLP` не заменяет `router` и не заменяет `decision layer`;
- более сложные ИНС-архитектуры не входят в текущий scope.

## 4. Источники данных

### 4.1 Основной supervised benchmark

Для train/test сравнения используется размеченный `host/field` dataset:

- host positives:
  `lab.v_nasa_gaia_train_dwarfs`
- field negatives:
  `lab.v_gaia_ref_mkgf_dwarfs`

Ограничения первой волны:

- только `M/K/G/F dwarf`;
- только признаки:
  `teff_gspphot`, `logg_gspphot`, `radius_gspphot`;
- целевая колонка:
  `is_host`;
- обязательные служебные поля:
  `source_id`, `spec_class`.

### 4.1.1 Контракт `baseline_mlp_small`

Нейросетевой baseline текущей волны подчиняется отдельным ограничениям:

- работает только на `M/K/G/F dwarf`;
- использует только три базовых физических признака:
  `teff_gspphot`, `logg_gspphot`, `radius_gspphot`;
- реализуется как `StandardScaler + MLPClassifier`;
- строится class-specific, то есть отдельная маленькая сеть на каждый
  класс `M`, `K`, `G`, `F`;
- выдаёт только `host-vs-field` score;
- не включает в себя downstream-факторы
  `mh_gspphot`, `parallax`, `parallax_over_error`, `ruwe`, `bp_rp`,
  `validation_factor`.

Все дополнительные decision-факторы, как и для остальных моделей,
остаются в общем оркестраторе и применяются после model head.

### 4.2 Pipeline snapshot

Для прикладного snapshot-режима используется relation:

- `public.gaia_dr3_training`

Этот режим не является главным supervised benchmark, потому что на нём нет
ground truth по наличию экзопланет. Он нужен только как operational preview:

- как меняется ranking;
- как выглядит top-k;
- как baseline ведёт себя на живом batch после общего `router + OOD`.

## 5. Split-контракт

Train/test split должен быть:

- детерминированным;
- воспроизводимым;
- общим для всех моделей;
- защищённым от leakage по `source_id`.

Правила:

1. В benchmark dataset `source_id` должен быть уникальным.
2. Если один и тот же `source_id` попал в набор более одного раза,
   загрузка benchmark должна завершаться ошибкой.
3. Split делается по всему benchmark frame, а не отдельно по моделям.
4. Stratify-метка строится как комбинация:
   `spec_class + is_host`.
5. Для первой волны внешний benchmark использует только `train/test`,
   а подбор гиперпараметров делается внутри train-части.

Параметры первой волны:

- `random_state = 42`
- `test_size = 0.30`

## 5.1 Контракт tuning-контура

Подбор гиперпараметров для benchmark comparison-layer должен быть:

- воспроизводимым;
- общим по логике для всех четырёх моделей;
- изолированным внутри train split;
- свободным от leakage в test.

Правила:

1. Для canonical benchmark используется `StratifiedKFold`.
2. Число fold-ов фиксируется как `10`.
3. `shuffle=True`, `random_state=42`.
4. Лучшая конфигурация выбирается по `refit_metric`.
5. После выбора best params модель переобучается на всём train split.
6. Test split используется только для финальной supervised оценки.

По модели:

- `baseline_random_forest` и `baseline_mlp_small` используют компактный
  `GridSearchCV`;
- `main_contrastive_v1` и `baseline_legacy_gaussian` используют
  эквивалентный manual CV search, потому что не являются sklearn-estimator
  из коробки;
- во всех случаях comparison-layer обязан сохранять search summary
  с лучшими параметрами и CV score.

## 6. Метрики

### 6.1 Основные supervised метрики

Для основной задачи `host vs field` сравниваются:

- `ROC-AUC`
- `PR-AUC`
- `Brier score`
- `precision@k`

Эти метрики обязательны для всех четырёх моделей.

### 6.2 Дополнительные сводки

Дополнительно допускаются:

- распределения score по `host` и `field`;
- class-wise таблицы по `M/K/G/F`;
- enrichment в top-k;
- краткий operational snapshot на `public.gaia_dr3_training`.

### 6.3 Threshold-based quality

Помимо threshold-free benchmark-метрик comparison-layer должен считать
классический quality-блок для бинарной классификации.

В него входят:

- `confusion matrix`;
- `precision`;
- `recall`;
- `f1`;
- `specificity`;
- `balanced_accuracy`;
- `accuracy` как вторичная метрика.

Правила первой волны:

- quality считается после выбора порога классификации;
- порог выбирается только на `train` split;
- test split не участвует в выборе порога;
- в первой волне используется один глобальный threshold на модель;
- threshold выбирается по `max F1`;
- quality-блок дополняет benchmark и не подменяет его.

### 6.4 Что не сравниваем напрямую

Не сравниваем “в лоб” значения score разных моделей:

- legacy `similarity`;
- contrastive `host_posterior`;
- `RandomForest predict_proba`.
- `MLP predict_proba`.

Сравнение делается по метрикам, ranking-качеству и quality-показателям,
а не по сырым шкалам.

## 7. Выходные артефакты

Comparison-layer должен порождать артефакты в `experiments/model_comparison/`.

Минимальный набор:

- markdown summary;
- CSV-таблица с итоговыми метриками;
- CSV-таблица `search_summary` с лучшими параметрами и CV score;
- CSV-таблица `thresholds` с train-selected classification threshold;
- CSV-таблица `quality_summary` с threshold-based quality-метриками;
- CSV-таблица `quality_classwise` с class-wise quality-показателями;
- CSV-таблица `confusion_matrices` с `TP/FP/TN/FN`;
- CSV-таблица `generalization` с `train/test` и `CV/test` diagnostics;
- CSV- и markdown-таблица `generalization_audit` с per-model verdict;
- markdown и CSV-артефакты `dataset_validation` до model fitting;
- CSV или parquet со score-frame test-части;
- отдельный snapshot report для `public.gaia_dr3_training`;
- top-k CSV по каждой модели в snapshot-режиме.

Важно:

- `dataset_validation` является отдельным preflight-слоем и должен
  сохраняться до benchmark markdown summary;
- `generalization_audit` не заменяет benchmark-метрики, а дополняет их
  как anti-overfitting слой;
- quality-блок не заменяет threshold-free benchmark-метрики, а даёт
  привычную прикладную интерпретацию в терминах бинарной классификации;
- snapshot остаётся отдельным operational preview и не считается главным
  supervised доказательством generalization.

## 8. Архитектурные ограничения

Baseline-слой не должен:

- писать результаты в production result-таблицы;
- менять `src/priority_pipeline`;
- менять `src/router_model`;
- дублировать production-код копипастой;
- собираться в один монолитный модуль.

Правильная файловая структура первой волны:

- `analysis/model_comparison/contracts.py`
- `analysis/model_comparison/tuning.py`
- `analysis/model_comparison/manual_search.py`
- `analysis/model_comparison/data.py`
- `analysis/model_comparison/contrastive.py`
- `analysis/model_comparison/legacy_gaussian.py`
- `analysis/model_comparison/random_forest.py`
- `analysis/model_comparison/mlp_baseline.py`
- `analysis/model_comparison/metrics.py`
- `analysis/model_comparison/quality.py`
- `analysis/model_comparison/reporting.py`
- `analysis/model_comparison/snapshot.py`
- `analysis/model_comparison/cli.py`

## 9. Definition of Done для baseline-этапа

Baseline-этап считается закрытым, когда:

1. Есть единый benchmark dataset и детерминированный split.
2. `main_contrastive_v1`, `baseline_legacy_gaussian`,
   `baseline_random_forest` и `baseline_mlp_small` считаются на одном
   test split.
3. Внешний benchmark использует `test_size = 0.30`.
4. Для всех моделей работает `10-fold` tuning-контур внутри train split.
5. До benchmark запускается dataset validation layer.
6. Для всех моделей построены общие supervised метрики.
7. Для всех моделей есть threshold-based quality с train-selected
   threshold.
8. Есть `generalization diagnostics` и `per-model audit`.
9. Есть единый markdown/CSV comparative report, `search_summary`,
   `thresholds`, quality-артефакты и validation-артефакты.
10. Есть минимальные smoke-tests для split, tuning, wrappers,
   validation-layer.
11. `README` и материалы ВКР ссылаются на этот protocol.
