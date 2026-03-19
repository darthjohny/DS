# Протокол generalization validation

Дата: 14 марта 2026 года

## 1. Назначение документа

Этот документ фиксирует, как проект проверяет generalization и признаки
переобучения.

Документ нужен для того, чтобы не смешивать:

- comparative benchmark;
- generalization validation;
- operational snapshot;
- production runtime.

Связанный канонический документ по scientific objective и decision-layer:

- `docs/orchestrator_host_prioritization_canon_ru.md`

## 2. Главный принцип

Проект не считает один `train/test` benchmark достаточным доказательством
отсутствия переобучения.

Benchmark остаётся обязательным и каноническим, но generalization должна
подтверждаться отдельным validation-layer.

## 3. Что считается benchmark

Benchmark в проекте — это controlled supervised comparison на одном и том
же `host vs field` dataset.

Его роль:

- сравнивать модели на одинаковом split;
- сравнивать train/test supervised-метрики;
- сравнивать class-wise поведение;
- выбирать кандидатов для дальнейшей validation-оценки.

Benchmark не должен трактоваться как единственный финальный аргумент о
реальной обобщающей способности модели.

## 4. Что считается generalization validation

Generalization validation — это отдельный слой проверки после benchmark.

Кодовый слой текущей реализации живёт в:

- `analysis.model_validation`

Канонический каталог heavy validation артефактов:

- `experiments/model_validation/`

Его роль:

- проверить, насколько модель устойчива;
- оценить, есть ли признаки переобучения;
- показать, насколько выбор модели зависит от конкретного split;
- оценить стабильность по fold-ам и по спектральным классам.

В этот слой входят:

- dataset preflight до model fitting;
- train/test gap;
- CV/test gap;
- out-of-fold summaries;
- repeated split evaluation;
- class-wise stability;
- calibration sanity;
- optional learning/permutation diagnostics.

## 5. Что считается operational snapshot

Snapshot — это проверка поведения моделей на живом batch после общего
`router + OOD`.

Его роль:

- проверить operational ranking;
- оценить распределение `HIGH/MEDIUM/LOW`;
- посмотреть top-k и общую прикладную картину.

Snapshot не является главным supervised доказательством generalization,
потому что в нём нет надёжной целевой разметки задачи `host vs field`.

## 6. Что считается production runtime

Production runtime — это боевой путь:

- input validation;
- router scoring;
- OOD / UNKNOWN;
- host scoring;
- decision layer;
- persist результатов.

Runtime не должен выполнять heavy overfitting-checks.

В runtime допустимы только:

- schema checks;
- range checks;
- missing-value checks;
- OOD / UNKNOWN rules;
- optional drift-style sanity checks, если они не требуют ground truth и
  не раздувают latency.

## 7. Evaluation discipline

Проект придерживается следующих правил:

1. Test split не используется для hyperparameter search.
2. Test split не должен играть роль постоянного dev-set.
3. Snapshot не используется как главный supervised критерий выбора модели.
4. In-sample train metrics не считаются достаточным индикатором
   отсутствия переобучения.
5. Validation-layer должен дополнять benchmark, а не подменять его.

## 8. Dependency policy

На текущем этапе основной validation-layer строится на:

- `scikit-learn`;
- собственных typed contracts проекта.

Внешние библиотеки уровня `Deepchecks` допускаются только как optional
validation extension.

Причина:

- текущий проект стабилизирован под Python `3.13` и современный
  `scikit-learn`;
- external validation tools не должны ломать основной стек.

Подробности вынесены в:

- `docs/model_validation_dependency_spike_ru.md`

## 9. Практический порядок внедрения

1. Усилить architecture против optimistic bias.
2. Добавить data validation до fit/search.
3. Добавить per-model generalization audit.
4. Зафиксировать validation artifacts и findings.
5. Только после этого использовать результаты в narrative ВКР.

## 10. Текущее покрытие

На текущем этапе в project current state уже реализованы:

- `dataset validation layer` до model fitting;
- `Generalization Diagnostics` в benchmark reporting;
- `Per-model Generalization Audit` с отдельным CSV/markdown-слоем;
- typed contracts и smoke-tests для validation-layer.
- отдельный package `analysis.model_validation` с reproducible heavy
  validation run:
  - repeated split evaluation;
  - `*_repeated_splits.csv`;
  - `*_model_summary.csv`;
  - `*_generalization_summary.csv`;
  - `*_gap_diagnostics.csv`;
  - `*_risk_audit.csv`;
  - markdown report для heavy validation wave.

Текущие validation-артефакты comparison-layer:

- `*_dataset_validation.md`
- `*_dataset_validation_summary.csv`
- `*_dataset_validation_stratify.csv`
- `*_dataset_validation_feature_drift.csv`
- `*_generalization.csv`
- `*_generalization_audit.csv`
- `*_generalization_audit.md`

## 11. Внешние источники

Официальные материалы, на которые опирается validation design:

- [scikit-learn: learning_curve](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html)
- [scikit-learn: validation curves](https://scikit-learn.org/stable/modules/learning_curve.html)
- [scikit-learn: nested vs non-nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
- [Deepchecks: installation](https://docs.deepchecks.com/stable/getting-started/installation.html)
- [Deepchecks: train-test validation suite](https://docs.deepchecks.com/stable/api/generated/deepchecks.tabular.suites.train_test_validation.html)
- [Deepchecks: train-test validation checks](https://docs.deepchecks.com/stable/api/generated/deepchecks.tabular.checks.train_test_validation.html)

## 12. Статус

Статус документа: `accepted`

Этот protocol считается каноническим для следующей волны работ по
validation-layer и anti-overfitting safeguards.
