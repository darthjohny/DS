# Host Priority Calibration TZ

## Цель

Этот документ фиксирует следующий пакет работ для калибровки `host/priority`
после подключения clean host relation и первого живого `priority` review.

Наша задача:

- не переписывать ranking-слой вслепую;
- сначала понять, насколько калиброван сам `host_similarity_score`;
- только потом менять `priority_score`, thresholds или scaling.

## Official Опора

- [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [scikit-learn: calibration_curve](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html)
- [scikit-learn: brier_score_loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html)
- [scikit-learn: log_loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)
- [scikit-learn: roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [PSCompPars semantics](https://exoplanetarchive.ipac.caltech.edu/docs/pscp_about.html)

## Почему Калибруем Не `priority`, А `host_similarity`

На текущем run:

- saturation `priority_score` не выглядит как code defect;
- он возникает из сочетания:
  - сильного `host_similarity_score`
  - высокого `class_priority_score` у target-классов
  - умеренно высокого `observability_score`

Поэтому first-line calibration target:

- `host_similarity_score`

А не:

- whole `priority` formula.

## Пакет HP-C

### HP-C01. Зафиксировать Calibration Contract

Цель:

- определить, что именно считаем калибровкой host-layer.

Что фиксируем:

- positive label: `host`
- probability signal: `host_similarity_score`
- quality metrics:
  - `brier_score`
  - `log_loss`
  - `roc_auc`
- reliability view:
  - `calibration_curve`
  - probability bins

### HP-C02. Собрать Generic Binary Calibration Helper

Цель:

- вынести reliability review в отдельный typed helper,
  а не считать его ad hoc в notebook.

Статус:

- закрыто.

Файлы:

- `src/exohost/reporting/binary_calibration_review.py`
- `tests/unit/test_binary_calibration_review.py`

### HP-C03. Спроектировать Host Calibration Source

Цель:

- получить reproducible source для host calibration-review.

Принцип:

- не использовать full-train fitted model как единственный источник истины;
- calibration-review должен опираться на train/test split или equivalent holdout path.

Статус:

- закрыто в коде.

Файлы:

- `src/exohost/reporting/host_calibration_source.py`
- `tests/unit/test_host_calibration_source.py`

### HP-C04. Собрать Host Calibration Review Layer

Цель:

- построить review поверх host task:
- reliability curve
- bin-level summary
- score summary
- coverage по `spec_class` и `evolution_stage`

Статус:

- закрыто в коде.

Файлы:

- `src/exohost/reporting/host_calibration_review.py`
- `tests/unit/test_host_calibration_review.py`

### HP-C05. Принять Calibration Decision

Статус:

- в работе.

Текущий live review артефакт:

- `analysis/notebooks/technical/host_priority_calibration_review.ipynb`
- baseline review findings:
  - [host_priority_calibration_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/host_priority_calibration_round1_ru.md)

Возможные решения:

- оставить `host_similarity_score` как есть;
- добавить post-hoc calibration;
- изменить scaling перед priority integration;
- ужать `high` threshold уже после calibration-review.

## Связанные Документы

- [priority_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/priority_review_round1_ru.md)
- [host_priority_integration_path_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/host_priority_integration_path_ru.md)
- [quality_gate_host_priority_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/quality_gate_host_priority_tz_ru.md)
