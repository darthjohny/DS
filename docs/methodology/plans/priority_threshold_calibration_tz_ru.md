# Priority Threshold Calibration TZ

## Цель

Этот пакет фиксирует следующий шаг после `priority_review_round1` и
`host_priority_calibration_round1`.

Наша задача:

- не переписывать ranking-формулу вслепую;
- сначала проверить, достаточно ли ужать `priority` thresholds;
- и только потом решать, нужен ли отдельный scaling-layer поверх
  `host_similarity_score` или `priority_score`.

## Official Опора

- [scikit-learn: Tuning the decision threshold for class prediction](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [Python dataclasses](https://docs.python.org/3/library/dataclasses.html)
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [Gaia DR3 astrophysical_parameters semantics](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_astrophysical_parameter_tables/ssec_dm_astrophysical_parameters.html)

Важно:

- `priority_score` — это не official score из Gaia или NASA;
- поэтому threshold policy у нас проектная;
- official docs здесь нужны, чтобы:
  - разделять probability-like signal и downstream threshold decision;
  - не смешивать калибровку модели и бизнес-решение по cutoffs.

## Почему Начинаем С Threshold Review

После `priority_review_round1` и `host_priority_calibration_round1` у нас уже есть:

- сильный `host_similarity_score`;
- понятная причина saturation;
- рабочий final pipeline;

Но пока еще нет доказательства, что нужно сразу:

- менять ranking weights;
- вводить scaling;
- или добавлять post-hoc calibration в host-model.

Поэтому следующий минимальный и чистый шаг:

- review threshold variants на уже посчитанном `priority_score`.

## Пакет PT-C

### PT-C01. Зафиксировать Threshold Review Contract

Цель:

- определить, как сравниваем threshold variants поверх current `priority_score`.

Что фиксируем:

- baseline thresholds:
  - `high_min = 0.75`
  - `medium_min = 0.45`
- variant review сравнивает:
  - label distribution
  - label transitions
  - class-level impact
- сам score пока не пересчитываем

### PT-C02. Собрать Threshold Review Helper Layer

Цель:

- вынести threshold review в отдельный typed helper,
  а не делать ad hoc groupby внутри notebook.

Файлы:

- `src/exohost/reporting/priority_threshold_review.py`
- `tests/unit/test_priority_threshold_review.py`

### PT-C03. Собрать Notebook Для Threshold Review

Цель:

- получить reproducible review для:
  - baseline
  - stricter high-threshold
  - stricter medium+high threshold
  - class-level impact

Файл:

- `analysis/notebooks/technical/priority_threshold_review.ipynb`

Статус:

- закрыто в code/review слое.

### PT-C04. Принять Threshold Decision

Статус:

- review round 1 проведен:
  - [priority_threshold_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/priority_threshold_review_round1_ru.md)
- live round 2 проведен:
  - [priority_threshold_review_round2_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/priority_threshold_review_round2_ru.md)

Принятое решение:

- active review thresholds:
  - `high_min = 0.85`
  - `medium_min = 0.55`
- `final_decision_review.ipynb` переводим на новый threshold candidate run
- scaling package пока не открываем

### PT-C05. Только Если Threshold Review Недостаточен

Следующий пакет:

- `priority scaling review`

Его открываем только если:

- threshold variants не дают внятной top-zone;
- или saturation остается operational problem даже после ужатия thresholds.

Текущий статус:

- не открыт;
- сначала нужен star-level review на live round 2.

## Related

- [priority_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/priority_review_round1_ru.md)
- [host_priority_calibration_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/host_priority_calibration_round1_ru.md)
- [post_run_stabilization_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/post_run_stabilization_tz_ru.md)
