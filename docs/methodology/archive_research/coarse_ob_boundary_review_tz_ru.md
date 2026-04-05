# Coarse O/B Boundary Review TZ

## Цель

Этот пакет открывает следующий narrow step после
[coarse_o_hot_subset_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_hot_subset_review_round1_ru.md).

Задача:

- не трогать всю coarse-модель;
- отдельно разобрать узкую границу `O vs B` на physically hot subset;
- понять, это genuine model-boundary issue или отражение source-label overlap;
- только после этого решать, нужен ли rebalance, class-weighting или source-cleaning.

## Official И Авторитетная Опора

### Gaia

- [Gaia DR3 documentation index](https://gea.esac.esa.int/archive/documentation/GDR3/)
- [Gaia DR3 astrophysical_parameters semantics](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_astrophysical_parameter_tables/ssec_dm_astrophysical_parameters.html)
- [Gaia DR3 Apsis overview](https://gea.esac.esa.int/archive/documentation/GDR3/Data_analysis/chap_cu8par/sec_cu8par_intro/ssec_cu8par_intro_apsis.html)
- [Astrophysical parameters associated to hot stars in Gaia DR3](https://www.aanda.org/articles/aa/full_html/2023/06/aa43709-22/aa43709-22.html)

### Spectral Temperature Semantics

- [NASA spectral classification overview](https://asd.gsfc.nasa.gov/archive/star_class/spectral_classification.html)
- [ESA star types overview](https://www.esa.int/Science_Exploration/Space_Science/Star_types)

### Python / pandas / scikit-learn

- [typing — Python docs](https://docs.python.org/3/library/typing.html)
- [collections.abc — Python docs](https://docs.python.org/3/library/collections.abc.html)
- [pandas missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
- [balanced_accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)
- [compute_class_weight](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)

## Почему Нужен Отдельный Boundary Review

Hot-subset review уже показал:

- после отсечения cool contamination `F/G/K/M` исчезают;
- physically hot subset почти целиком уходит в `B`;
- значит следующий научно корректный вопрос уже не “что не так с `O` вообще”,
  а “что происходит на границе `O/B`”.

Это отдельный слой проблемы:

- он уже не про broad source contamination;
- и еще не про retrain/rebalance.

## Project Baseline

Первая волна boundary-review работает на подмножестве:

- `spectral_class IN ('O', 'B')`
- `quality_state = 'pass'`
- `teff_gspphot >= 10000 K`

Это project baseline narrow slice, а не официальный hard cut Gaia.

## Пакет OB-C

### OB-C01. Зафиксировать Boundary Contract

Что нужно уметь показать:

1. сколько `O` и `B` строк есть в hot pass-boundary source;
2. как coarse-model распределяет предсказания между `O/B` и остальными классами;
3. как выглядят `P(O)` и `P(B)` на true `O` и true `B`;
4. есть ли физический overlap между true `O` и true `B` на этом срезе;
5. нужен ли следующий шаг уже в сторону retrain или class-weighting.

### OB-C02. Собрать Narrow Loader Для `O/B` Boundary Source

Файлы:

- `src/exohost/datasets/archive_research/load_coarse_ob_boundary_review_dataset.py`
- `tests/archive_research/archived_load_coarse_ob_boundary_review_dataset.py`

Требования:

- читать `lab.gaia_mk_quality_gated`;
- фильтровать `spectral_class IN ('O', 'B')`;
- поддерживать baseline `quality_state` и `teff`-условия;
- без notebook-level SQL.

### OB-C03. Собрать Typed Boundary Review Helper

Файлы:

- `src/exohost/reporting/archive_research/coarse_ob_boundary_review.py`
- `tests/archive_research/archived_coarse_ob_boundary_review.py`

Что должен уметь helper:

- строить source summary по true `O/B`;
- считать coarse scoring с полными `P(O)` и `P(B)`;
- строить confusion-like review frame;
- сравнивать медианную физику true `O` и true `B`;
- показывать high-confidence `O -> B` и `B -> O` случаи.

### OB-C04. Собрать Notebook Review

Файл:

- `analysis/notebooks/archive_research/13_coarse_ob_boundary_review.ipynb`

Notebook должен отвечать:

- это реальная boundary-problem `O/B` или остаточный source noise;
- насколько сильна asymmetry `O -> B` против `B -> O`;
- похожи ли true `O` и true `B` по hot-source physics.

### OB-C05. Зафиксировать Round 1 Findings

Файл:

- `docs/methodology/archive_research/coarse_ob_boundary_review_round1_ru.md`

После первого review решаем:

- нужен ли отдельный class-weight/rebalance для `O`;
- нужен ли narrow retrain на `O/B` slice;
- или сначала нужен source-cleaning hottest tail.

## Инженерный Стандарт

- `1 файл = 1 ответственность`
- без монолитов
- `PEP 8`
- явная типизация
- простая логика раньше сложной
- comments только там, где они реально помогают читать код
- после каждого slice:
  - `ruff`
  - `mypy`
  - `pyright`
  - targeted `pytest`
- notebook отдельно проходит `nbclient` execution

## Related

- [coarse_o_tail_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_tail_review_round1_ru.md)
- [coarse_o_hot_subset_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_hot_subset_review_round1_ru.md)
- [stabilization_issue_ledger_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/stabilization_issue_ledger_ru.md)
