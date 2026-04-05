# Coarse O/B Feature Separability Review TZ

## Цель

Этот пакет открывает следующий narrow step после
[coarse_o_train_support_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_train_support_review_round1_ru.md).

Задача:

- не ретрейнить coarse-модель вслепую;
- проверить, насколько true `O` и true `B` вообще separable по текущему coarse feature contract;
- посмотреть, как ведет себя текущий coarse artifact именно на train-time `O/B` boundary source;
- понять, это feature-boundary problem или downstream domain-shift problem.

## Official И Авторитетная Опора

### Gaia / Spectral Semantics

- [Gaia DR3 documentation index](https://gea.esac.esa.int/archive/documentation/GDR3/)
- [Gaia DR3 astrophysical_parameters semantics](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_astrophysical_parameter_tables/ssec_dm_astrophysical_parameters.html)
- [Astrophysical parameters associated to hot stars in Gaia DR3](https://www.aanda.org/articles/aa/full_html/2023/06/aa43709-22/aa43709-22.html)
- [NASA spectral classification overview](https://asd.gsfc.nasa.gov/archive/star_class/spectral_classification.html)
- [ESA star types overview](https://www.esa.int/Science_Exploration/Space_Science/Star_types)

### Python / pandas / scikit-learn

- [typing — Python docs](https://docs.python.org/3/library/typing.html)
- [pandas groupby](https://pandas.pydata.org/docs/user_guide/groupby.html)
- [roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
- [permutation_importance](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html)
- [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)

## Почему Нужен Отдельный Separability Review

Train support review уже показал:

- true `O` не starving в source;
- hot `O/B` boundary по support симметрична;
- hottest `O` tail полноценно присутствует в train/test.

Значит следующий корректный вопрос уже другой:

- различимы ли true `O` и true `B` по текущим coarse-признакам вообще;
- и что делает текущий coarse artifact на собственном train-time `O/B` source.

## Project Baseline

Review работает на train-time boundary source:

- source: prepared coarse training frame
- baseline slice:
  - `spec_class IN ('O', 'B')`
  - `teff_gspphot >= 10000 K`

Это project baseline, а не официальный hard cut Gaia.

## Пакет OBS-C

### OBS-C01. Зафиксировать Separability Contract

Что нужно уметь показать:

1. сколько true `O` и true `B` входит в train-time boundary source;
2. как текущий coarse artifact предсказывает этот boundary source;
3. как выглядят `P(O)` и `P(B)` на true `O` и true `B`;
4. какие признаки дают лучшую single-feature separability;
5. какие признаки действительно использует coarse artifact на этом slice.

### OBS-C02. Собрать Typed Review Helper

Файлы:

- `src/exohost/reporting/archive_research/coarse_ob_feature_separability_review.py`
- `tests/archive_research/archived_coarse_ob_feature_separability_review.py`

Что должен уметь helper:

- строить train-time hot `O/B` boundary source;
- считать coarse scoring на boundary source;
- строить class-level physics summary;
- считать univariate separability по ROC AUC;
- считать permutation importance текущего coarse artifact.

### OBS-C03. Собрать Notebook Review

Файл:

- `analysis/notebooks/archive_research/15_coarse_ob_feature_separability_review.ipynb`

Notebook должен отвечать:

- separable ли true `O` и true `B` по текущим coarse-признакам;
- где strongest signal: `teff`, `radius`, `bp_rp` и т.д.;
- умеет ли текущий coarse artifact различать `O/B` на собственном train-time source;
- является ли основной следующий риск model failure или downstream domain shift.

### OBS-C04. Зафиксировать Round 1 Findings

Файл:

- `docs/methodology/archive_research/coarse_ob_feature_separability_review_round1_ru.md`

После первого review решаем:

- нужен ли retrain coarse model вообще;
- нужен ли narrow `O/B` feature-policy change;
- или сначала нужно разбирать downstream source mismatch.

## Инженерный Стандарт

- `1 файл = 1 ответственность`
- без монолитов
- `PEP 8`
- явная типизация
- простая логика раньше сложной
- комментарии только там, где они реально помогают читать код
- после каждого slice:
  - `ruff`
  - `mypy`
  - `pyright`
  - targeted `pytest`
- notebook отдельно проходит `nbclient` execution

## Related

- [coarse_o_train_support_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_train_support_review_round1_ru.md)
- [coarse_ob_boundary_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_ob_boundary_review_round1_ru.md)
- [stabilization_issue_ledger_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/stabilization_issue_ledger_ru.md)
