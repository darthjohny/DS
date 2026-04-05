# Coarse O/B Domain Shift Review TZ

## Контекст

После цепочки review:

- [stabilization_issue_ledger_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/stabilization_issue_ledger_ru.md)
- [archive_research/coarse_ob_boundary_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_ob_boundary_review_round1_ru.md)
- [archive_research/coarse_o_train_support_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_train_support_review_round1_ru.md)
- [archive_research/coarse_ob_feature_separability_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_ob_feature_separability_review_round1_ru.md)

у нас осталась одна главная рабочая гипотеза по `SI-009`:

- текущий `O -> B` collapse возникает не из-за недостатка support;
- не из-за отсутствия separability на train-time source;
- а из-за `domain shift / source mismatch` между:
  - train-time `O/B` boundary source
  - downstream hot pass `O/B` boundary source

Этот пакет нужен, чтобы проверить именно эту гипотезу.

## Цель

Сравнить:

- train-time hot `O/B` boundary из coarse training source;
- downstream hot pass `O/B` boundary из `lab.gaia_mk_quality_gated`;

и понять:

- насколько они различаются по физике;
- насколько они различаются по missingness;
- одинаково ли current coarse artifact ведет себя на обоих domains;
- какой feature/domain shift выглядит главным кандидатом на корень `O -> B`.

## Official Опора

### Gaia / Astrophysics

- [Gaia DR3 documentation index](https://gea.esac.esa.int/archive/documentation/GDR3/)
- [Gaia DR3 gaia_source semantics](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html)
- [Gaia DR3 astrophysical_parameters semantics](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_astrophysical_parameter_tables/ssec_dm_astrophysical_parameters.html)
- [Gaia DR3 GSP-Phot](https://gea.esac.esa.int/archive/documentation/GDR3/Data_analysis/chap_cu8par/sec_cu8par_apsis/ssec_cu8par_apsis_gspphot.html)

### Python / pandas / scikit-learn

- [typing — Python docs](https://docs.python.org/3/library/typing.html)
- [pandas missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
- [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)

## Scope

В пакет входит только review-layer.

Не входит:

- retrain coarse model;
- class weighting;
- threshold tuning;
- правка production routing.

## Рабочая Гипотеза

Если train-time `O/B` boundary и downstream hot pass `O/B` boundary
сильно различаются по физике, feature coverage или missingness,
то следующий шаг должен быть не retrain вслепую, а source-alignment review.

## Реализация

### OBDS-01. Собрать downstream boundary loader

Файл:

- `src/exohost/datasets/load_coarse_ob_downstream_boundary_dataset.py`

Что должен делать:

- читать `lab.gaia_mk_quality_gated`;
- выбирать только:
  - `spectral_class IN ('O', 'B')`
  - `quality_state = 'pass'`
  - `teff_gspphot >= 10000 K`
- возвращать reproducible downstream hot pass boundary source.

### OBDS-02. Собрать review contracts

Файл:

- `src/exohost/reporting/coarse_ob_domain_shift_contracts.py`

Что должно быть:

- config dataclass;
- bundle dataclass;
- никакой IO и никакого scoring.

### OBDS-03. Собрать scoring слой

Файл:

- `src/exohost/reporting/coarse_ob_domain_shift_scoring.py`

Что должен уметь:

- прогонять current coarse artifact на train-time и downstream boundary;
- возвращать compact scored frame;
- не смешивать scoring и bundle assembly.

### OBDS-04. Собрать frame builders

Файл:

- `src/exohost/reporting/coarse_ob_domain_shift_frames.py`

Что должно быть:

- membership summary;
- class balance by domain;
- prediction/confusion by domain;
- probability summary by domain и true class;
- physics summary by domain и true class;
- missingness summary;
- feature-level domain shift AUC внутри true `O` и true `B`.

### OBDS-05. Собрать bundle/load слой

Файл:

- `src/exohost/reporting/coarse_ob_domain_shift_bundle.py`

Что должен уметь:

- загрузить train-time boundary source;
- загрузить downstream boundary source;
- прогнать current coarse artifact на обоих domains;
- собрать единый typed bundle.

### OBDS-06. Публичный review facade

Файл:

- `src/exohost/reporting/coarse_ob_domain_shift_review.py`

Что должно быть:

- только re-export публичного API;
- без тяжелой логики.

### OBDS-07. Собрать notebook review

Файл:

- `analysis/notebooks/research/coarse_ob_domain_shift.ipynb`

Notebook должен отвечать:

- одинаков ли `O/B` balance в обоих domains;
- одинаково ли coarse artifact ведет себя на train-time и downstream domains;
- какие физические признаки смещены;
- где domain shift сильнее всего;
- следующий шаг: retrain или source-alignment.

### OBDS-08. Зафиксировать findings

Файл:

- `docs/methodology/coarse_ob_domain_shift_review_round1_ru.md`

## Инженерный Стандарт

- `1 файл = 1 ответственность`
- без монолитов
- `PEP 8`
- явная типизация
- простой код без лишней абстракции
- comments только там, где помогают чтению
- notebooks тонкие, логика в тестируемых модулях

## QA

После code-slice:

- `ruff`
- `mypy`
- `pyright`
- targeted `pytest`

После notebook:

- исполнение через `nbclient`

## Критерий Готовности

Пакет считается закрытым, если мы можем честно ответить:

- подтверждается ли гипотеза `domain shift`;
- какие именно признаки и quality-факторы сдвинуты;
- нужен ли следующий шаг на retrain,
  или сначала нужно выравнивать source.
