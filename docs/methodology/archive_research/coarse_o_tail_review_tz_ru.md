# Coarse O-Tail Review TZ

## Цель

Этот пакет открывает отдельный review-слой для редкого класса `O` в coarse
classification pipeline.

Задача пакета:

- не менять модель вслепую;
- сначала воспроизводимо показать, где именно пропадает `O`;
- отделить `quality_gate`-эффект от поведения coarse-model;
- зафиксировать evidence до любого retrain, rebalance или policy change.

## Official Опора

### Gaia

- [Gaia DR3 documentation index](https://gea.esac.esa.int/archive/documentation/GDR3/)
- [Gaia DR3 gaia_source semantics](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html)
- [Gaia DR3 astrophysical_parameters semantics](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_astrophysical_parameter_tables/ssec_dm_astrophysical_parameters.html)
- [Gaia DR3 Apsis overview](https://gea.esac.esa.int/archive/documentation/GDR3/Data_analysis/chap_cu8par/sec_cu8par_intro/ssec_cu8par_intro_apsis.html)

### scikit-learn

- [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
- [confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
- [balanced_accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)
- [compute_class_weight](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)

### Python / pandas

- [typing — Python docs](https://docs.python.org/3/library/typing.html)
- [collections.abc — Python docs](https://docs.python.org/3/library/collections.abc.html)
- [pandas missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html)

## Почему Это Отдельный Пакет

По текущему stabilization-review уже видно:

- `O` не исчезает на уровне source;
- `O` не полностью вырезается `quality_gate`;
- но в final outputs почти не появляется как `final_coarse_class`.

Это уже не `priority`-проблема и не notebook drift.

Значит нужен отдельный narrow package под rare-tail review, а не очередная
правка общего pipeline.

## Пакет OR-C

### OR-C01. Зафиксировать Review Contract

Что нужно уметь показать:

1. сколько `O` строк есть в source;
2. сколько из них проходит `quality_state = pass`;
3. что coarse-model предсказывает именно на pass-части;
4. как эти же `O` строки выглядят в текущем `final decision` run;
5. high-confidence ошибки, если `O` системно уходит в другой coarse class.

### OR-C02. Собрать Loader Для `O`-Source

Файлы:

- `src/exohost/datasets/archive_research/load_coarse_o_review_dataset.py`
- `tests/archive_research/archived_load_coarse_o_review_dataset.py`

Требования:

- читать `lab.gaia_mk_quality_gated`;
- фильтровать `spectral_class = 'O'`;
- без notebook-level SQL;
- поддерживать optional filter по `quality_state`.

### OR-C03. Собрать Typed Review Helper

Файлы:

- `src/exohost/reporting/archive_research/coarse_o_review.py`
- `tests/archive_research/archived_coarse_o_review.py`

Что должен уметь helper:

- готовить `O`-source summary;
- строить coarse scoring только по pass-строкам;
- объединять `O` source с final decision run;
- показывать распределение predicted coarse labels;
- показывать high-confidence non-`O` predictions.

### OR-C04. Собрать Notebook Review

Файл:

- `analysis/notebooks/archive_research/11_coarse_o_tail_review.ipynb`

Notebook должен отвечать:

- где именно пропадает `O`;
- сколько `O` теряется на gate;
- что coarse-model делает с surviving `O`;
- какой downstream outcome получают истинные `O` объекты.

### OR-C05. Зафиксировать Round 1 Findings

Файл:

- `docs/methodology/archive_research/coarse_o_tail_review_round1_ru.md`

После первого review решаем:

- нужен ли rebalance;
- нужен ли отдельный threshold/policy для `O`;
- или проблема объясняется source/physics ограничениями и не требует срочного
  retrain.

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
- notebook отдельно проходит compile/execute check

## Related

- [post_run_stabilization_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/post_run_stabilization_tz_ru.md)
- [stabilization_issue_ledger_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/stabilization_issue_ledger_ru.md)
- [star_level_review_round2_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/star_level_review_round2_ru.md)
