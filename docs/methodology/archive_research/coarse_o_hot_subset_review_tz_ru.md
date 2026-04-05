# Coarse O Hot-Subset Review TZ

## Цель

Этот пакет открывает следующий narrow step после
[coarse_o_tail_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_tail_review_round1_ru.md).

Задача:

- не трогать coarse-model вслепую;
- сначала выделить внутри true `O` только physically hot `O/B-like` subset;
- отделить label/physics inconsistency от настоящего rare-tail model failure;
- только после этого решать, нужен ли rebalance, special policy или source-cleaning.

## Official И Авторитетная Опора

### Gaia

- [Gaia DR3 documentation index](https://gea.esac.esa.int/archive/documentation/GDR3/)
- [Gaia DR3 gaia_source semantics](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html)
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

## Почему Нужен Hot-Subset

Round 1 по всему `O` source уже показал:

- `quality_gate` режет значимую часть `O`;
- surviving `O` rows coarse-model почти не переводит в `O`;
- часть surviving `O` уже физически не похожа на горячие звезды.

Значит следующий корректный шаг:

- не ребалансировать весь `O` пул;
- сначала посмотреть только на те строки, которые хотя бы минимально согласуются
  с hot `O/B-like` physics.

## Project Baseline Для Hot-Subset

Baseline-критерий для первой волны:

- `teff_gspphot >= 10000 K`

Почему именно так:

- по стандартной spectral sequence `B`-stars начинаются примерно от `10000 K`;
- значит threshold `10000 K` отсекает очевидно cool `F/G/K/M-like` contamination и
  оставляет минимум `O/B-like` physics.

Это не “официальный жёсткий Gaia cut”, а project baseline для narrow review.

## Пакет OH-C

### OH-C01. Зафиксировать Hot-Subset Contract

Что нужно уметь показать:

1. сколько true `O` строк попадает в `teff_gspphot >= 10000 K`;
2. сколько из них проходит `quality_state = pass`;
3. что coarse-model предсказывает именно на hot pass-subset;
4. какой downstream outcome получает hot-subset в final decision run;
5. остаются ли системные non-`O/B` high-confidence ошибки после удаления cool contamination.

### OH-C02. Собрать Typed Hot-Subset Helper

Файлы:

- `src/exohost/reporting/archive_research/coarse_o_hot_subset_review.py`
- `tests/archive_research/archived_coarse_o_hot_subset_review.py`

Требования:

- не дублировать DB loader;
- использовать уже собранный `true O` source;
- отдельно выделять hot-subset;
- не смешивать этот слой с общим `O` review helper.

### OH-C03. Собрать Notebook Review

Файл:

- `analysis/notebooks/archive_research/12_coarse_o_hot_subset_review.ipynb`

Notebook должен отвечать:

- насколько hot-subset меньше полного `O` source;
- исчезает ли после этого системный увод в `F/G/K`;
- остается ли проблема именно в `O -> B` routing;
- выглядит ли hot-subset уже научно чистым кандидатом на отдельный retrain/rebalance.

### OH-C04. Зафиксировать Round 1 Findings

Файл:

- `docs/methodology/archive_research/coarse_o_hot_subset_review_round1_ru.md`

После первого review решаем:

- нужен ли retrain только на hot-subset;
- нужен ли отдельный class-weight/rebalance для `O`;
- или even hot-subset все еще слишком шумный.

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

- [coarse_o_tail_review_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_tail_review_tz_ru.md)
- [coarse_o_tail_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_tail_review_round1_ru.md)
- [stabilization_issue_ledger_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/stabilization_issue_ledger_ru.md)
