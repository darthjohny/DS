# Coarse O Train Support Review TZ

## Цель

Этот пакет открывает следующий narrow step после
[coarse_ob_boundary_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_ob_boundary_review_round1_ru.md).

Задача:

- не ретрейнить coarse-модель вслепую;
- проверить, хватает ли true `O` строк в самом coarse training source;
- восстановить именно тот `train/test split`, который использует benchmark;
- отдельно посмотреть hot `O` tail и narrow `O/B` boundary support до inference;
- только после этого решать, это support issue или уже model-boundary issue.

## Official И Авторитетная Опора

### Gaia / Spectral Semantics

- [Gaia DR3 documentation index](https://gea.esac.esa.int/archive/documentation/GDR3/)
- [Gaia DR3 astrophysical_parameters semantics](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_astrophysical_parameter_tables/ssec_dm_astrophysical_parameters.html)
- [Astrophysical parameters associated to hot stars in Gaia DR3](https://www.aanda.org/articles/aa/full_html/2023/06/aa43709-22/aa43709-22.html)
- [NASA spectral classification overview](https://asd.gsfc.nasa.gov/archive/star_class/spectral_classification.html)
- [ESA star types overview](https://www.esa.int/Science_Exploration/Space_Science/Star_types)

### Python / pandas / scikit-learn

- [typing — Python docs](https://docs.python.org/3/library/typing.html)
- [dataclasses — Python docs](https://docs.python.org/3/library/dataclasses.html)
- [pandas categorical and groupby user guide](https://pandas.pydata.org/docs/user_guide/groupby.html)
- [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)

## Почему Нужен Отдельный Train Support Review

Boundary review уже показал:

- true `O` на hot pass-slice всегда уходит в `B`;
- асимметрия `O -> B` односторонняя;
- сама модель почти не использует `O` как output.

Но до retrain нужно ответить на более базовый вопрос:

- сколько true `O` вообще есть в train и test;
- есть ли hot `O` tail в train/test в достаточном объеме;
- не теряем ли hottest часть `O` еще на этапе split;
- и совпадает ли reconstructed split с тем, что лежит в benchmark artifacts.

## Project Baseline

Этот review не вводит новый source.
Он использует уже существующий coarse training contract и повторяет текущую benchmark policy:

- source: `lab.v_gaia_id_coarse_training`
- prepared frame: `prepare_gaia_id_coarse_training_frame(...)`
- split policy:
  - `test_size = 0.30`
  - `random_state = 42`
  - `stratify_columns = ('spec_class', 'evolution_stage')`

Hot baseline для `O` tail:

- `teff_gspphot >= 10000 K`

Это project baseline, а не официальный hard cut Gaia.

## Пакет OTS-C

### OTS-C01. Зафиксировать Contract Для Train Support Review

Что нужно уметь показать:

1. совпадает ли reconstructed split с сохраненным coarse benchmark;
2. сколько true `O` есть в `full/train/test`;
3. как true `O` раскладывается по `evolution_stage` в `train/test`;
4. как выглядит temperature-band support для true `O`;
5. сколько hot `O/B` boundary rows есть в `train/test` до inference.

### OTS-C02. Собрать Typed Review Helper

Файлы:

- `src/exohost/reporting/archive_research/coarse_o_train_support_review.py`
- `tests/archive_research/archived_coarse_o_train_support_review.py`

Что должен уметь helper:

- загружать prepared coarse source через существующий source-layer;
- восстанавливать deterministic split через текущий protocol;
- строить benchmark alignment frame;
- строить `O` support summary по `full/train/test`;
- строить `evolution_stage` support для true `O`;
- строить temperature-band support для true `O`;
- строить hot `O/B` boundary support до inference;
- показывать hottest true `O` preview с train/test membership.

### OTS-C03. Собрать Notebook Review

Файл:

- `analysis/notebooks/archive_research/14_coarse_o_train_support_review.ipynb`

Notebook должен отвечать:

- действительно ли `O` хватает в train и test;
- хватает ли именно hottest `O` tail;
- совпадает ли reconstructed split с benchmark;
- support issue это или уже чисто model-boundary issue.

### OTS-C04. Зафиксировать Round 1 Findings

Файл:

- `docs/methodology/archive_research/coarse_o_train_support_review_round1_ru.md`

После первого review решаем:

- нужен ли новый source-cleaning/hot-tail clean-up;
- нужен ли narrow `O/B` retrain;
- или support уже достаточен, и проблема чисто model-side.

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

- [coarse_o_tail_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_tail_review_round1_ru.md)
- [coarse_o_hot_subset_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_hot_subset_review_round1_ru.md)
- [coarse_ob_boundary_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_ob_boundary_review_round1_ru.md)
- [stabilization_issue_ledger_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/stabilization_issue_ledger_ru.md)
