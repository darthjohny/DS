# ВКР: Приоритизация наблюдений и оценка вероятности наличия экзопланет

Выпускная квалификационная работа для МГТУ им. Н. Э. Баумана.

Тема работы:
`Data Science как инструмент приоритизации наблюдений и оценки вероятности наличия экзопланет на основе параметров звезд. Классификация звезд по спектральным классам и подклассам.`

## Что это за проект

Этот репозиторий содержит исследовательский и прикладной контур проекта,
посвященного анализу звездных данных из `Gaia DR3` и внешних каталогов.

Проект решает две связанные задачи:

- основную прикладную задачу: построение слоя отбора объектов для последующих
  наблюдений, похожих на звезды-хозяева экзопланет;
- поддерживающую исследовательскую задачу: классификацию звезд по
  спектральным классам и подклассам.

Практически проект:

- читает и нормализует астрономические данные;
- сопоставляет внешние метки с `Gaia DR3`;
- обучает и сравнивает модели классификации;
- оценивает качество и надежность наблюдений;
- отделяет рабочую область от сомнительных и внешних объектов;
- формирует итоговый приоритет наблюдений.

## Зачем нужен проект

В задачах поиска экзопланет важна не только формальная классификация звезд, но
и ответ на прикладной вопрос: какие объекты стоит наблюдать в первую очередь.

Проект нужен для того, чтобы:

- перевести большие каталоги звезд в воспроизводимый инженерный контур;
- отделить пригодные объекты от шумных и сомнительных;
- дать интерпретируемое ранжирование целей для последующих наблюдений.

Классификация по спектральным классам и подклассам здесь выступает не самоцелью,
а физическим слоем, который помогает сделать итоговый список кандидатов более
осмысленным.

## Цель и постановка задачи

Цель текущей версии проекта состоит в том, чтобы поверх данных `Gaia DR3` и
внешних каталогов построить воспроизводимый контур, который:

- определяет крупный спектральный класс объекта;
- при возможности уточняет подкласс;
- отделяет `ID / OOD / unknown` объекты;
- применяет `quality_gate`;
- оценивает сходство со звездами-хозяевами экзопланет;
- формирует итоговый `priority_score` и shortlist для наблюдений.

Таким образом, проект отвечает на два вопроса:

1. К какому спектральному классу и подклассу ближе объект.
2. Насколько этот объект интересен как цель для дальнейших наблюдений.

## Главный результат текущей версии

Главный прикладной результат на текущем этапе:

- построен воспроизводимый слой верхнего приоритета для последующих наблюдений;
- верхняя группа включает `72 113` объектов;
- это не подтвержденные планетные системы, а shortlist наиболее интересных
  кандидатов, похожих на звезды-хозяева экзопланет.

Что важно зафиксировать:

- основная прикладная задача текущей версии решена;
- рабочий контур уверенно поднимает наверх спокойные объекты в зоне `F/G/K`,
  похожие на профили звезд-хозяев;
- более глубокая подклассовая детализация остается отдельной точкой роста.

## Обзорная схема системы

![Обзорная схема системы проекта](assets/diagrams/system_overview_ru.svg)

## Как запустить проект за 5 минут

Если нужно быстро поднять проект и проверить рабочий контур, достаточно
следующего маршрута.

### 1. Установить зависимости

```bash
python3.13 -m venv .venv-v2
source .venv-v2/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-v2.txt
python -m pip install -e .
python -m pip install -r requirements-streamlit-v2.txt
```

После `python -m pip install -e .` в окружении появляется CLI-команда
`exohost`.

### 2. Подготовить вход

Базовый боевой вход текущего прикладного контура:

- таблица `lab.gaia_mk_quality_gated` в PostgreSQL.

Если локальной БД нет, проект можно запустить от внешнего `CSV`. Минимально
нужны поля:

- `source_id`, `ra`, `dec`
- `phot_g_mean_mag`, `bp_rp`
- `parallax`, `parallax_over_error`, `ruwe`
- `teff_gspphot`, `logg_gspphot`, `mh_gspphot`
- `radius_gspphot`, `radius_flame`, `lum_flame`, `evolstage_flame`
- `non_single_star`, `classprob_dsc_combmod_star`

Полный контракт внешнего запуска зафиксирован в
[external_decide_input_contract_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/external_decide_input_contract_ru.md).

### 3. Быстро запустить прикладной пайплайн через интерфейс

Самый короткий демонстрационный путь для внешнего `CSV`:

```bash
python -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Дальше в браузере:

- открыть `http://127.0.0.1:8501`;
- перейти на страницу `CSV-запуск`;
- выбрать базовый `run_dir`;
- загрузить входной `CSV`;
- получить новый запуск в `artifacts/decisions/...`.

### 4. Запустить основной пайплайн через CLI

Если нужен CLI-сценарий, базовая форма команды такая:

```bash
exohost decide \
  --input-csv /path/to/input.csv \
  --ood-model-run-dir /path/to/ood_model_run_dir \
  --ood-threshold-run-dir /path/to/ood_threshold_run_dir \
  --coarse-model-run-dir /path/to/coarse_model_run_dir \
  --refinement-model-run-dir /path/to/refinement_family_run_dir \
  --host-model-run-dir /path/to/host_model_run_dir
```

Результат сохраняется в `artifacts/decisions`.

### 5. Если нужен запуск benchmark и обучения

Типовой стартовый пример:

```bash
exohost benchmark --task spectral_class_classification --models hist_gradient_boosting
exohost train --task spectral_class_classification --model hist_gradient_boosting
```

Что важно:

- benchmark-артефакты сохраняются в `artifacts/benchmarks`;
- model artifacts сохраняются в `artifacts/models`;
- эти команды требуют настроенный `.env` и доступ к PostgreSQL.

### 6. Базовые проверки

```bash
.venv-v2/bin/ruff check src tests
.venv-v2/bin/mypy src tests
.venv-v2/bin/pyright src tests
.venv-v2/bin/pytest -q tests
```

Если нужен только быстрый регресс-проход:

```bash
.venv-v2/bin/pytest -q tests/regression
```

## Данные проекта

### Источники данных

Основные источники:

- [Gaia Archive DR3](https://gea.esac.esa.int/archive/)
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)

Дополнительный источник внешних спектральных меток:

- [CDS/VizieR B/mk](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/B/mk?format=html&tex=true)

Роль источников в проекте:

- `Gaia DR3` дает астрометрию, фотометрию, астрофизические параметры и
  сигналы качества;
- `NASA Exoplanet Archive` дает внешний контекст по звездам-хозяевам и
  опорный слой для задачи host-like отбора;
- `B/mk` используется как источник внешних спектральных меток для supervised
  классификации.

### Основные группы признаков

Проект работает со следующими группами полей:

1. Идентификаторы и привязка объектов:
   `source_id`, `ra`, `dec`, `hostname`, `external_catalog_name`,
   `external_object_id`.
2. Фотометрия и цвет:
   `phot_g_mean_mag`, `bp_rp`.
3. Астрометрия и качество измерений:
   `parallax`, `parallax_over_error`, `ruwe`, `non_single_star`,
   `classprob_dsc_combmod_star`.
4. Астрофизические параметры Gaia:
   `teff_gspphot`, `logg_gspphot`, `mh_gspphot`, `radius_gspphot`,
   `radius_flame`, `lum_flame`, `evolstage_flame`.
5. Поля для hot-star и `O/B` анализа:
   `spectraltype_esphs`, `teff_esphs`, `logg_esphs`, `ag_esphs`,
   `azero_esphs`, `ebpminrp_esphs`.
6. Внешние спектральные метки:
   спектральный класс, подкласс, класс светимости и производные признаки
   эволюционной стадии.
7. Данные для host-like задачи:
   кроссматч `Gaia + NASA Exoplanet Archive`, включая таблицы
   `PSCompPars / Stellar Hosts`.

## Модели и логика рабочего контура

В benchmark-слое и в сравнении моделей используются:

- `HGB` — `HistGradientBoostingClassifier`
- `MLP` — `Multi-Layer Perceptron Classifier`
- `GMM` — `Gaussian Mixture Model classifier`

Основной рабочий контур текущей версии в наибольшей степени опирается на
`HistGradientBoostingClassifier`, поскольку он лучше всего показал себя на
табличных данных и в иерархическом сценарии.

В проекте отдельно присутствуют:

- модели coarse-классификации;
- family/refinement-модели для подклассов;
- слой `ID/OOD` и quality-aware маршрутизации;
- слой `quality_gate`;
- модель сходства с популяцией звезд-хозяев;
- слой ранжирования для итогового `priority_score`.

## Что получается на выходе

Итоговый контур формирует:

- `coarse`-класс звезды;
- `refinement`-подкласс;
- `ID / OOD / unknown` routing;
- решение `quality_gate`;
- `host_similarity_score`;
- `observability_score`;
- `priority_score`;
- итоговый список приоритетных целей для последующих наблюдений.

## Структура репозитория

```text
streamlit_app.py          - точка входа демонстрационного Streamlit-интерфейса
requirements-v2.txt       - общий рабочий набор зависимостей
requirements-streamlit-v2.txt - зависимости интерфейсного слоя

src/exohost/
  cli/          - запуск сценариев проекта
  contracts/    - контракты колонок, датасетов и слоев правил
  datasets/     - загрузка и сборка dataframe
  db/           - материализация, SQL и слой таблиц
  evaluation/   - метрики и слой контрольной оценки
  features/     - подготовка признаков и training frames
  ingestion/    - разбор и нормализация внешних меток
  labels/       - логика спектральных меток
  models/       - модели и обертки применения
  posthoc/      - маршрутизация, фильтрация и итоговое решение
  ranking/      - приоритизация наблюдений
  reporting/    - обзорный слой и вспомогательные модули для notebook
  training/     - обучение и контрольные прогоны
  ui/           - интерфейсный слой Streamlit

analysis/notebooks/
  eda/          - обзор данных и обучающих выборок
  research/     - исследовательские разборы
  technical/    - технический обзор работы контура и моделей

assets/
  diagrams/     - обзорные схемы для README и презентации

docs/
  architecture/ - архитектурные и cleanup-планы
  methodology/  - контракты, run review, стабилизация и документы по ВКР

tests/
  unit/         - локальные модульные проверки
  integration/  - короткие сквозные связки между слоями
  smoke/        - быстрые проверки стартового контура
  regression/   - поведенческий регресс `quality_gate`, `priority` и `decide`
```

## Что в репозитории главное

Если проект видит новый человек, удобнее всего идти по таким точкам:

- основной прикладной пайплайн:
  [src/exohost/cli/decide](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/cli/decide),
  [src/exohost/posthoc](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/posthoc)
- обучение и benchmark:
  [src/exohost/cli/train](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/cli/train),
  [src/exohost/cli/benchmark](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/cli/benchmark),
  [src/exohost/training](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/training)
- inference и итоговое решение:
  [src/exohost/posthoc/final_decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/posthoc/final_decision.py),
  [src/exohost/posthoc/final_decision_artifact_runner.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/posthoc/final_decision_artifact_runner.py)
- Streamlit-интерфейс:
  [streamlit_app.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/streamlit_app.py),
  [src/exohost/ui](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/ui)
- тесты:
  [tests](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests),
  [tests/README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/README.md)
- документация:
  [docs](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs),
  [docs/methodology](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology)

## Тестовый контур

Проект использует четыре активных слоя тестирования:

- `unit` — проверяет локальную бизнес-логику, контракты и helper-слой;
- `integration` — проверяет короткие связки между несколькими модулями;
- `smoke` — подтверждает, что пакет, CLI и интерфейс не сломаны на старте;
- `regression` — страхует поведение системы на frozen fixtures.

Если нужно быстро понять тестовую организацию, смотри:

- [tests/README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/README.md)
- [tests/regression/README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/regression/README.md)

## Что посмотреть в репозитории в первую очередь

Если нужно быстро разобраться в проекте, удобно идти в таком порядке:

1. [analysis/notebooks/technical/final_decision_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/final_decision_review.ipynb)
2. [assets/diagrams/system_overview_ru.svg](/Users/evgeniikuznetsov/Desktop/dspro-vkr/assets/diagrams/system_overview_ru.svg)
3. [analysis/notebooks/technical/model_pipeline_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/model_pipeline_review.ipynb)
4. [analysis/notebooks/technical/host_priority_calibration_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/host_priority_calibration_review.ipynb)
5. [analysis/notebooks/research/quality_gate_calibration.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/research/quality_gate_calibration.ipynb)
6. [docs/methodology/README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/README.md)
7. [analysis/notebooks/README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/README.md)
8. [tests/README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/README.md)

## Основная документация и источники

Python и инженерный стек:

- [Python Documentation](https://docs.python.org/3/)
- [typing](https://docs.python.org/3/library/typing.html)
- [pathlib](https://docs.python.org/3/library/pathlib.html)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/current/)
- [SQLAlchemy 2.0](https://docs.sqlalchemy.org/en/20/)
- [pytest](https://docs.pytest.org/en/stable/)
- [Ruff](https://docs.astral.sh/ruff/)
- [mypy](https://mypy.readthedocs.io/en/stable/)
- [Pyright](https://microsoft.github.io/pyright/)
- [Jupyter](https://docs.jupyter.org/en/latest/)
- [nbclient](https://nbclient.readthedocs.io/en/latest/)

Data Science и библиотеки:

- [pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [Decision threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)

Gaia:

- [Gaia Archive](https://gea.esac.esa.int/archive/)
- [Gaia DR3 documentation index](https://gea.esac.esa.int/archive/documentation/GDR3/)
- [Gaia Archive: writing queries](https://www.cosmos.esa.int/web/gaia-users/archive/writing-queries)
- [Gaia Archive Use Cases](https://www.cosmos.esa.int/web/gaia-users/archive/use-cases)
- [Gaia DR3 gaia_source](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html)
- [Gaia DR3 astrophysical_parameters](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_astrophysical_parameter_tables/ssec_dm_astrophysical_parameters.html)
- [Gaia DR3 GSP-Phot](https://gea.esac.esa.int/archive/documentation/GDR3/Data_analysis/chap_cu8par/sec_cu8par_apsis/ssec_cu8par_apsis_gspphot.html)
- [Gaia DR3 Apsis overview](https://gea.esac.esa.int/archive/documentation/GDR3/Data_analysis/chap_cu8par/sec_cu8par_intro/ssec_cu8par_intro_apsis.html)
- [Gaia DR3 astrometric validation](https://gea.esac.esa.int/archive/documentation/GDR3/Catalogue_consolidation/chap_cu9val/sec_cu9val_942/ssec_cu9val_942_astrometry.html)

NASA и внешние каталоги:

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [NASA Exoplanet Archive TAP guide](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html)
- [Planetary Systems Composite Parameters](https://exoplanetarchive.ipac.caltech.edu/docs/pscp_about.html)
- [Stellar Hosts Column Definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_STELLARHOSTS_columns.html)
- [CDS/VizieR B/mk](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/B/mk?format=html&tex=true)

Работы и статьи, на которые проект опирался при анализе:

- [Quality flags for GSP-Phot Gaia DR3 astrophysical parameters with machine learning](https://academic.oup.com/mnras/article-abstract/527/3/7382/7442087)
- [A classifier for spurious astrometric solutions in Gaia EDR3](https://arxiv.org/abs/2101.11641)
- [The Gaia-Kepler-TESS-Host Stellar Properties Catalog](https://arxiv.org/abs/2301.11338)
- [Astrophysical parameters associated to hot stars in Gaia DR3](https://www.aanda.org/articles/aa/full_html/2023/06/aa43709-22/aa43709-22.html)

## Короткий итог

Этот репозиторий — не просто набор notebook и моделей, а оформленная
исследовательская и инженерная платформа для ВКР, которая:

- классифицирует звезды по спектральным классам и подклассам;
- строит контур обработки данных с учетом качества наблюдений;
- формирует список объектов для последующих наблюдений;
- дает интерпретируемый верхний слой кандидатов, похожих на звезды-хозяева.
