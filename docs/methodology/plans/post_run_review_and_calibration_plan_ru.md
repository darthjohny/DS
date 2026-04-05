# Post-Run Review And Calibration Plan

## Цель

Этот документ фиксирует следующий этап после первого end-to-end прогона:

- довести review-notebooks до уровня, где они отвечают на главный прикладной вопрос;
- отдельно провести calibration-study для `quality_gate`;
- отдельно спроектировать clean `host/priority` integration;
- только после этого делать второй full run и сравнение с первым.

Детальное исполнимое разбиение `QG-*` и `HP-*` живет отдельно в
[quality_gate_host_priority_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/quality_gate_host_priority_tz_ru.md).

## Текущее Состояние

После первого artifact-backed прогона уже есть:

- `OOD -> coarse -> refinement -> final decision` routing;
- end-to-end final decision artifacts;
- notebooks для model pipeline и final decision review.

Но пока есть три содержательных пробела:

1. notebooks показывают stage-level observability, но еще не дают удобный star-level answer;
2. большая доля `unknown` пока не разобрана как отдельный calibration-case;
3. `priority` и `host-likelihood` пока не подключены clean way, потому что legacy host-model
   живет на старом радиусном контракте.

## Инженерный Инвариант

Для этого этапа сохраняются те же правила:

- `1 файл = 1 ответственность`;
- без монолитных notebook-helper модулей;
- `PEP 8`;
- явная типизация;
- простая логика раньше сложной;
- без ad hoc compatibility magic;
- после каждого небольшого куска:
  - micro-QA;
  - `ruff`;
  - точечный `mypy/pyright`;
  - целевые тесты;
- после завершения микро-ТЗ:
  - scoped big-QA только по новому слою.

## Official Опора

### Scikit-learn Calibration And Thresholding

- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
- [Tuning the decision threshold for class prediction](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [balanced_accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)

### Gaia DR3 Quality And Source Semantics

- [Gaia DR3 documentation index](https://gea.esac.esa.int/archive/documentation/GDR3/)
- [Gaia DR3 contents summary](https://www.cosmos.esa.int/web/gaia/dr3)
- [Gaia DR3 astrophysical_parameters datamodel](https://gaia.aip.de/metadata/gaiadr3/astrophysical_parameters/)
- [Gaia DR3 Apsis III: non-stellar content and source classification](https://www.aanda.org/articles/aa/full_html/2023/06/aa43423-22/aa43423-22.html)

### Exoplanet Host Data Source

- [NASA Exoplanet Archive](https://exoplanet.ipac.caltech.edu/)
- [NASA Exoplanet Archive TAP guide](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html)
- [IPAC DOI page for archive tables](https://www.ipac.caltech.edu/dois/exoplanet-archive)

## Аналогичные И Полезные Работы Для Разбора

### Quality / Reliability / Selection

- [Quality flags for GSP-Phot Gaia DR3 astrophysical parameters with machine learning](https://academic.oup.com/mnras/article-abstract/527/3/7382/7442087)
- [A classifier for spurious astrometric solutions in Gaia EDR3](https://arxiv.org/abs/2101.11641)

### Selective / Reject Classification

- [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500)
- [SelectiveNet: A Deep Neural Network with an Integrated Reject Option](https://arxiv.org/abs/1901.09192)

### Host / Planet-Centric Stellar Catalogues

- [The Gaia-Kepler-TESS-Host Stellar Properties Catalog](https://arxiv.org/abs/2301.11338)

## Следующий Пакет Работ

### MTZ-N1. Notebook Review Uplift

Цель:

- сделать notebooks useful не только для pipeline metrics, но и для star-level interpretation.

Что добавляем:

- в model notebook:
  - compact stage summary;
  - model artifact summary;
  - threshold artifact summary;
  - compact host-source context, без смешения с final physics review;
  - явные короткие выводы;
- в final decision notebook:
  - star-level preview table по individual stars;
  - поля:
    - `source_id`
    - `ra`
    - `dec`
    - `final_domain_state`
    - `final_quality_state`
    - `final_coarse_class`
    - `final_refinement_label`
    - `final_decision_reason`
    - `priority_score`
    - `priority_label`
    - `priority_reason`
    - `mh_gspphot`
    - `parallax`
    - `parallax_over_error`
    - `ruwe`
    - `phot_g_mean_mag`
    - `radius_flame`
    - `quality_reason`
    - `review_bucket`
- если `priority` еще не подключен:
  - notebook должен явно писать, что `priority` и `host-likelihood` пока не доступны в clean contract.

### MTZ-N2. Quality Gate Calibration Study

Цель:

- понять, является ли большая доля `unknown` логичной и научно корректной, или gate слишком агрессивен.

Что делаем:

1. Разбиваем `unknown` и `reject` на reason buckets.
2. Разделяем:
   - official Gaia field semantics;
   - project policy thresholds.
3. Проверяем влияние каждого project-threshold по отдельности:
   - `ruwe`
   - `parallax_over_error`
   - `classprob_dsc_combmod_star`
   - отсутствие `radius_flame`
   - `non_single_star`
4. Строим coverage/risk analysis для разных gate variants.
5. Отдельно фиксируем conservative baseline hypothesis:
   - начинаем с общепринятых astrophysical quality checks;
   - `RUWE <= 1.4` используем как первую project-policy гипотезу для calibration-study;
   - не считаем этот порог official hard Gaia cut, пока не проверим его влияние на
     coverage и review pool.
6. Если большая доля `unknown` оказывается логичной:
   - оставляем ее как есть;
   - сохраняем в отдельную таблицу как review pool;
   - трактуем это как полезный результат, а не как failure.

Текущее решение после первого calibration-study:

- baseline gate policy первой волны оставлена без изменений;
- `unknown/review` признан валидным selective outcome;
- следующий открытый пакет работ смещается на clean `host/priority` integration.

### MTZ-N3. Clean Host / Priority Integration Design

Цель:

- подключить `host-likelihood` и `priority` без смешения legacy и нового контракта.

Что делаем:

1. Фиксируем target semantics:
   - это `confirmed_host prior`, а не "вероятность существования планеты вообще".
2. Сверяемся с NASA Exoplanet Archive table semantics:
   - `stellarhosts`
   - `ps`
   - `pscomppars`
3. Фиксируем новый feature contract:
   - без silent fallback на legacy-only поля;
   - с явным отношением к `radius_flame / radius_feature`.
4. Решаем, что чище:
   - retrain host-model на новом contract;
   - или ввести explicit compatibility adapter как временный слой.
5. Только после этого возвращаем `priority` в final decision pipeline.

Текущее состояние:

- `HP-01` host target semantics зафиксирован;
- `HP-02` clean host feature contract зафиксирован;
- `HP-03` host-source review layer собран;
- `HP-04` integration path зафиксирован;
- отдельный host enrichment design зафиксирован в
  [host_enrichment_design_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/host_enrichment_design_ru.md);
- следующий открытый шаг: materialize host enrichment, потом открыть `HP-05`,
  то есть вернуть priority уже после host enrichment и retrain.

### MTZ-N4. Second End-To-End Review Run

Цель:

- после `N1-N3` сделать второй честный прогон и сравнить его с первым.

Что сравниваем:

- долю `id / candidate_ood / ood / unknown`;
- распределение `final_coarse_class`;
- распределение `final_refinement_label`;
- долю и причины `unknown`;
- наличие `priority` и `host` output;
- explainability output в notebooks.

## Критерий Готовности Следующего Этапа

Следующий этап считается закрытым, когда:

- notebooks отвечают на прикладной вопрос по отдельным звездам;
- `unknown` объяснен и количественно разобран;
- `host/priority` спроектирован clean way;
- второй end-to-end run сравним с первым и интерпретируем.
