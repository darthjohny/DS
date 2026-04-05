# Quality Gate And Host Priority Execution TZ

## Цель

Этот документ фиксирует следующий исполнимый пакет работ после первого
`end-to-end` прогона.

Наша цель сейчас:

- не трогать без необходимости `coarse` и `refinement` модели;
- сначала понять, корректен ли текущий `quality_gate`;
- только потом clean way вернуть `host/priority` в итоговый pipeline.

## Инженерный Инвариант

Для этого этапа сохраняется тот же рабочий стандарт:

- `1 файл = 1 ответственность`;
- без монолитных helper-модулей и notebook-магии;
- `PEP 8`;
- явная типизация;
- простые решения раньше сложных;
- без скрытых fallback и неявных legacy-adapter;
- комментарии только там, где они реально помогают читать код;
- после каждого небольшого куска:
  - micro-QA;
  - `ruff`;
  - точечный `mypy/pyright`;
  - целевые тесты;
- после завершения блока:
  - scoped big-QA только по затронутому слою.

## Official Опора

### Gaia DR3

- [Gaia DR3 gaia_source datamodel](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html)
- [Gaia DR3 astrophysical_parameters datamodel](https://gaia.aip.de/metadata/gaiadr3/astrophysical_parameters/)
- [Gaia DR3 astrometric validation](https://gea.esac.esa.int/archive/documentation/GDR3/Catalogue_consolidation/chap_cu9val/sec_cu9val_942/ssec_cu9val_942_astrometry.html)
- [Gaia DR3 Apsis III](https://www.aanda.org/articles/aa/full_html/2023/06/aa43423-22/aa43423-22.html)

### Scikit-Learn

- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [Tuning the decision threshold for class prediction](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [balanced_accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)

### NASA Exoplanet Archive

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [TAP guide](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html)
- [Planetary Systems Composite Parameters](https://exoplanetarchive.ipac.caltech.edu/docs/pscp_about.html)

## Аналогичные И Полезные Работы

### Quality / Reliability

- [Quality flags for GSP-Phot Gaia DR3 astrophysical parameters with machine learning](https://academic.oup.com/mnras/article-abstract/527/3/7382/7442087)
- [A classifier for spurious astrometric solutions in Gaia EDR3](https://arxiv.org/abs/2101.11641)

### Selective / Reject Classification

- [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500)
- [SelectiveNet](https://arxiv.org/abs/1901.09192)

### Host / Stellar Catalog Integration

- [The Gaia-Kepler-TESS-Host Stellar Properties Catalog](https://arxiv.org/abs/2301.11338)

## Пакет QG: Quality Gate

### QG-01. Зафиксировать Calibration Contract

Цель:

- разделить official semantics полей и project thresholds.

Что фиксируем:

- `ruwe`
- `parallax_over_error`
- `non_single_star`
- `classprob_dsc_combmod_star`
- `has_core_features`
- `has_flame_features`
- `missing_core_features`
- `missing_radius_flame`

Важная policy-гипотеза первой волны:

- `RUWE <= 1.4` используем как conservative project baseline для calibration-study;
- не считаем этот порог official hard Gaia cut до отдельного разбора coverage/risk.

Результат:

- отдельный calibration contract;
- explicit список signals и baseline thresholds.

### QG-02. Собрать Узкий Audit Source

Цель:

- не калибровать gate на сырых notebook-join;
- иметь устойчивый typed source для review.

Что делаем:

- loader для `lab.gaia_mk_quality_gated`;
- loader для `lab.gaia_mk_unknown_review`;
- явный contract для колонок audit-layer.

Результат:

- typed quality-gate audit source, пригодный для review и notebook-study.

Статус:

- закрыто в коде.

Файлы:

- `src/exohost/contracts/quality_gate_dataset_contracts.py`
- `src/exohost/datasets/load_quality_gate_audit_dataset.py`

### QG-03. Собрать Review Helper Layer

Цель:

- вынести repeatable quality-gate review из notebook в тестируемый helper-модуль.

Что делаем:

- distributions по `quality_state`, `ood_state`, `quality_reason`, `review_bucket`;
- rule-by-rule summary по boolean gate signals;
- coverage summary для review/reject buckets.

Результат:

- reusable review helper для calibration-study.

Статус:

- закрыто в коде.

Файлы:

- `src/exohost/reporting/quality_gate_review.py`
- `tests/unit/test_load_quality_gate_audit_dataset.py`
- `tests/unit/test_quality_gate_review.py`

### QG-04. Собрать Calibration Notebook

Цель:

- понять, оправдана ли текущая доля `unknown/reject`.

Что делаем:

- отдельный notebook под quality-gate calibration-study;
- сравниваем rule families и coverage;
- не смешиваем его с final decision review.

Результат:

- объяснимый ответ:
  - gate корректно осторожный;
  - или отдельные thresholds слишком агрессивны.

Статус:

- закрыто в коде и notebook-слое.

Файлы:

- `src/exohost/reporting/quality_gate_calibration_review.py`
- `tests/unit/test_quality_gate_calibration_review.py`
- `analysis/notebooks/research/quality_gate_calibration.ipynb`

### QG-05. Принять Policy Decision

Цель:

- решить, что меняем в `quality_gate`, а что оставляем.

Если большая доля `unknown` оказывается логичной:

- ничего не “улучшаем силой”;
- сохраняем review-pool как отдельный аналитический контур;
- трактуем это как корректный selective behavior.

Статус:

- закрыто после первого calibration-study.

Решение первой волны:

- production baseline gate policy оставляем без изменений;
- `unknown/review` трактуем как корректный selective outcome, а не как failure моделей;
- `lab.gaia_mk_unknown_review` продолжаем использовать как отдельный review-pool;
- реальных изменений в gate-code на этом этапе не вносим.

Live-основание решения:

- baseline:
  - `pass`: `178439` (`44.36%`)
  - `unknown`: `63823` (`15.87%`)
  - `reject`: `159964` (`39.77%`)
- relaxed:
  - `pass`: `195815` (`48.68%`)
  - `unknown`: `46447` (`11.55%`)
  - `reject`: `159964`
- strict:
  - `pass`: `162741` (`40.46%`)
  - `unknown`: `79521` (`19.77%`)
  - `reject`: `159964`

Ключевой вывод:

- главный driver для `reject` сейчас это missing core features, а не `RUWE`;
- relaxed variant возвращает в `pass` только `17376` строк;
- это ограниченный выигрыш, который не оправдывает ослабление gate до отдельной host/priority wave;
- основные `unknown` reasons:
  - `high_ruwe`
  - `missing_radius_flame`
  - `low_parallax_snr`

Итог:

- baseline thresholds сохраняем;
- review-pool не пытаемся протолкнуть в normal `id` pipeline силой;
- следующий открытый пакет работ: `HP-01 ... HP-05`.

## Пакет HP: Host / Priority

### HP-01. Зафиксировать Host Target Semantics

Цель:

- определить, что именно предсказывает host-layer.

Принцип:

- не называем это “вероятностью существования планеты вообще”;
- используем более честную постановку:
  - `host-likeness`
  - или `confirmed-host prior`
  - относительно нашего host source.

Статус:

- закрыто в docs.

Файлы:

- `docs/methodology/host_target_semantics_ru.md`

### HP-02. Зафиксировать Clean Feature Contract

Цель:

- убрать неявный legacy-контракт вокруг радиусов и старых host-признаков.

Что фиксируем:

- какие поля Gaia/NASA входят в новый host layer;
- что считается canonical radius;
- как обрабатываются пропуски;
- где проходит граница между host feature engineering и ranking.

Статус:

- закрыто в docs.

Файлы:

- `docs/methodology/host_priority_feature_contract_ru.md`

### HP-03. Собрать Host Source Review Layer

Цель:

- до модели проверить coverage и structure текущего host source.

Что делаем:

- class/stage balance;
- feature coverage;
- coverage по `radius_flame`, metallicity, parallax, RUWE;
- relation между host source и новым final-decision contract.

Статус:

- закрыто в коде.

Файлы:

- `src/exohost/contracts/host_priority_feature_contracts.py`
- `src/exohost/reporting/host_priority_review.py`
- `tests/unit/test_host_priority_review.py`

### HP-04. Спроектировать Integration Path

Цель:

- решить, что чище:
  - retrain host model на новом contract;
  - или ввести explicit temporary adapter.

Принцип:

- без silent fallback;
- без смешения legacy и нового clean contract.

Статус:

- закрыто в docs.

Решение:

- mainline path = `clean retrain after host enrichment`;
- temporary compatibility adapter не используем как основной путь.

Файлы:

- `docs/methodology/host_priority_integration_path_ru.md`

### HP-05. Вернуть Priority В Final Pipeline

Цель:

- только после `HP-01 ... HP-04` снова подключить `priority_score`, `priority_label`
  и `priority_reason` в `end-to-end` run.

Следующий обязательный подшаг перед реализацией:

- спроектировать и материализовать clean host enrichment relation с `radius_flame`.

Design-опора:

- [host_enrichment_design_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/host_enrichment_design_ru.md)

Текущее состояние:

- прямой переход к `HP-05` еще рано;
- сначала нужен host enrichment из official Gaia DR3 `astrophysical_parameters`.

Порядок:

1. materialize `lab.nasa_gaia_host_flame_enrichment_source`;
2. выгрузить `source_id` в Gaia;
3. собрать `public.nasa_gaia_host_flame_enrichment_raw`;
4. собрать `public.nasa_gaia_host_flame_enrichment_clean`;
5. собрать `lab.nasa_gaia_host_training_enriched`;
6. переснять host review;
7. только потом открывать retrain и возврат `priority` в mainline.

Live status:

- шаги `1-6` уже закрыты;
- clean host retrain и возврат `priority` в mainline уже закрыты;
- следующий открытый пакет сместился в probability calibration.

Связанный следующий документ:

- [host_priority_calibration_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/host_priority_calibration_tz_ru.md)

Смысл следующего пакета:

- не калибровать whole `priority` formula вслепую;
- сначала разобрать и при необходимости откалибровать `host_similarity_score`;
- только потом менять scaling или thresholds в ranking-layer.

## Порядок Выполнения

1. `QG-01`
2. `QG-02`
3. `QG-03`
4. `QG-04`
5. `QG-05`
6. `HP-01`
7. `HP-02`
8. `HP-03`
9. `HP-04`
10. `HP-05`

## Критерий Готовности Этого Пакета

Этот пакет считается закрытым, когда:

- `quality_gate` объяснен и калибруется не “на глаз”;
- `unknown/review` логически интерпретируем;
- host-layer спроектирован clean way;
- второй `end-to-end` run можно делать уже на понятном contract.
