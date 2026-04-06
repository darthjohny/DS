# ТЗ На Интерпретацию Результатов Для ВКР

Дата фиксации: `2026-04-06`

Связанные документы:

- [baseline_run_registry_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/baseline_run_registry_ru.md)
- [high_priority_cohort_review_2026_04_05_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/high_priority_cohort_review_2026_04_05_ru.md)
- [pre_battle_policy_candidate_run_2026_04_05_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/pre_battle_policy_candidate_run_2026_04_05_ru.md)
- [regression_validation_run_2026_04_06_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/regression_validation_run_2026_04_06_ru.md)
- [analysis/notebooks/technical/final_decision_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/final_decision_review.ipynb)
- [analysis/notebooks/technical/model_pipeline_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/model_pipeline_review.ipynb)

## Зачем Нужен Этот Пакет

На текущем этапе у проекта уже есть:

- рабочий и воспроизводимый pipeline;
- active baseline run;
- интерпретируемый верхний shortlist для follow-up наблюдений;
- технические notebook и run-review по качеству системы;
- отдельный регресс-слой, который подтверждает стабильность боевого контура.

Но для ВКР этого недостаточно в прямом виде.

Сейчас результат разложен по notebook, review-документам, tuning-пакетам и
техническим комментариям. Для текста ВКР нужно собрать из этого не новый
эксперимент, а понятную научно-инженерную интерпретацию:

- что именно построено;
- что получилось на практике;
- почему этому результату можно доверять;
- где честные ограничения;
- что является следующим шагом развития.

## Цель

Подготовить отдельный интерпретационный слой для ВКР, который:

- опирается только на уже подтвержденные артефакты и review;
- переводит технические результаты проекта в понятный текст;
- отделяет главный прикладной результат от вторичных расследований;
- честно фиксирует ограничения и дальнейшее развитие;
- не смешивает интерпретацию, новые вычисления и подготовку презентации.

## Что Считаем Главным Результатом Работы

Главный прикладной результат проекта в текущей версии:

- построен воспроизводимый pipeline классификации и приоритизации;
- сформирован stable shortlist из `72 113` объектов верхнего приоритета;
- этот shortlist не трактуется как найденные планетные системы;
- он трактуется как набор host-like кандидатов для дальнейших наблюдений;
- shortlist физически и operationally правдоподобен:
  - в нем доминируют `F/G/K` объекты;
  - он держится после tuning `quality_gate + priority`;
  - он воспроизводится на validation run после введения regression-слоя.

Именно это и должно стать центром интерпретации для ВКР.

## Что Должно Получиться После Пакета

После завершения этого пакета должны появиться три основные текстовые опоры:

### 1. Главный результат работы

Документ с ответом на вопросы:

- что построено;
- как работает pipeline на верхнем уровне;
- что означает top shortlist;
- почему `72 113` объектов это осмысленный результат, а не просто output модели.

### 2. Интерпретация качества системы

Документ с ответом на вопросы:

- какие модели в проекте сильные;
- какие слои ограничивают результат;
- как читать метрики `id_ood`, `coarse`, `refinement`, `host_field`;
- почему tuned `quality_gate` и `priority` стали ключевыми policy-слоями;
- почему validation run после regression-слоя важен для доверия к результату.

### 3. Ограничения и развитие

Документ с ответом на вопросы:

- что пока не решено полностью;
- где остаются ограничения данных и модели;
- почему `O/B` boundary была сложной областью;
- что можно делать дальше:
  - spectroscopic crossmatch;
  - refinement hot-star слоя;
  - расширение host-like признаков;
  - более сильный исследовательский контур для rare-tail классов.

## Что Не Входит В Этот Пакет

В этот пакет не входит:

- подготовка слайдов и презентации;
- написание всей ВКР целиком;
- новые эксперименты и новый tuning;
- переделка active baseline без подтвержденной необходимости;
- расширение notebook только ради текста.

Сначала нужен интерпретационный каркас, а уже потом поверх него можно будет
готовить презентацию и финальный текст.

## Источники И Опора Для Интерпретации

Интерпретация должна опираться только на уже зафиксированные источники:

### Внутренние

- active baseline run и validation run;
- technical notebook;
- research notebook;
- run-review документы;
- benchmark artifacts;
- текущий README проекта.

### Внешние

- [Python Documentation](https://docs.python.org/3/)
- [pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Gaia DR3 documentation index](https://gea.esac.esa.int/archive/documentation/GDR3/)
- [Gaia DR3 gaia_source](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html)
- [Gaia DR3 astrophysical_parameters](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_astrophysical_parameter_tables/ssec_dm_astrophysical_parameters.html)
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)

## Инженерный Инвариант

Для каждого шага этого пакета:

- не делаем новых вычислений без необходимости;
- не пересказываем notebook дословно;
- работаем только с подтвержденными результатами;
- держим комментарии и документы на русском;
- формулируем выводы коротко, ясно и без маркетинговой риторики;
- различаем:
  - результат;
  - интерпретацию;
  - ограничение;
  - следующий шаг.

## Предлагаемый Порядок Работы

1. Зафиксировать источник истины для интерпретации.
2. Сформировать каркас главного результата.
3. Разобрать качество системы и доверие к результату.
4. Сформулировать ограничения и развитие.
5. Сверить, что все это согласовано с notebook и run-review.

## Критерий Готовности

Пакет считается завершенным, когда:

- у ВКР есть отдельный интерпретационный каркас;
- главный результат изложен без технической каши;
- качество системы описано честно и понятно;
- ограничения и дальнейшее развитие сформулированы отдельно;
- выводы не противоречат active baseline, review-docs и notebook.
