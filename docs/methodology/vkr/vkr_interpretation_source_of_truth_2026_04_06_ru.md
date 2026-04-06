# Источник Истины Для Интерпретации Результатов ВКР

Дата фиксации: `2026-04-06`

Связанные документы:

- [vkr_interpretation_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/vkr_interpretation_tz_ru.md)
- [baseline_run_registry_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/baseline_run_registry_ru.md)

## Зачем Нужен Этот Документ

Для текста ВКР нельзя одновременно опираться на все historical run, старые
candidate review и разовые исследования.

Нужен единый источник истины, который фиксирует:

- какой run считаем основным;
- какой run подтверждает воспроизводимость;
- какие notebook и review используем как опору для интерпретации;
- какие документы считаем вспомогательными, но не центральными.

## Основной Run Для Интерпретации

Главным run для текста ВКР считается current active baseline:

- [hierarchical_final_decision_2026_04_05_123111_055017](/Users/evgeniikuznetsov/Desktop/dspro-vkr/artifacts/decisions/hierarchical_final_decision_2026_04_05_123111_055017)

Именно он используется как основной источник для:

- описания итогового pipeline;
- описания `quality_gate` и `priority` после донастройки;
- интерпретации верхнего `high-priority` shortlist;
- всех итоговых чисел, если не оговорено иное.

## Run Для Проверки Доверия И Воспроизводимости

Для подтверждения устойчивости результата используем validation run:

- [hierarchical_final_decision_2026_04_06_095722_391062](/Users/evgeniikuznetsov/Desktop/dspro-vkr/artifacts/decisions/hierarchical_final_decision_2026_04_06_095722_391062)

Его роль:

- не новый baseline;
- не новый scientific result;
- а подтверждение того, что после введения regression-layer боевое поведение
  системы не изменилось.

## Основные Технические Notebook

Как источник истины для технической интерпретации используем:

- [final_decision_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/final_decision_review.ipynb)
- [model_pipeline_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/model_pipeline_review.ipynb)
- [host_priority_calibration_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/host_priority_calibration_review.ipynb)
- [priority_threshold_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/priority_threshold_review.ipynb)

Они нужны для:

- описания поведения pipeline;
- описания model-layer;
- объяснения tuned `priority`;
- связи между артефактами и итоговыми выводами.

## Основные Исследовательские Notebook

Как источник истины для исследовательской интерпретации используем:

- [quality_gate_calibration.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/research/quality_gate_calibration.ipynb)
- [coarse_ob_domain_shift.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/research/coarse_ob_domain_shift.ipynb)
- [secure_o_tail.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/research/secure_o_tail.ipynb)

Они нужны для:

- ограничений и честной трактовки данных;
- объяснения сложной `O/B` boundary;
- описания того, где проект требует дальнейшего развития.

## Основные Review-Документы

Центральные run-review, на которые дальше можно опираться в тексте:

- [pre_battle_policy_candidate_run_2026_04_05_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/pre_battle_policy_candidate_run_2026_04_05_ru.md)
- [high_priority_cohort_review_2026_04_05_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/high_priority_cohort_review_2026_04_05_ru.md)
- [regression_validation_run_2026_04_06_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/regression_validation_run_2026_04_06_ru.md)

Их роль:

- tuned policy;
- главный прикладной результат;
- доверие к воспроизводимости результата.

## Benchmark-Опора Для Метрик

Для интерпретации качества моделей используем benchmark-артефакты из
[baseline_run_registry_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/baseline_run_registry_ru.md):

- `id_ood`
- `coarse`
- `refinement_flat`
- `host_field`

Важно:

- benchmark-метрики нужны для описания состояния моделей;
- live final-decision run нужен для описания поведения всей системы;
- эти два слоя нельзя смешивать без явного пояснения.

## Что Считаем Вспомогательным, Но Не Центральным

Для текста ВКР не считаем центральным источником:

- historical run до tuned baseline;
- архивные deep-dive расследования;
- одноразовые candidate review, если они уже не задают active policy;
- старые notebook из archive-research.

Их можно использовать как фон, но не как основу итоговой интерпретации.

## Правило Для Следующих Шагов

Во всех последующих шагах интерпретационного пакета:

- если нужен главный результат, берем active baseline;
- если нужен аргумент о доверии и устойчивости, берем validation run;
- если нужен технический разбор, опираемся на technical notebook;
- если нужен блок ограничений, опираемся на research notebook и специальные
  review по `O/B`.

## Вывод

Источник истины для интерпретации ВКР теперь зафиксирован.

Это позволяет дальше писать не “по памяти” и не по случайным historical
артефактам, а по одному согласованному набору run, notebook и review-docs.
