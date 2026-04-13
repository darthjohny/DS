# Источник истины для интерпретации результатов ВКР

Дата фиксации: `2026-04-06`

Связанные документы:

- [baseline_run_registry_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/baseline_run_registry_ru.md)

## Зачем нужен этот документ

Для текста ВКР нельзя одновременно опираться на все исторические прогоны,
старые документы обзора и разовые исследования.

Нужен единый источник истины, который фиксирует:

- какой прогон считаем основным;
- какой прогон подтверждает воспроизводимость;
- какие ноутбуки и документы обзора используем как опору для интерпретации;
- какие документы считаем вспомогательными, но не центральными.

## Основной прогон для интерпретации

Главным прогоном для текста ВКР считается текущий базовый прогон:

- [hierarchical_final_decision_2026_04_05_123111_055017](/Users/evgeniikuznetsov/Desktop/dspro-vkr/artifacts/decisions/hierarchical_final_decision_2026_04_05_123111_055017)

Именно он используется как основной источник для:

- описания итогового контура;
- описания `quality_gate` и `priority` после донастройки;
- интерпретации верхнего списка целей с высоким приоритетом;
- всех итоговых чисел, если не оговорено иное.

## Прогон для проверки доверия и воспроизводимости

Для подтверждения устойчивости результата используем проверочный прогон:

- [hierarchical_final_decision_2026_04_06_095722_391062](/Users/evgeniikuznetsov/Desktop/dspro-vkr/artifacts/decisions/hierarchical_final_decision_2026_04_06_095722_391062)

Его роль:

- не новый базовый прогон;
- не новый научный результат;
- а подтверждение того, что после введения слоя регрессионных тестов рабочее
  поведение системы не изменилось.

## Основные технические ноутбуки

Как источник истины для технической интерпретации используем:

- [final_decision_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/final_decision_review.ipynb)
- [model_pipeline_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/model_pipeline_review.ipynb)
- [host_priority_calibration_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/host_priority_calibration_review.ipynb)
- [priority_threshold_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/priority_threshold_review.ipynb)

Они нужны для:

- описания поведения контура;
- описания слоя моделей;
- объяснения настроенного `priority`;
- связи между артефактами и итоговыми выводами.

## Основные исследовательские ноутбуки

Как источник истины для исследовательской интерпретации используем:

- [quality_gate_calibration.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/research/quality_gate_calibration.ipynb)
- [coarse_ob_domain_shift.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/research/coarse_ob_domain_shift.ipynb)
- [secure_o_tail.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/research/secure_o_tail.ipynb)

Они нужны для:

- ограничений и честной трактовки данных;
- объяснения сложной границы `O/B`;
- описания того, где проект требует дальнейшего развития.

## Основные документы обзора

Центральные документы обзора, на которые дальше можно опираться в тексте:

- [pre_battle_policy_candidate_run_2026_04_05_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/pre_battle_policy_candidate_run_2026_04_05_ru.md)
- [high_priority_cohort_review_2026_04_05_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/high_priority_cohort_review_2026_04_05_ru.md)
- [regression_validation_run_2026_04_06_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/regression_validation_run_2026_04_06_ru.md)

Их роль:

- настроенные правила;
- главный прикладной результат;
- доверие к воспроизводимости результата.

## Benchmark-опора для метрик

Для интерпретации качества моделей используем benchmark-артефакты из
[baseline_run_registry_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/baseline_run_registry_ru.md):

- `id_ood`
- `coarse`
- `refinement_flat`
- `host_field`

Важно:

- benchmark-метрики нужны для описания состояния моделей;
- итоговый рабочий прогон нужен для описания поведения всей системы;
- эти два слоя нельзя смешивать без явного пояснения.

## Что считаем вспомогательным, но не центральным

Для текста ВКР не считаем центральным источником:

- исторические прогоны до текущего базового варианта;
- архивные глубокие разборы;
- одноразовые документы обзора, если они уже не задают действующие правила;
- старые ноутбуки из `archive_research`.

Их можно использовать как фон, но не как основу итоговой интерпретации.

## Правило для следующих шагов

Во всех последующих шагах интерпретационного пакета:

- если нужен главный результат, берем текущий базовый прогон;
- если нужен аргумент о доверии и устойчивости, берем проверочный прогон;
- если нужен технический разбор, опираемся на технические ноутбуки;
- если нужен блок ограничений, опираемся на исследовательские ноутбуки и
  специальные обзоры по `O/B`.

## Вывод

Источник истины для интерпретации ВКР теперь зафиксирован.

Это позволяет дальше писать не “по памяти” и не по случайным историческим
артефактам, а по одному согласованному набору прогонов, ноутбуков и документов
обзора.
