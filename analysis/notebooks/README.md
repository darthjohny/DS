# Слой notebook проекта

Активные notebook разложены по трем рабочим папкам:

- `eda` — обзор данных и обучающих выборок;
- `research` — исследовательские углубленные разборы;
- `technical` — технический обзор пайплайна, моделей и калибровок.

Отдельно живет архив глубоких расследований:

- [archive_research/README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/archive_research/README.md)

## EDA

EDA-ноутбуки отвечают на вопрос, с какими данными мы вообще работаем.

- [router_training.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/eda/router_training.ipynb)
  - обзор обучающей выборки router-ветки;
  - пропуски, распределения признаков и баланс coarse-меток.
- [host_training.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/eda/host_training.ipynb)
  - обзор обучающей выборки host-ветки;
  - покрытие признаков и баланс по классам и стадиям.
- [label_coverage.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/eda/label_coverage.ipynb)
  - покрытие меток перед subclass-волной;
  - разбор `spec_class`, `evolution_stage` и `spec_subclass`.

## Research

Исследовательский слой нужен для научных гипотез, ограничений данных и
содержательных выводов.

- [quality_gate_calibration.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/research/quality_gate_calibration.ipynb)
  - разбор `quality_gate`, состояний `quality/ood`, причин и `review_bucket`;
  - сравнение базового, смягченного и строгого вариантов правил.
- [coarse_ob_domain_shift.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/research/coarse_ob_domain_shift.ipynb)
  - исследование доменного сдвига на границе `O/B`;
  - сравнение обучающего и рабочего доменов, физических признаков и вероятностей.
- [secure_o_tail.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/research/secure_o_tail.ipynb)
  - разбор небольшой надежной группы объектов класса `O`;
  - сравнение с эталонными наборами `O` и `B`.

## Technical

Технический слой показывает инженерное состояние пайплайна и моделей.

- [scoring_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/scoring_review.ipynb)
  - обзор scoring- и ranking-артефактов.
- [model_pipeline_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/model_pipeline_review.ipynb)
  - состояние модельного контура по этапам;
  - benchmark, runtime и целевые балансы.
- [final_decision_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/final_decision_review.ipynb)
  - итог полного прогона `final decision`;
  - распределения `domain/quality/refinement`, `priority` и итоговые таблицы по объектам.
- [host_priority_calibration_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/host_priority_calibration_review.ipynb)
  - калибровка `host_similarity_score`;
  - `brier_score`, `log_loss`, `roc_auc`, кривая надежности и корзины вероятности.
- [priority_threshold_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/priority_threshold_review.ipynb)
  - работа порогов поверх `priority_score`;
  - переходы между состояниями и влияние порогов по классам.

## Архив

Архивные notebook не входят в обычный рабочий цикл и не участвуют в активном
QA по умолчанию.

Сейчас в архиве лежат:

- `11_coarse_o_tail_review.ipynb`
- `12_coarse_o_hot_subset_review.ipynb`
- `13_coarse_ob_boundary_review.ipynb`
- `14_coarse_o_train_support_review.ipynb`
- `15_coarse_ob_feature_separability_review.ipynb`

## Общие Правила

- активный notebook отвечает за обзор, визуализацию и выводы, а не за тяжелую прикладную логику;
- повторяемую логику сначала выносим в `src/exohost/reporting`;
- пользовательский вывод в active notebook держим на русском;
- имена active notebook не нумеруем;
- `eda`, `research` и `technical` не смешиваем в одной папке.

## QA

- `pytest` проверяет smoke-уровень:
  - JSON-валидность;
  - непустую структуру;
  - синтаксическую корректность code-cell.
- `nbclient` запускается адресно для активных notebook, затронутых текущим шагом.

Практически это означает:

- быстрый `pytest` страхует структуру notebook;
- `nbclient` используется адресно для тех активных notebook, которые были
  изменены.
