# План разбора редкого хвоста O-класса

## Зачем проводился этот шаг

Этот документ сохраняет исходный план первого предметного разбора редкого
класса `O` в coarse-контуре.

На старте нужно было понять:

- что именно происходит с true `O` после `quality_gate`;
- как выглядит часть выборки со статусом `quality_state = 'pass'`;
- есть ли основания считать, что проблема сидит только в coarse-модели.

## На чем строился разбор

- coarse model artifact:
  `artifacts/models/gaia_id_coarse_classification__hist_gradient_boosting__2026_03_28_215003_509969`
- final decision run:
  `artifacts/decisions/hierarchical_final_decision_2026_03_29_111132_270743`
- review notebook:
  [11_coarse_o_tail_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/archive_research/11_coarse_o_tail_review.ipynb)

## Что должен был показать разбор

На этом этапе требовалось:

1. отделить влияние `quality_gate` от поведения coarse-модели;
2. посмотреть, как coarse-модель ведет себя на части выборки со статусом
   `pass`;
3. проверить, согласуется ли физика surviving `O` с горячими `O/B`-звездами;
4. решить, есть ли основания сразу говорить о переобучении или ребалансировке.

## Почему документ сохранен в архиве

Этот план важен как первая постановка проблемы. Позднее стало ясно, что
основной вопрос связан не со всем пулом `O` целиком, а с более узким горячим
подмножеством и границей `O/B`.

Итоговый разбор этого шага зафиксирован в документе
[coarse_o_tail_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_tail_review_round1_ru.md).

## Связанные документы

- [coarse_o_tail_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_tail_review_round1_ru.md)
- [coarse_o_hot_subset_review_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_hot_subset_review_tz_ru.md)
- [stabilization_issue_ledger_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/stabilization_issue_ledger_ru.md)
