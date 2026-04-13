# План разбора разделимости признаков на границе O/B

## Зачем проводился этот шаг

Этот документ сохраняет исходный план следующего шага после разбора обучающей
поддержки `O`.

На том этапе требовалось:

- не переобучать coarse-модель вслепую;
- проверить, различимы ли true `O` и true `B` по текущим coarse-признакам;
- посмотреть, как ведет себя текущий coarse artifact на собственном
  обучающем источнике `O/B`.

## Почему понадобился этот разбор

Разбор обучающей поддержки уже показал, что класс `O` не теряется в
обучающем контуре и достаточно представлен в `train` и `test`. Значит
следующий корректный вопрос был уже не о количестве строк, а о разделимости
самих признаков на границе `O/B`.

## На чем строился разбор

Этот шаг работал на обучающем граничном срезе:

- source: prepared coarse training frame
- baseline slice:
  - `spec_class IN ('O', 'B')`
  - `teff_gspphot >= 10000 K`

## Что должен был показать разбор

На этом шаге требовалось:

1. оценить размер обучающего граничного среза `O/B`;
2. посмотреть, как текущий coarse artifact ведет себя на этом срезе;
3. сравнить `P(O)` и `P(B)` на true `O` и true `B`;
4. оценить, какие признаки дают наилучшую разделимость;
5. понять, идет ли речь о слабости признаков или о сдвиге между доменами.

## Почему документ сохранен в архиве

Этот план важен как постановка вопроса о разделимости `O/B` в обучающем
контуре. Позднее он был закрыт фактическим обзором и стал частью исторической
цепочки исследований по редкому хвосту `O`.

Итог этого шага зафиксирован в документе
[coarse_ob_feature_separability_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_ob_feature_separability_review_round1_ru.md).

## Связанные документы

- [coarse_o_train_support_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_train_support_review_round1_ru.md)
- [coarse_ob_boundary_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_ob_boundary_review_round1_ru.md)
- [stabilization_issue_ledger_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/stabilization_issue_ledger_ru.md)
