# План разбора обучающей поддержки O-класса

## Зачем проводился этот шаг

Этот документ сохраняет исходный план следующего шага после первого разбора
границы `O/B`.

На том этапе требовалось:

- не переобучать coarse-модель вслепую;
- проверить, хватает ли true `O` в самом обучающем источнике;
- восстановить тот же `train/test split`, который использовался в benchmark;
- понять, не теряется ли редкий класс `O` еще до этапа предсказания.

## Почему понадобился этот разбор

Разбор границы `O/B` уже показал, что true `O` на горячем проходящем срезе
системно уходит в `B`. Но перед любыми выводами о модели нужно было ответить
на более базовый вопрос: достаточно ли класс `O` представлен в обучающем
источнике и в разбиении `train/test`.

## На чем строился разбор

Этот шаг не вводил новый источник данных. Он использовал существующий
обучающий контракт:

- source: `lab.v_gaia_id_coarse_training`
- prepared frame: `prepare_gaia_id_coarse_training_frame(...)`
- split policy:
  - `test_size = 0.30`
  - `random_state = 42`
  - `stratify_columns = ('spec_class', 'evolution_stage')`

Дополнительно для горячего хвоста использовался рабочий порог:

- `teff_gspphot >= 10000 K`

## Что должен был показать разбор

На этом шаге требовалось:

1. сравнить восстановленный split с сохраненным benchmark;
2. оценить число true `O` в `full`, `train` и `test`;
3. посмотреть, как true `O` раскладывается по `evolution_stage`;
4. проверить поддержку горячего хвоста `O` и граничного среза `O/B`.

## Почему документ сохранен в архиве

Этот план важен как постановка вопроса о поддержке редкого класса в
обучающем контуре. Позднее он был закрыт фактическим разбором и продолжен
анализом разделимости признаков.

Итог этого шага зафиксирован в документе
[coarse_o_train_support_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_train_support_review_round1_ru.md).

## Связанные документы

- [coarse_o_tail_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_tail_review_round1_ru.md)
- [coarse_o_hot_subset_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_hot_subset_review_round1_ru.md)
- [coarse_ob_boundary_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_ob_boundary_review_round1_ru.md)
- [stabilization_issue_ledger_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/stabilization_issue_ledger_ru.md)
