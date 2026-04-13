# План разбора границы O/B

## Зачем проводился этот шаг

Этот документ сохраняет исходный план следующего шага после выделения горячего
подмножества `O`.

На том этапе требовалось:

- не трогать всю coarse-модель;
- отдельно разобрать узкую границу `O` и `B` на горячем подмножестве;
- понять, это ошибка модели на границе классов или отражение пересечения
  исходных меток и физики.

## Почему понадобился отдельный разбор границы

Разбор горячего подмножества уже показал:

- после отсечения холодного хвоста исчезают `F/G/K/M`;
- физически горячее подмножество почти целиком уходит в `B`;
- значит вопрос смещается с общего класса `O` на более узкую границу `O/B`.

## Базовый срез этого шага

Первый разбор границы строился на подмножестве:

- `spectral_class IN ('O', 'B')`
- `quality_state = 'pass'`
- `teff_gspphot >= 10000 K`

## Что должен был показать разбор

На этом шаге требовалось:

1. оценить размер горячего граничного среза `O/B`;
2. посмотреть, как coarse-модель распределяет предсказания между `O` и `B`;
3. сравнить вероятности `P(O)` и `P(B)` на true `O` и true `B`;
4. проверить, есть ли заметное физическое перекрытие между этими группами.

## Почему документ сохранен в архиве

Этот план важен как постановка узкой проблемы на границе `O/B`. Позднее стало
ясно, что после него нужно было разбирать уже не только границу в прогнозах, а
еще и обучающую поддержку `O` и разделимость признаков.

Итог этого шага зафиксирован в документе
[coarse_ob_boundary_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_ob_boundary_review_round1_ru.md).

## Связанные документы

- [coarse_o_tail_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_tail_review_round1_ru.md)
- [coarse_o_hot_subset_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_hot_subset_review_round1_ru.md)
- [stabilization_issue_ledger_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/stabilization_issue_ledger_ru.md)
