# План калибровки порогов наблюдательного ранга

Дата фиксации: `2026-04-05`

Связанные документы:

- [priority_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/priority_review_round1_ru.md)
- [host_priority_calibration_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/host_priority_calibration_round1_ru.md)
- [priority_threshold_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/priority_threshold_review_round1_ru.md)
- [priority_threshold_review_round2_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/priority_threshold_review_round2_ru.md)

## Зачем понадобился этот этап

После первого разбора ранга наблюдений стало видно, что верхняя группа слишком
широкая. Нужно было понять, можно ли сделать ее более читаемой за счет одних
только порогов, не меняя саму формулу ранга.

## Цель

Проверить, достаточно ли скорректировать пороги для верхней, средней и нижней
групп наблюдательного ранга, прежде чем переходить к более сложной
перенастройке.

## Главный принцип

Сначала проверяются простые изменения:

- сохраняется существующий `priority_score`;
- сравниваются несколько вариантов порогов;
- оценивается, как меняется размер и состав верхней группы;
- только после этого рассматриваются более сложные варианты.

## Что входило в работу

В рамках этого плана были выполнены:

1. Подготовка единого слоя обзора порогов.
2. Сравнение нескольких наборов порогов для `high`, `medium` и `low`.
3. Проверка, как меняется верхняя группа на уровне классов и причин попадания.
4. Выбор более читаемого рабочего варианта.

## Итог

По результатам этого этапа был выбран более строгий вариант:

- `high_min = 0.85`
- `medium_min = 0.55`

Этот вариант лучше сжал верхнюю группу и сделал итоговый ранг наблюдений более
удобным для интерпретации, не требуя новой формулы или дополнительной
калибровки.
