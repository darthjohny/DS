# План калибровки слоя host priority

Дата фиксации: `2026-04-02`

Связанные документы:

- [host_priority_calibration_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/host_priority_calibration_round1_ru.md)
- [priority_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/priority_review_round1_ru.md)
- [host_priority_integration_path_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/host_priority_integration_path_ru.md)

## Зачем понадобился этот этап

После первого обзора итогового ранга наблюдений стало видно, что перегруженной
может быть не сама формула ранга, а ее главный входной сигнал — сходство объекта
с host-популяцией.

## Цель

Проверить, насколько хорошо откалиброван `host_similarity_score`, прежде чем
менять итоговый ранг наблюдений, пороги или дополнительные шкалы.

## Что входило в работу

В рамках этого этапа были подготовлены:

1. Общий слой обзора бинарной калибровки.
2. Источник данных для разбора задачи `host` на отложенной выборке.
3. Сводки по кривой надежности, корзинам вероятности и основным метрикам.
4. Основа для принятия решения: оставлять слой как есть или калибровать его
   дополнительно.

## Итог

Этот план отделил разбор качества `host_similarity_score` от более поздней
донастройки самого наблюдательного ранга. Благодаря этому удалось сначала
понять поведение главного сигнала, а уже затем переходить к настройке порогов
для верхней, средней и нижней групп наблюдений.
