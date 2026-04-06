# Интерпретация Доверия К Результату Для ВКР

Дата фиксации: `2026-04-06`

Связанные документы:

- [vkr_interpretation_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/vkr_interpretation_tz_ru.md)
- [vkr_interpretation_source_of_truth_2026_04_06_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/vkr/vkr_interpretation_source_of_truth_2026_04_06_ru.md)
- [regression_validation_run_2026_04_06_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/regression_validation_run_2026_04_06_ru.md)
- [regression_test_policy_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/regression_test_policy_ru.md)
- [regression_test_runbook_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/regression_test_runbook_ru.md)

## Зачем Нужен Этот Документ

Для ВКР важно не только показать итоговый результат, но и объяснить, почему
этому результату можно доверять.

В нашем случае доверие к результату строится не на одной красивой метрике, а
на нескольких слоях проверки:

- reproducible active baseline;
- validation run после введения regression-layer;
- отдельный слой регресс-тестирования;
- согласованность run-review, notebook и artifact bundle.

## Что Именно Здесь Понимается Под Доверием

В этом документе доверие означает не “научную окончательность” и не
“доказательство наличия экзопланет”.

Здесь под доверием понимается следующее:

- current active baseline воспроизводим;
- итоговый pipeline не меняется тихо после инженерных правок;
- основной прикладной результат держится после повторной проверки;
- итоговые артефакты согласованы между собой;
- shortlist верхнего приоритета устойчив и не является случайным разовым
  артефактом.

То есть мы говорим об engineering trust и operational reliability, а не о
полном снятии всех научных ограничений.

## Active Baseline Как Опора

Главным результатом проекта считается active baseline:

- [hierarchical_final_decision_2026_04_05_123111_055017](/Users/evgeniikuznetsov/Desktop/dspro-vkr/artifacts/decisions/hierarchical_final_decision_2026_04_05_123111_055017)

Он используется как источник для:

- итоговых чисел по routing;
- итоговой `quality_gate` policy;
- итоговой `priority` policy;
- интерпретации верхнего shortlist.

Но сам по себе один run еще не дает достаточного доверия.

## Почему Validation Run Важен

После введения regression-layer был выполнен validation run:

- [hierarchical_final_decision_2026_04_06_095722_391062](/Users/evgeniikuznetsov/Desktop/dspro-vkr/artifacts/decisions/hierarchical_final_decision_2026_04_06_095722_391062)

Его задача:

- не заменить baseline;
- не дать новый scientific result;
- а подтвердить, что система после инженерной доработки ведет себя так же, как
  раньше.

### Что Именно Он Подтвердил

Validation run дал полное совпадение с active baseline по:

- `decision_input.csv`
- `final_decision.csv`
- `priority_input.csv`
- `priority_ranking.csv`

И по основным счетчикам:

- `n_rows_input = 402226`
- `id = 183631`
- `unknown = 216491`
- `ood = 2104`
- `high = 72113`
- `medium = 42318`
- `low = 69200`

Это очень важный вывод.

Он означает, что после введения нового regression-layer:

- pipeline не дрейфнул;
- policy не “поехала”;
- и главный результат сохраняется без ручного подгона.

## Почему Регресс-Слой Усиливает Доверие

До появления `tests/regression` проект уже имел:

- сильный `unit`;
- `smoke`;
- `integration`.

Но этого не хватало для ответа на вопрос:

> если мы что-то правим в policy, review или decision-layer, система в целом
> все еще ведет себя так же?

Теперь этот вопрос закрыт отдельным regression-layer.

Он страхует:

- `quality_gate`;
- `priority`;
- малый `decide` roundtrip;
- schema artifact bundle;
- high-priority cohort;
- итоговые review-summary.

То есть доверие к результату теперь поддерживается не только документами и
ноутбуками, но и формальным автоматическим слоем проверки.

## Почему Это Важно Для Shortlist

Главный прикладной результат проекта — верхний shortlist из `72 113` объектов.

Для доверия к нему важно не только то, что он выглядит разумно физически, но и
то, что он:

- воспроизводится;
- не меняется после инженерных правок;
- остается тем же после validation run;
- сохраняет тот же классовый профиль и те же медианные показатели.

Именно это и было подтверждено после введения regression-layer.

## Что Этот Блок Не Доказывает

Этот слой доверия не доказывает:

- что все объекты верхнего shortlist обязательно имеют экзопланеты;
- что данные Gaia и NASA не имеют ограничений;
- что проблема `O/B` полностью решена;
- что refinement-layer уже полностью зрелый на уровне всех подклассов.

То есть доверие здесь ограничено разумными рамками:

- reproducibility;
- consistency;
- stable pipeline behavior.

## Как Это Правильно Формулировать В Тексте ВКР

Корректная формулировка:

> Доверие к итоговому результату обеспечивается не только качеством отдельных
> моделей, но и воспроизводимостью всего pipeline. После введения отдельного
> regression-layer был выполнен validation run, который показал полное
> совпадение с active baseline по основным artifact-таблицам и итоговым
> счетчикам. Это позволяет рассматривать текущий shortlist как устойчивый
> результат работы системы, а не как разовый удачный прогон.

Некорректная формулировка:

- “validation run доказал научную истинность результата”;
- “раз все совпало, значит ограничений больше нет”;
- “регресс-тесты заменяют научную проверку”.

## Вывод

Текущему результату можно доверять в инженерном и прикладном смысле.

Это доверие опирается на:

- active baseline;
- validation run;
- regression-layer;
- согласованность notebook, run-review и artifact bundle.

Следовательно, верхний shortlist и текущее состояние pipeline можно использовать
в ВКР как устойчивый и воспроизводимый результат, при этом отдельно сохраняя
честный блок научных ограничений.
