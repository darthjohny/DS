# Star-Level Review Round 1

## Цель

Этот документ фиксирует первый предметный разбор baseline run:

- `artifacts/decisions/hierarchical_final_decision_2026_03_29_073129_772461`

Задача документа:

- не спорить абстрактно про “хорошая/плохая модель”;
- посмотреть, как система ведет себя на реальных объектах;
- отделить engineering defects от scientifically conservative behaviour.

## Сводка По Run

- `n_rows_input = 402226`
- `n_rows_final_decision = 402226`
- `n_rows_priority_input = 177674`
- `n_rows_priority_ranking = 177674`
- `n_unique_source_id = 402226`

Распределение `final_domain_state`:

- `unknown = 223787` (`55.64%`)
- `id = 177674` (`44.17%`)
- `ood = 765` (`0.19%`)

Распределение `final_quality_state`:

- `pass = 178439` (`44.36%`)
- `reject = 159964` (`39.77%`)
- `unknown = 63823` (`15.87%`)

Топ `final_decision_reason`:

- `refinement_accepted = 177674`
- `quality_reject = 159964`
- `quality_unknown = 63823`
- `hard_ood = 765`

## Первичные Наблюдения

### 1. Главный Драйвер `unknown` По-Прежнему Не Модельный

На baseline run `unknown` формируется в основном не слабостью `coarse` или
`refinement`, а routing по `quality_reject / quality_unknown`.

Это согласуется с предыдущим calibration-study:

- system conservative by design;
- главный вопрос здесь не “переучить модель”, а “корректно ли задан gate”.

### 2. Top Priority Saturation Уже Видна

Top priority rows почти сразу насыщаются к `~1.0`:

- `0.999995`
- `0.999993`
- `0.999992`
- `0.999992`
- `0.999991`

И reason почти одинаковый:

- `сильный host-like сигнал; хорошая наблюдательная пригодность`

Это не доказывает bug, но это уже явный review-сигнал:

- нужно проверить, не слишком ли агрессивен новый `host_similarity_score`;
- нужно проверить распределение `priority_score`, а не только top rows.

### 3. Unknown/Reject Cases Часто Имеют Хорошую Астометрию

В top unknown preview уже видны объекты с:

- очень высоким `parallax_over_error`
- нормальным или умеренным `ruwe`
- при этом с `quality_reject`

Типовой паттерн:

- `parallax_over_error` может быть `> 2000`
- `ruwe` около `1.0 ... 1.2`
- но core physics columns пустые

Это подтверждает старый вывод:

- часть reject-cases уходит в reject не по quality degradation,
  а по `missing_core_features`.

### 4. ID-Кейсы Уже Выглядят Предметно

Для `id` уже есть понятный answer:

- `final_coarse_class`
- `final_refinement_label`
- `radius_flame`
- `lum_flame`
- физические Gaia fields

То есть system уже умеет отвечать на вопрос:

- какой класс,
- какой подкласс,
- с какой confidence,
- с какой downstream priority.

## Выявленные Проблемные Или Сомнительные Места

### SR-01. Priority Saturation

Симптом:

- top priority candidates все очень быстро уходят в `high`
- scores почти насыщены к `1.0`

Риск:

- ranking может быть недостаточно discriminative;
- итоговый список top targets может быть слишком “плоским”.

Следующий шаг:

- посмотреть полное распределение `priority_score` и `priority_label`;
- отдельно проверить вклад `host_similarity_score` и `observability_score`.

### SR-02. Missing Quality Explanation In Saved Decision Bundle

Симптом:

- в saved `decision_input` bundle `quality_reason` и `review_bucket`
  фактически пусты на review-layer;
- из-за этого final review notebook не дает полный explainability trail
  по reject/unknown.

Риск:

- scientific review теряет причину, почему конкретный объект ушел в `unknown`;
- notebook отвечает на вопрос неполно.

Следующий шаг:

- traceability audit decision artifacts;
- проверить persistence layer `decision_input -> saved artifacts`.

### SR-03. Unknown Dominance Needs Interpretation, Not Blind Fix

Симптом:

- `unknown` все еще больше половины run.

Текущий вывод:

- это пока не defect;
- это требует distinction between:
  - valid selective behaviour
  - слишком жесткий source contract

Следующий шаг:

- связать `unknown` rows с availability of core Gaia physics;
- не ослаблять gate вслепую.

## Практический Вывод После Round 1

Система уже боеспособна:

- `class`
- `subclass`
- `OOD/unknown`
- `priority`

Но stabilization-фаза теперь должна идти в три направления:

1. traceability и explainability для `unknown/reject`;
2. проверка saturation в `priority`;
3. measured profiling, а не догадки про тормоза.
