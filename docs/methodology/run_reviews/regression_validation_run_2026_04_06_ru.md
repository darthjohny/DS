# Валидационный Run После Введения Регресс-Слоя

Дата фиксации: `2026-04-06`

## Цель

Этот run нужен не для смены baseline, а для проверки, что введение
`tests/regression` и связанного QA-слоя не изменило боевое поведение системы.

## Run

- validation run:
  - [hierarchical_final_decision_2026_04_06_095722_391062](/Users/evgeniikuznetsov/Desktop/dspro-vkr/artifacts/decisions/hierarchical_final_decision_2026_04_06_095722_391062)
- active baseline:
  - [hierarchical_final_decision_2026_04_05_123111_055017](/Users/evgeniikuznetsov/Desktop/dspro-vkr/artifacts/decisions/hierarchical_final_decision_2026_04_05_123111_055017)

CLI-policy:

- `--no-quality-require-flame-for-pass`
- `--priority-high-min 0.85`
- `--priority-medium-min 0.55`

## Сравнение С Active Baseline

### Final Decision

Сравнение полных artifact-таблиц дало точное совпадение:

- `decision_input.csv` — совпадает полностью
- `final_decision.csv` — совпадает полностью
- `priority_input.csv` — совпадает полностью
- `priority_ranking.csv` — совпадает полностью

### Основные Счетчики

Оба run дают один и тот же результат:

- `n_rows_input = 402226`
- `id = 183631`
- `unknown = 216491`
- `ood = 2104`

`quality_gate`:

- `pass = 185735`
- `reject = 159964`
- `unknown = 56527`

`priority`:

- `high = 72113`
- `medium = 42318`
- `low = 69200`

## Состояние Системы

Этот validation run подтверждает:

1. Регресс-слой не изменил поведение рабочего pipeline.
2. Active baseline остается воспроизводимым.
3. Текущая tuned policy `quality_gate + priority` держится стабильно.

## Состояние Моделей

Рабочий model-layer остается прежним:

- `id_ood`:
  - `accuracy = 0.995734`
  - `balanced_accuracy = 0.926215`
  - `macro_f1 = 0.944521`
  - `roc_auc_ovr = 0.996188`
- `coarse`:
  - `accuracy = 0.992926`
  - `balanced_accuracy = 0.992379`
  - `macro_f1 = 0.992573`
  - `roc_auc_ovr = 0.999977`
- `refinement_flat`:
  - `accuracy = 0.320336`
  - `balanced_accuracy = 0.187861`
  - `macro_f1 = 0.189683`
  - `roc_auc_ovr = 0.713111`
- `host_field`:
  - `accuracy = 0.955446`
  - `balanced_accuracy = 0.955446`
  - `macro_f1 = 0.955442`
  - `roc_auc = 0.990521`
  - `brier_score = 0.035357`
  - `log_loss = 0.134080`

## High-Priority Cohort

Верхний shortlist остается таким же:

- `n_high = 72113`
- классовый профиль:
  - `K = 30114`
  - `F = 27488`
  - `G = 14341`
  - `M = 170`
- медианные показатели:
  - `priority_score = 0.877863`
  - `host_similarity_score = 0.997999`
  - `observability_score = 0.662852`

Это подтверждает, что верхняя приоритетная группа остается host-like и
физически правдоподобной для follow-up shortlist.

## Вывод

Боевой pipeline работает стабильно.

После введения регресс-слоя проект сохраняет тот же active baseline, тот же
ranking и те же итоговые решения. Значит дальше можно переходить не к спасению
контуров, а к спокойной интерпретации результатов и последующей косметической
доводке проекта.
