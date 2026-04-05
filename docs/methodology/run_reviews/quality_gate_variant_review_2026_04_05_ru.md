# Variant Review Для `quality_gate` На Диагностическом Baseline

Дата фиксации: `2026-04-05`

Baseline:

- [hierarchical_final_decision_2026_04_05_090717_885503](/Users/evgeniikuznetsov/Desktop/dspro-vkr/artifacts/decisions/hierarchical_final_decision_2026_04_05_090717_885503)

Связанные документы:

- [quality_gate_rule_roles_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/quality_gate_rule_roles_ru.md)
- [pre_battle_tuning_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/pre_battle_tuning_tz_ru.md)

## Цель

Проверить candidate policy-варианты для `quality_gate` без изменения mainline.

Важно:

- этот review затрагивает только `review`-правила;
- hard reject по `missing_core_features` не меняется;
- OOD-сигналы тоже не меняются в этом сравнении.

## Сравниваемые Варианты

### Baseline

- `ruwe_unknown_threshold = 1.4`
- `parallax_snr_unknown_threshold = 5.0`
- `require_flame_for_pass = True`

### Relaxed

- `ruwe_unknown_threshold = 1.6`
- `parallax_snr_unknown_threshold = 3.0`
- `require_flame_for_pass = False`

### Strict

- `ruwe_unknown_threshold = 1.2`
- `parallax_snr_unknown_threshold = 7.0`
- `require_flame_for_pass = True`

## Итоговая Сводка

### Baseline

- `pass = 178439`
- `unknown = 63823`
- `reject = 159964`

### Relaxed

- `pass = 195815`
- `unknown = 46447`
- `reject = 159964`

Изменение относительно baseline:

- `+17376` строк переходят из `unknown` в `pass`
- `reject` не меняется вообще

### Strict

- `pass = 162741`
- `unknown = 79521`
- `reject = 159964`

Изменение относительно baseline:

- `15698` строк переходят из `pass` в `unknown`
- `reject` не меняется вообще

## Что Важно По Переходам

### Relaxed

Переходы:

- только `unknown -> pass`
- нет ни одного `reject -> pass`
- нет ни одного `pass -> reject`

Это означает:

- relaxed-вариант действительно работает только по review-правилам;
- hard reject слой не размывается.

### Strict

Переходы:

- только `pass -> unknown`
- нет ни одного `reject -> pass`
- нет ни одного `unknown -> reject`

Это тоже подтверждает:

- strict-вариант изолированно усиливает review-слой;
- hard reject слой остается тем же.

## Какие Правила Реально Двигают Строки

### Relaxed

Основные причины возврата строк в `pass`:

- `review_high_ruwe`
- `review_missing_radius_flame`
- `review_low_parallax_snr`

Пример changed rows подтверждает, что меняются именно эти review-cases.

### Strict

Основные причины перевода строк в `unknown`:

- `review_high_ruwe`
- `review_low_parallax_snr`

## Вывод

Этот review подтверждает два важных факта:

1. helper-слой сравнения вариантов работает корректно и не смешивает review с
   hard reject;
2. главный trade-off сейчас действительно сидит в review-правилах, а не в
   `reject_missing_core_features`.

Практический вывод для следующего шага:

- обсуждать нужно не “ослаблять весь gate”;
- обсуждать нужно только:
  - `high_ruwe`
  - `low_parallax_snr`
  - `missing_radius_flame`

## Следующий Шаг

- принять policy decision по `quality_gate` на основании этого review.
