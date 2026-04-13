# Обзор вариантов для `quality_gate` на диагностическом базовом прогоне

Дата фиксации: `2026-04-05`

Базовый прогон:

- [hierarchical_final_decision_2026_04_05_090717_885503](/Users/evgeniikuznetsov/Desktop/dspro-vkr/artifacts/decisions/hierarchical_final_decision_2026_04_05_090717_885503)

Связанные документы:

- [quality_gate_rule_roles_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/quality_gate_rule_roles_ru.md)

## Цель

Проверить варианты политики для `quality_gate` без изменения основной логики.

Важно:

- этот review затрагивает только `review`-правила;
- жесткое отклонение по `missing_core_features` не меняется;
- OOD-сигналы тоже не меняются в этом сравнении.

## Сравниваемые варианты

### Базовый вариант

- `ruwe_unknown_threshold = 1.4`
- `parallax_snr_unknown_threshold = 5.0`
- `require_flame_for_pass = True`

### Смягченный вариант

- `ruwe_unknown_threshold = 1.6`
- `parallax_snr_unknown_threshold = 3.0`
- `require_flame_for_pass = False`

### Строгий вариант

- `ruwe_unknown_threshold = 1.2`
- `parallax_snr_unknown_threshold = 7.0`
- `require_flame_for_pass = True`

## Итоговая Сводка

### Базовый вариант

- `pass = 178439`
- `unknown = 63823`
- `reject = 159964`

### Смягченный вариант

- `pass = 195815`
- `unknown = 46447`
- `reject = 159964`

Изменение относительно базового варианта:

- `+17376` строк переходят из `unknown` в `pass`
- `reject` не меняется вообще

### Строгий вариант

- `pass = 162741`
- `unknown = 79521`
- `reject = 159964`

Изменение относительно базового варианта:

- `15698` строк переходят из `pass` в `unknown`
- `reject` не меняется вообще

## Что Важно По Переходам

### Смягченный вариант

Переходы:

- только `unknown -> pass`
- нет ни одного `reject -> pass`
- нет ни одного `pass -> reject`

Это означает:

- смягченный вариант действительно работает только по правилам разбора;
- слой жесткого отклонения не размывается.

### Строгий вариант

Переходы:

- только `pass -> unknown`
- нет ни одного `reject -> pass`
- нет ни одного `unknown -> reject`

Это тоже подтверждает:

- строгий вариант изолированно усиливает слой разбора;
- слой жесткого отклонения остается тем же.

## Какие правила реально двигают строки

### Смягченный вариант

Основные причины возврата строк в `pass`:

- `review_high_ruwe`
- `review_missing_radius_flame`
- `review_low_parallax_snr`

Пример изменившихся строк подтверждает, что меняются именно эти случаи из слоя
разбора.

### Строгий вариант

Основные причины перевода строк в `unknown`:

- `review_high_ruwe`
- `review_low_parallax_snr`

## Вывод

Этот review подтверждает два важных факта:

1. вспомогательный слой сравнения вариантов работает корректно и не смешивает
   разбор с жестким отклонением;
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
