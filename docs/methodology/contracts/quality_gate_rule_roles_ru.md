# Роли Правил `quality_gate` Первой Волны

Дата фиксации: `2026-04-05`

Связанные документы:

- [quality_ood_contract_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/quality_ood_contract_ru.md)
- [pre_battle_tuning_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/pre_battle_tuning_tz_ru.md)

## Зачем Нужен Этот Документ

Перед донастройкой `quality_gate` нужно явно разделить текущие сигналы по
ролям.

Иначе легко смешать:

- hard reject;
- review;
- OOD-подозрение;
- и просто диагностические признаки.

## Правило Разделения

В первой волне используем три роли:

- `reject`
- `review`
- `info`

И три области действия:

- `quality`
- `ood`
- `support`

## Текущее Разделение

### 1. Hard Reject

#### `missing_core_features`

- роль: `reject`
- область: `quality`
- live reason:
  - `reject_missing_core_features`

Смысл:

- если нет критических core-признаков, объект не должен проходить в normal
  контур как валидный `pass`.

### 2. Review-Правила

#### `high_ruwe`

- роль: `review`
- область: `quality`
- live reason:
  - `review_high_ruwe`

Смысл:

- это quality-risk, но не hard reject.

#### `low_parallax_snr`

- роль: `review`
- область: `quality`
- live reason:
  - `review_low_parallax_snr`

Смысл:

- слабая дистанционная информация ограничивает доверие, но не делает строку
  автоматическим reject.

#### `missing_flame_features`

- роль: `review`
- область: `quality`
- live reason:
  - `review_missing_radius_flame`

Смысл:

- отсутствие FLAME-радиуса ограничивает normal pass, но это review, а не hard
  reject.

### 3. Информационные И OOD-Чувствительные Сигналы

#### `non_single_star_flag`

- роль: `info`
- область: `ood`
- live bucket:
  - `review_non_single_star`

Смысл:

- это не quality reject, а признак повышенного domain/interpretation risk.

#### `low_single_star_probability`

- роль: `info`
- область: `ood`
- live bucket:
  - `review_low_single_star_probability`

Смысл:

- это OOD-подозрение и review-сигнал, а не quality reject.

### 4. Поддерживающие Диагностические Сигналы

#### `has_core_features`

- роль: `info`
- область: `support`

#### `has_flame_features`

- роль: `info`
- область: `support`

Смысл:

- это supporting audit-signals;
- они нужны для explainability и review;
- они не являются самостоятельной policy-командой.

## Вывод Для Следующего Шага

После такого разделения следующий variant review должен сравнивать:

- только те изменения, которые реально относятся к `review`-правилам;
- не трогать `reject_missing_core_features` без отдельного доказательства;
- не смешивать `quality` и `ood` сигналы в одном варианте политики.
