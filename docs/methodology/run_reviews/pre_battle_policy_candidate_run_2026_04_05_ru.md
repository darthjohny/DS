# Candidate Run Перед Боевым Прогоном После Донастройки

Дата фиксации: `2026-04-05`

## Базовый Диагностический Run

- [hierarchical_final_decision_2026_04_05_090717_885503](/Users/evgeniikuznetsov/Desktop/dspro-vkr/artifacts/decisions/hierarchical_final_decision_2026_04_05_090717_885503)

## Candidate Run

- [hierarchical_final_decision_2026_04_05_123111_055017](/Users/evgeniikuznetsov/Desktop/dspro-vkr/artifacts/decisions/hierarchical_final_decision_2026_04_05_123111_055017)

Статус:

- принят как новый active baseline для технического review-слоя;
- используется перед следующим содержательным разбором системы.

CLI-policy candidate:

- `--no-quality-require-flame-for-pass`
- `--priority-high-min 0.85`
- `--priority-medium-min 0.55`

Важно:

- `RUWE` оставлен на `1.4`;
- `parallax_over_error` оставлен на `5.0`;
- hard reject по `missing_core_features` не меняется.

## Почему Выбран Именно Такой Вариант

### `quality_gate`

Локальный variant review показал:

- отключение требования `radius_flame` для `pass` дает самый крупный
  аккуратный выигрыш среди одиночных review-ослаблений;
- смягчение `RUWE` тоже двигает coverage, но это уже более рискованная
  астрометрическая правка;
- смягчение `parallax_over_error` в одиночку дает заметно меньший эффект.

Практическое решение:

- пока не трогаем `RUWE`;
- пока не трогаем `parallax_over_error`;
- убираем только `missing_radius_flame -> unknown` из active run policy.

### `priority`

Grid review по порогам дал такую картину:

- baseline `0.75 / 0.45`:
  - `high = 56.38%`
- `0.80 / 0.50`:
  - `high = 50.82%`
- `0.82 / 0.52`:
  - `high = 47.82%`
- `0.85 / 0.55`:
  - `high = 40.55%`
  - `medium = 23.55%`
- `0.88 / 0.58`:
  - `high = 18.14%`
  - `medium` становится слишком широким bucket
- `0.90 / 0.60`:
  - `high = 6.62%`
  - уже слишком жестко

Практическое решение:

- candidate threshold policy для следующего run:
  - `high_min = 0.85`
  - `medium_min = 0.55`

## Сравнение Baseline И Candidate Run

### Final Domain

Baseline:

- `id = 177674`
- `unknown = 223787`
- `ood = 765`

Candidate:

- `id = 183631`
- `unknown = 216491`
- `ood = 2104`

Изменение:

- `unknown -> id = 5957`
- часть строк из старого `unknown` пошла в `ood`, потому что теперь они
  проходят quality-layer и доходят до binary `ID/OOD` gate

### Final Quality State

Baseline:

- `pass = 178439`
- `unknown = 63823`
- `reject = 159964`

Candidate:

- `pass = 185735`
- `unknown = 56527`
- `reject = 159964`

Изменение:

- `+7296` строк переходят из `unknown` в `pass`
- `reject` не меняется вообще

### Priority

Baseline:

- `high = 100173`
- `medium = 18373`
- `low = 59128`

Candidate:

- `high = 72113`
- `medium = 42318`
- `low = 69200`

Переходы:

- `high -> medium = 28125`
- `medium -> low = 4647`

## Вывод

Candidate run подтверждает выбранную политику:

1. `quality_gate` стал мягче ровно в том месте, где это было научно наиболее
   безопасно;
2. `priority` перестал держать слишком широкую `high`-зону;
3. hard reject и верхний routing contract не были размыты.

На текущем этапе это выглядит как лучший рабочий кандидат перед следующим
полноценным боевым прогоном.

После повторной сверки run принят как новый active baseline.
