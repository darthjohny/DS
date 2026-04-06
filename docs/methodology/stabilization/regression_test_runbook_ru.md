# Runbook Регресс-Тестирования

Связанные документы:

- [regression_test_policy_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/regression_test_policy_ru.md)
- [tests/regression/README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/regression/README.md)
- [README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/README.md)

## Зачем Нужен Этот Runbook

Этот документ фиксирует, как запускать регресс-слой на практике.

Он нужен, чтобы `tests/regression` запускались одинаково у всех и не
смешивались по смыслу с `unit`, `integration` и `smoke`.

## Базовый Быстрый Запуск

Если нужно проверить только регресс-слой после правок в policy, review или
сквозном `decide`, используем:

```bash
.venv-v2/bin/pytest -q tests/regression
```

Это основной локальный запуск для поведенческого контроля проекта.

## Scoped-Запуск По Доменам

Если правка затрагивает только часть регресс-слоя, можно запускать его
точечно.

### `posthoc`

Используем после правок в `quality_gate`, `priority`, routing и tuned policy:

```bash
.venv-v2/bin/pytest -q tests/regression/posthoc
```

### `decision`

Используем после правок в `decide`, artifact bundle и final decision:

```bash
.venv-v2/bin/pytest -q tests/regression/decision
```

### `reporting`

Используем после правок в review-summary и helper-слое notebook:

```bash
.venv-v2/bin/pytest -q tests/regression/reporting
```

## Когда Гонять Весь Регресс-Слой

Полный `tests/regression` обязателен:

- после изменения policy `quality_gate`;
- после изменения порогов или логики `priority`;
- после правок в `decide` и artifact bundle;
- после правок в summary-layer, который используется в technical notebook;
- перед фиксацией нового active baseline.

## Когда Нужен Полный Активный Контур

Если затронуты не только policy и review, а структура проекта шире, после
регресс-слоя запускаем весь активный QA:

```bash
.venv-v2/bin/ruff check src tests
.venv-v2/bin/mypy src tests
.venv-v2/bin/pyright src tests
.venv-v2/bin/pytest -q tests
```

## Что Не Делаем Через Регресс-Слой

Регресс-слой не заменяет:

- `unit` для локальной логики;
- `smoke` для стартового контура;
- полный боевой прогон на рабочей БД.

Он нужен именно для маленьких frozen fixtures и устойчивых инвариантов
поведения системы.
