# Тестовый Контур Проекта

Этот каталог хранит весь активный тестовый слой проекта.

Он разделен по ролям, чтобы было понятно, какой тип риска страхует каждая
группа тестов и когда ее нужно запускать.

## Слои Тестирования

- `unit/`:
  проверяет локальную логику модулей, контрактов и helper-слоя;
- `integration/`:
  проверяет короткие связки между несколькими слоями проекта;
- `smoke/`:
  проверяет, что пакет стартует и базовый импортный контур не сломан;
  сюда же входит headless smoke слоя `Streamlit`-страниц;
- `regression/`:
  страхует системное поведение `quality_gate`, `priority`, малого `decide`
  roundtrip и ключевых review-summary.

## Что Не Входит В Активный Контур

- `archive_research/`:
  хранит архив исследовательских проверок и не входит в регулярный `pytest`
  прогон.

## Что Смотреть В Первую Очередь

Если нужно быстро понять тестовую стратегию проекта, лучше идти в таком
порядке:

1. [regression/README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/regression/README.md)
2. [regression/decision](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/regression/decision)
3. [unit/reporting](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/reporting)
4. [smoke/test_package_smoke.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/smoke/test_package_smoke.py)
5. [integration/test_train_score_roundtrip.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/integration/test_train_score_roundtrip.py)

## Базовые Команды

Полный активный тестовый контур:

```bash
.venv-v2/bin/pytest -q tests
```

Только регресс-слой:

```bash
.venv-v2/bin/pytest -q tests/regression
```

Обычный рабочий порядок простой:

- перед крупным изменением достаточно `unit`;
- перед push или после затрагивания слоя правил гоняем весь `tests`;
- если менялся routing или `decide`, обязательно прогоняем `tests/regression`.
