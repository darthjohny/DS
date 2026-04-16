# Замороженные fixtures для регресс-тестов

Здесь будут лежать небольшие замороженные входы и ожидаемые структуры для
регресс-слоя.

Правила для этого каталога:

- fixtures должны быть компактными;
- fixtures должны быть читаемыми;
- fixtures не должны зависеть от живой БД;
- fixtures не должны превращаться в большие неуправляемые snapshot-дампы.

Типичные файлы этого каталога:

- маленькие входные `csv`;
- компактные `json` со схемами и expected-полями;
- вспомогательные frozen summary-таблицы, если они действительно нужны.

## Текущий Набор Fixtures

### `quality_gate_small.csv`

Нужен для:

- регресса на post-hoc `quality_gate`;
- проверки, что hard reject и review-переходы не размываются.

Источник:

- минимизированная версия synthetic frame из
  [test_posthoc_quality_gate_tuning.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/posthoc/test_posthoc_quality_gate_tuning.py)

### `priority_base_small.csv`

Нужен для:

- регресса на ranking и `priority` integration;
- проверки сжатия `high`-зоны и корректности итоговых labels.

Источник:

- минимизированная версия base frame из
  [test_posthoc_priority_integration.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/posthoc/test_posthoc_priority_integration.py)

### `priority_final_decision_small.csv`

Нужен для:

- регресса на merge `priority` обратно в `final_decision`;
- проверки правил допуска объектов в ranking-контур.

Источник:

- минимизированная версия synthetic final decision frame из
  [test_posthoc_priority_integration.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/posthoc/test_posthoc_priority_integration.py)

### `decide_input_small.csv`

Нужен для:

- малого сквозного регресса `decide`;
- проверки структуры `decision input` и обязательных файлов артефактов.

Источник:

- минимизированная версия input CSV из
  [test_cli_decide.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/cli/test_cli_decide.py)

### `decision_artifact_schema_small.json`

Нужен для:

- регресса на schema и metadata малого `decide`-bundle;
- проверки обязательных колонок и допустимых итоговых состояний.

Источник:

- зафиксирован по фактическому контракту малого roundtrip-сценария из
  [tests/regression/decision/test_decide_roundtrip_regression.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/regression/decision/test_decide_roundtrip_regression.py)

## Правило Для Следующих Шагов

Если потребуется новый frozen input:

- сначала ищем, можно ли собрать его из уже существующего `unit`-сценария;
- только потом добавляем новую fixture;
- каждый новый файл должен быть описан в этом README.

Этот каталог является внутренней базой frozen fixtures для активного
регресс-слоя и должен оставаться компактным и читаемым.
