# Регресс-Тесты Проекта

Этот каталог хранит отдельный слой регресс-тестирования.

Он нужен для того, чтобы страховать не только отдельные функции, но и ключевые
системные инварианты проекта:

- поведение `quality_gate`;
- поведение `priority`;
- малый сквозной контур `decide`;
- структуру и смысл основных artifact bundle;
- устойчивость ключевых summary-слоев.

Текущая роль каталога:

- каркас нового слоя;
- база для frozen fixtures;
- доменные подпапки под системные regression-тесты.

Внутренний helper-слой:

- [fixture_loaders.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/regression/fixture_loaders.py)
  читает frozen fixtures;
- [assertions.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/regression/assertions.py)
  дает компактные проверки schema/scalar/dataframe;
- [test_regression_helpers.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/regression/test_regression_helpers.py)
  страхует сам helper-контур.

Сквозной decision-регресс:

- [decision/decide_roundtrip_testkit.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/regression/decision/decide_roundtrip_testkit.py)
  собирает маленький локальный контур artifacts и frozen input для `decide`;
- [decision/test_decide_roundtrip_regression.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/regression/decision/test_decide_roundtrip_regression.py)
  страхует полный run bundle, metadata и базовые допустимые состояния выхода.
- [decision/test_decision_artifact_schema_regression.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/regression/decision/test_decision_artifact_schema_regression.py)
  страхует обязательные колонки и metadata-контракты малого decision-bundle.
- [decision/test_high_priority_cohort_regression.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/regression/decision/test_high_priority_cohort_regression.py)
  страхует, что верхняя приоритетная группа остается непустой и сохраняет host-like профиль.

Review и summary-регресс:

- [reporting/test_final_decision_summary_regression.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/regression/reporting/test_final_decision_summary_regression.py)
  страхует итоговые summary-таблицы, причины routing и статус priority-review слоя.

Подробная политика слоя зафиксирована в
[/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/regression_test_policy_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/regression_test_policy_ru.md).

Практический runbook запуска лежит в
[/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/regression_test_runbook_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/regression_test_runbook_ru.md).
