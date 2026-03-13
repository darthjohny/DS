# Operational Batch 5 Status

Дата фиксации: 13 марта 2026 года

## Что проверялось

Батч 5 из
[qa_operational_consolidation_plan_2026-03-13_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/QA/qa_operational_consolidation_plan_2026-03-13_ru.md):
canonical notebooks и SQL/ADQL.

Проверяемый слой:

- `notebooks/eda/00_data_extraction_and_preprocessing.ipynb`
- `notebooks/eda/01_host_eda_overview.ipynb`
- `notebooks/eda/02_router_readiness.ipynb`
- `notebooks/eda/03_host_vs_field_contrastive.ipynb`
- `notebooks/eda/04_model_comparison_summary.ipynb`
- `sql/2026-03-13_gaia_results_unknown_constraints.sql`
- `sql/2026-03-13_gaia_results_unknown_constraints.rollback.sql`
- `sql/adql/*`
- `sql/preprocessing/*`

## Acceptance-check

### Notebook execution

Переисполнены:

- `./venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/eda/00_data_extraction_and_preprocessing.ipynb`
- `./venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/eda/04_model_comparison_summary.ipynb`

Результат:

- оба notebook-а успешно записаны обратно без execution-errors.

### Notebook bootstrap

Проверено содержимое `01`, `02`, `03` на явный bootstrap:

- поиск `REPO_ROOT`
- поиск `SRC_ROOT`
- добавление `src/` в `sys.path`

Результат:

- все три notebook-а содержат явную bootstrap-схему через `REPO_ROOT`
  и `SRC_ROOT`;
- скрытой зависимости от import-time path side effects не осталось.

### SQL/ADQL existence check

Проверено существование:

- root SQL change/rollback pair;
- `sql/adql/01..03`;
- `sql/preprocessing/01..04`.

Результат:

- missing SQL/ADQL files не обнаружены.

## Итог

Батч 5 считается operationally ready:

- канонические notebook-и `00` и `04` реально исполняются;
- `01-03` технически синхронизированы по import bootstrap;
- SQL/ADQL слой присутствует в полном объёме;
- `notebooks/eda/data/` по-прежнему не считается частью канонического
  versioned current state.

## Следующий шаг

Следующий практический шаг по plan-order:

- Батч 6: canonical generated artifacts и QA wave.
