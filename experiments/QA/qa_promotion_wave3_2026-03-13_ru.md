# Promotion Wave 3: notebooks and SQL/ADQL

Дата фиксации: 13 марта 2026 года

## Назначение

Этот документ фиксирует третью promotion wave для versioned current
state проекта.

Wave 3 отделяет воспроизводимые notebook- и SQL-артефакты от
documentation-wave и от code/test wave:

- notebooks проверяются как исполнимые или явно средозависимые артефакты;
- SQL/ADQL проверяются как канонический preprocessing/data-lineage слой;
- schema-change SQL фиксируются отдельно от production Python-кода.

## Состав wave

### Notebooks

- `notebooks/eda/00_data_extraction_and_preprocessing.ipynb`
- `notebooks/eda/04_model_comparison_summary.ipynb`

### SQL and ADQL

- `sql/adql/01_nasa_hosts_crossmatch_batch_template.adql`
- `sql/adql/02_validation_physics_enrichment.adql`
- `sql/adql/03_gaia_reference_sampling_examples.adql`
- `sql/preprocessing/01_nasa_gaia_crossmatch.sql`
- `sql/preprocessing/02_train_classification_views.sql`
- `sql/preprocessing/03_router_reference_layer.sql`
- `sql/preprocessing/04_data_quality_checks.sql`
- `sql/2026-03-13_gaia_results_unknown_constraints.sql`
- `sql/2026-03-13_gaia_results_unknown_constraints.rollback.sql`

## Что проверялось

Для этой wave проверялись:

1. исполнимость канонических notebook-ов через `nbconvert`;
2. отсутствие error outputs после исполнения;
3. целостность markdown-ссылок внутри notebook-ов;
4. наличие осмысленных header-комментариев в SQL/ADQL;
5. согласованность SQL-набора с current preprocessing/OOD narrative.

## Проверки

Выполнены следующие проверки:

```bash
./venv/bin/python -m nbconvert --to notebook --execute --inplace \
  notebooks/eda/04_model_comparison_summary.ipynb

./venv/bin/python -m nbconvert --to notebook --execute --inplace \
  notebooks/eda/00_data_extraction_and_preprocessing.ipynb

./venv/bin/python - <<'PY'
# link-check markdown cells inside 00 and 04 notebooks
PY

./venv/bin/python - <<'PY'
# count notebook outputs and error outputs
PY

for f in sql/adql/*.adql sql/preprocessing/*.sql \
  sql/2026-03-13_gaia_results_unknown_constraints.sql \
  sql/2026-03-13_gaia_results_unknown_constraints.rollback.sql; do
  sed -n '1,20p' "$f"
done
```

Результат:

- `00_data_extraction_and_preprocessing.ipynb`: executed successfully
- `04_model_comparison_summary.ipynb`: executed successfully
- notebook markdown links: `green`
- notebook error outputs: `0`
- SQL/ADQL header set: `present`

## Что важно в этой wave

- preprocessing notebook теперь подтверждён как реально исполнимый
  current-state артефакт в текущем окружении;
- summary notebook comparison-layer также исполним top-to-bottom;
- SQL/ADQL слой оформлен как осмысленный канонический preprocessing
  контур, а не как набор случайных выгрузок;
- migration `2026-03-13_*` читается как актуальное расширение OOD /
  UNKNOWN contract, а не как исторический residue.

## Статус

Статус wave: `promotion-ready`.

Это означает:

- notebooks и SQL/ADQL можно рассматривать как следующий цельный блок
  для консолидации versioned current state;
- для этой wave не требуется дополнительная содержательная правка перед
  переходом к следующему организационному шагу;
- дальше остаётся в основном слой canonical generated artifacts.

## Что не входит в wave

В эту wave сознательно не включены:

- `notebooks/eda/01_host_eda_overview.ipynb`
- `notebooks/eda/02_router_readiness.ipynb`
- `notebooks/eda/03_host_vs_field_contrastive.ipynb`
- `notebooks/eda/data/`
- historical SQL `2026-03-11_*`
- generated comparison artifacts в `experiments/model_comparison/`

Эти материалы либо уже закрыты другой wave, либо не являются главным
текущим ядром preprocessing/comparison packaging.

## Следующий кандидат на wave

Следующий естественный блок:

- canonical comparison artifacts в `experiments/model_comparison/`;
- затем полный operational current-state consolidation.
