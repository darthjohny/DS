# Notebook and SQL Ledger

Дата фиксации: 13 марта 2026 года

Статусы:
- `OK` — можно оставлять как есть
- `TOLERABLE` — в целом нормально, но есть ограничения или шум
- `FIX` — заметная проблема актуальности, чистоты или version-control статуса
- `REMOVE?` — кандидат на удаление или вынос из git

## Notebooks

- [notebooks/eda/00_data_extraction_and_preprocessing.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/00_data_extraction_and_preprocessing.ipynb): `TOLERABLE`
  Содержательно актуален и важен для ВКР, но в outputs сохранён абсолютный локальный путь.

- [notebooks/eda/01_host_eda_overview.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/01_host_eda_overview.ipynb): `TOLERABLE`
  По содержанию живой и полезный, но в outputs остались font/glyph warnings с локальным путём.

- [notebooks/eda/02_router_readiness.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/02_router_readiness.ipynb): `OK`
  Выглядит актуальным и не содержит заметных признаков устаревания.

- [notebooks/eda/03_host_vs_field_contrastive.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/03_host_vs_field_contrastive.ipynb): `OK`
  Роль понятна, прямых следов устаревших comparison run-name не видно.

- [notebooks/eda/04_model_comparison_summary.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/notebooks/eda/04_model_comparison_summary.ipynb): `OK`
  Актуальный summary-notebook под новый `vkr30_cv10` benchmark и snapshot preview.

## SQL migrations

- [sql/2026-03-11_gaia_results_posterior_host_fields.sql](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/2026-03-11_gaia_results_posterior_host_fields.sql): `OK`
  Каноническая forward migration с понятной шапкой и осмысленным scope.

- [sql/2026-03-11_gaia_results_posterior_host_fields.rollback.sql](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/2026-03-11_gaia_results_posterior_host_fields.rollback.sql): `OK`
  Нормальный rollback pair к основной migration.

- [sql/2026-03-13_gaia_results_unknown_constraints.sql](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/2026-03-13_gaia_results_unknown_constraints.sql): `OK`
  Чистая targeted migration под `UNKNOWN/unknown` constraints.

- [sql/2026-03-13_gaia_results_unknown_constraints.rollback.sql](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/2026-03-13_gaia_results_unknown_constraints.rollback.sql): `OK`
  Корректный rollback pair.

## SQL preprocessing

- [sql/preprocessing/01_nasa_gaia_crossmatch.sql](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/preprocessing/01_nasa_gaia_crossmatch.sql): `OK`
  Хорошо оформленный upstream preprocessing layer.

- [sql/preprocessing/02_train_classification_views.sql](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/preprocessing/02_train_classification_views.sql): `OK`
  Ясно выделенный classification layer без лишней магии.

- [sql/preprocessing/03_router_reference_layer.sql](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/preprocessing/03_router_reference_layer.sql): `OK`
  Нормальная router reference aggregation view-логика.

- [sql/preprocessing/04_data_quality_checks.sql](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/preprocessing/04_data_quality_checks.sql): `OK`
  Полезный read-only QA layer, а не одноразовый ad-hoc SQL.

## ADQL

- [sql/adql/01_nasa_hosts_crossmatch_batch_template.adql](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/adql/01_nasa_hosts_crossmatch_batch_template.adql): `OK`
  Понятный batch template для Gaia Archive.

- [sql/adql/02_validation_physics_enrichment.adql](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/adql/02_validation_physics_enrichment.adql): `OK`
  Осмысленный enrichment step.

- [sql/adql/03_gaia_reference_sampling_examples.adql](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/adql/03_gaia_reference_sampling_examples.adql): `OK`
  Нормальный набор reference-sampling examples.

## Cleanup candidates

- [sql/.DS_Store](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/.DS_Store): `REMOVE?`
  Явный файловый мусор, не несущий полезной роли.

## Краткий вывод

- Notebook-слой компактный и живой.
- SQL/ADQL слой оформлен surprisingly чисто и структурно силён.
- Реальные проблемы здесь — не логика, а repo hygiene и немного output-noise в notebooks.
