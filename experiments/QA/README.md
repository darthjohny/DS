# QA Artifacts

Дата актуализации: 13 марта 2026 года

## Назначение

Каталог хранит результаты QA-аудита и связанные рабочие артефакты.

## Канонический набор полного аудита

- `qa_full_audit_log_2026-03-13_ru.md`
- `qa_backlog_and_decision_map_2026-03-13_ru.md`
- `qa_tracking_manifest_2026-03-13_ru.md`
- `qa_operational_consolidation_plan_2026-03-13_ru.md`
- `qa_operational_batch1_runtime_foundation_2026-03-13_ru.md`
- `qa_operational_batch2_comparison_wave_2026-03-13_ru.md`
- `qa_operational_batch3_repo_policy_2026-03-13_ru.md`
- `qa_operational_batch4_docs_presentation_2026-03-13_ru.md`
- `qa_operational_batch5_notebooks_sql_2026-03-13_ru.md`
- `qa_operational_batch6_artifacts_qa_2026-03-13_ru.md`
- `qa_operational_batch7_historical_layer_2026-03-13_ru.md`
- `qa_promotion_wave1_2026-03-13_ru.md`
- `qa_promotion_wave2_2026-03-13_ru.md`
- `qa_promotion_wave3_2026-03-13_ru.md`
- `qa_promotion_wave4_2026-03-13_ru.md`
- `qa_file_ledger_python_2026-03-13_ru.md`
- `qa_test_coverage_matrix_2026-03-13_ru.md`
- `qa_runbook_2026-03-13_ru.md`
- `qa_docs_ledger_2026-03-13_ru.md`
- `qa_notebooks_sql_ledger_2026-03-13_ru.md`
- `qa_artifacts_cleanup_2026-03-13_ru.md`

## Исторические файлы

- `qa_mvp_report_2026-03-11.md` — ранний, более узкий QA-срез до полного
  file-by-file и architecture review.

## Политика обновления

- новые полноценные QA-проходы сохраняются отдельной датированной волной;
- старые волны не считаются мусором, но текущим источником истины
  считается самая свежая полная волна;
- для большого `untracked` current state отдельно ведётся tracking-manifest,
  чтобы не путать ценные материалы с локальным шумом;
- operational consolidation следующего шага отдельно разложен в
  `qa_operational_consolidation_plan_2026-03-13_ru.md`;
- generated QA-файлы имеют смысл держать в репозитории, потому что они
  фиксируют findings, backlog и принятые решения.
