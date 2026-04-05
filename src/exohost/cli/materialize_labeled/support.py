# Файл `support.py` слоя `cli`.
#
# Этот файл отвечает только за:
# - CLI-команды и orchestration entrypoints;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - CLI-команды или support-модули этого же домена;
# - пользовательский запуск через `python -m exohost.cli.main`.

from __future__ import annotations

from exohost.db.bmk_labeled import BmkExternalLabeledLoadSummary


def print_materialize_labeled_stage(message: str) -> None:
    # Печатаем короткий статус materialize-labeled шага.
    print(f"[materialize-labeled] {message}")


def print_bmk_external_labeled_load_summary(
    summary: BmkExternalLabeledLoadSummary,
) -> None:
    # Печатаем компактную сводку после materialization labeled relation.
    print(
        "[materialize-labeled] summary "
        f"filtered_relation={summary.filtered_relation_name} "
        f"crossmatch_relation={summary.crossmatch_relation_name} "
        f"target_relation={summary.target_relation_name} "
        f"xmatch_batch_id={summary.xmatch_batch_id} "
        f"rows_loaded={summary.rows_loaded} "
        f"distinct_external_rows={summary.distinct_external_rows} "
        f"distinct_source_ids={summary.distinct_source_ids} "
        f"duplicate_source_ids={summary.duplicate_source_ids} "
        f"parsed_rows={summary.parsed_rows} "
        f"partial_rows={summary.partial_rows} "
        f"unsupported_rows={summary.unsupported_rows} "
        f"empty_rows={summary.empty_rows} "
        f"rows_without_luminosity_class={summary.rows_without_luminosity_class}"
    )
