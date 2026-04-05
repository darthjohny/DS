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

from exohost.db.bmk_crossmatch import BmkCrossmatchMaterializationSummary


def print_materialize_crossmatch_stage(message: str) -> None:
    # Печатаем короткий статус materialize-crossmatch шага.
    print(f"[materialize-crossmatch] {message}")


def print_bmk_crossmatch_materialization_summary(
    summary: BmkCrossmatchMaterializationSummary,
) -> None:
    # Печатаем компактную сводку после materialization canonical crossmatch layer.
    print(
        "[materialize-crossmatch] summary "
        f"source_relation={summary.source_relation_name} "
        f"target_relation={summary.target_relation_name} "
        f"xmatch_batch_id={summary.xmatch_batch_id} "
        f"rows_loaded={summary.rows_loaded} "
        f"distinct_external_rows={summary.distinct_external_rows} "
        f"selected_rows={summary.selected_rows} "
        f"multi_match_external_rows={summary.multi_match_external_rows}"
    )
