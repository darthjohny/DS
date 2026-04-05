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

from exohost.db.bmk_parser_sync_contracts import BmkParserSyncSummary
from exohost.db.coarse_ob_provenance_refresh import CoarseObProvenanceRefreshSummary


def print_sync_bmk_parser_stage(message: str) -> None:
    # Печатаем короткий статус sync-bmk-parser шага.
    print(f"[sync-bmk-parser] {message}")


def print_bmk_parser_sync_summary(summary: BmkParserSyncSummary) -> None:
    # Печатаем компактную сводку sync-а parser-derived полей.
    for relation_summary in summary.relation_summaries:
        print(
            "[sync-bmk-parser] relation "
            f"relation={relation_summary.relation_name} "
            f"rows_updated={relation_summary.rows_updated} "
            f"ambiguous_ob_rows={relation_summary.ambiguous_ob_rows} "
            f"ob_rows={relation_summary.ob_rows} "
            f"o_rows={relation_summary.o_rows}"
        )


def print_coarse_ob_provenance_refresh_summary(
    summary: CoarseObProvenanceRefreshSummary,
) -> None:
    # Печатаем компактную сводку refresh-а local O/B provenance layer.
    print(
        "[sync-bmk-parser] provenance "
        f"source_relation={summary.source_relation_name} "
        f"source_rows_loaded={summary.source_rows_loaded} "
        f"audit_clean_rows_updated={summary.audit_clean_rows_updated} "
        f"summary_rows_loaded={summary.summary_rows_loaded} "
        f"crosswalk_rows_loaded={summary.crosswalk_rows_loaded}"
    )
