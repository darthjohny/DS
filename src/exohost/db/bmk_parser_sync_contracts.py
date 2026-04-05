# Файл `bmk_parser_sync_contracts.py` слоя `db`.
#
# Этот файл отвечает только за:
# - relation-layer, materialization и SQL-обвязку;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `db` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

B_MK_PARSER_SYNC_SOURCE_RELATION_NAME = "lab.gaia_mk_external_labeled"
B_MK_TRAINING_REFERENCE_RELATION_NAME = "lab.gaia_mk_training_reference"
B_MK_QUALITY_GATED_RELATION_NAME = "lab.gaia_mk_quality_gated"
B_MK_UNKNOWN_REVIEW_RELATION_NAME = "lab.gaia_mk_unknown_review"
B_MK_TRAINING_SUMMARY_RELATION_NAME = "lab.gaia_mk_training_reference_summary"
B_MK_QUALITY_SUMMARY_RELATION_NAME = "lab.gaia_mk_quality_gated_summary"
B_MK_UNKNOWN_SUMMARY_RELATION_NAME = "lab.gaia_mk_unknown_review_summary"

B_MK_PARSER_SYNC_TARGET_RELATION_NAMES: tuple[str, ...] = (
    B_MK_TRAINING_REFERENCE_RELATION_NAME,
    B_MK_QUALITY_GATED_RELATION_NAME,
    B_MK_UNKNOWN_REVIEW_RELATION_NAME,
)

B_MK_PARSER_DERIVED_COLUMNS: tuple[str, ...] = (
    "spectral_class",
    "spectral_subclass",
    "luminosity_class",
    "peculiarity_suffix",
    "label_parse_status",
    "label_parse_notes",
)

B_MK_PARSER_SYNC_JOIN_COLUMNS: tuple[str, ...] = (
    "xmatch_batch_id",
    "source_id",
    "external_row_id",
)


class DbapiCursorProtocol(Protocol):
    # Минимальный DB-API cursor contract, который реально используем в sync-helper.
    def execute(self, operation: str) -> object: ...

    def fetchone(self) -> tuple[object, ...] | None: ...


@dataclass(frozen=True, slots=True)
class BmkParserSyncRelationSummary:
    # Фактическая сводка sync-а для одной downstream relation.
    relation_name: str
    rows_updated: int
    ambiguous_ob_rows: int
    ob_rows: int
    o_rows: int


@dataclass(frozen=True, slots=True)
class BmkParserSyncSummary:
    # Сводка полного downstream sync после parser-fix.
    source_relation_name: str
    relation_summaries: tuple[BmkParserSyncRelationSummary, ...]


__all__ = [
    "B_MK_PARSER_DERIVED_COLUMNS",
    "B_MK_PARSER_SYNC_JOIN_COLUMNS",
    "B_MK_PARSER_SYNC_SOURCE_RELATION_NAME",
    "B_MK_PARSER_SYNC_TARGET_RELATION_NAMES",
    "B_MK_QUALITY_GATED_RELATION_NAME",
    "B_MK_QUALITY_SUMMARY_RELATION_NAME",
    "B_MK_TRAINING_REFERENCE_RELATION_NAME",
    "B_MK_TRAINING_SUMMARY_RELATION_NAME",
    "B_MK_UNKNOWN_REVIEW_RELATION_NAME",
    "B_MK_UNKNOWN_SUMMARY_RELATION_NAME",
    "BmkParserSyncRelationSummary",
    "BmkParserSyncSummary",
    "DbapiCursorProtocol",
]
