# Unit-тесты SQL-контракта narrow loader для true `O` rows.

from __future__ import annotations

from exohost.contracts.quality_gate_dataset_contracts import (
    GAIA_MK_QUALITY_GATE_AUDIT_CONTRACT,
)
from exohost.datasets.archive_research.load_coarse_o_review_dataset import (
    build_coarse_o_review_query,
)


def test_build_coarse_o_review_query_contains_true_o_filter_and_quality_state() -> None:
    query = build_coarse_o_review_query(
        GAIA_MK_QUALITY_GATE_AUDIT_CONTRACT.relation_name,
        GAIA_MK_QUALITY_GATE_AUDIT_CONTRACT.required_columns + ("spectral_class", "random_index"),
        quality_state="pass",
        limit=25,
    )

    assert "FROM lab.gaia_mk_quality_gated" in query
    assert "spectral_class = 'O'" in query
    assert "quality_state = 'pass'" in query
    assert "ORDER BY random_index ASC, source_id ASC" in query
    assert "LIMIT 25" in query


def test_build_coarse_o_review_query_without_quality_filter_keeps_all_o_rows() -> None:
    query = build_coarse_o_review_query(
        GAIA_MK_QUALITY_GATE_AUDIT_CONTRACT.relation_name,
        GAIA_MK_QUALITY_GATE_AUDIT_CONTRACT.required_columns + ("spectral_class",),
    )

    assert "spectral_class = 'O'" in query
    assert "quality_state =" not in query
    assert "ORDER BY source_id ASC" in query
