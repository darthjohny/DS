# Unit-тесты SQL-контракта narrow loader для `O/B` boundary source.

from __future__ import annotations

from exohost.contracts.quality_gate_dataset_contracts import (
    GAIA_MK_QUALITY_GATE_AUDIT_CONTRACT,
)
from exohost.datasets.archive_research.load_coarse_ob_boundary_review_dataset import (
    build_coarse_ob_boundary_review_query,
)


def test_build_coarse_ob_boundary_review_query_contains_ob_filters() -> None:
    query = build_coarse_ob_boundary_review_query(
        GAIA_MK_QUALITY_GATE_AUDIT_CONTRACT.relation_name,
        GAIA_MK_QUALITY_GATE_AUDIT_CONTRACT.required_columns + ("spectral_class", "random_index"),
        quality_state="pass",
        teff_min_k=10000.0,
        limit=25,
    )

    assert "FROM lab.gaia_mk_quality_gated" in query
    assert "spectral_class IN ('O', 'B')" in query
    assert "quality_state = 'pass'" in query
    assert "teff_gspphot >= 10000.0" in query
    assert "ORDER BY random_index ASC, source_id ASC" in query
    assert "LIMIT 25" in query
