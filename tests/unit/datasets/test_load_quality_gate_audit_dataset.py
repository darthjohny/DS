# Тестовый файл `test_load_quality_gate_audit_dataset.py` домена `datasets`.
#
# Этот файл проверяет только:
# - проверку логики домена: loader-слой и shape рабочих dataframe;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `datasets` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from exohost.contracts.quality_gate_dataset_contracts import (
    GAIA_MK_QUALITY_GATE_AUDIT_CONTRACT,
)
from exohost.datasets.load_quality_gate_audit_dataset import (
    build_quality_gate_audit_query,
)


def test_build_quality_gate_audit_query_contains_required_filters() -> None:
    query = build_quality_gate_audit_query(
        GAIA_MK_QUALITY_GATE_AUDIT_CONTRACT.relation_name,
        GAIA_MK_QUALITY_GATE_AUDIT_CONTRACT.required_columns + ("random_index",),
        limit=125,
    )

    assert "FROM lab.gaia_mk_quality_gated" in query
    assert "quality_state IS NOT NULL" in query
    assert "ood_state IS NOT NULL" in query
    assert "ORDER BY random_index ASC, quality_state ASC, source_id ASC" in query
    assert "LIMIT 125" in query


def test_build_quality_gate_audit_query_falls_back_to_quality_state_order() -> None:
    query = build_quality_gate_audit_query(
        GAIA_MK_QUALITY_GATE_AUDIT_CONTRACT.relation_name,
        GAIA_MK_QUALITY_GATE_AUDIT_CONTRACT.required_columns,
    )

    assert "ORDER BY quality_state ASC, source_id ASC" in query
