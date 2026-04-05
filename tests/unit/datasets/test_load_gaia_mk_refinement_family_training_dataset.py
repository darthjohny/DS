# Тестовый файл `test_load_gaia_mk_refinement_family_training_dataset.py` домена `datasets`.
#
# Этот файл проверяет только:
# - проверку логики домена: loader-слой и shape рабочих dataframe;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `datasets` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from exohost.contracts.refinement_family_dataset_contracts import (
    build_gaia_mk_refinement_family_training_contract,
)
from exohost.datasets.load_gaia_mk_refinement_family_training_dataset import (
    build_gaia_mk_refinement_family_training_query,
)


def test_build_gaia_mk_refinement_family_training_query_contains_expected_shape() -> None:
    contract = build_gaia_mk_refinement_family_training_contract("G")

    query = build_gaia_mk_refinement_family_training_query(
        contract.relation_name,
        contract.required_columns,
        limit=50,
    )

    assert "FROM lab.v_gaia_mk_refinement_training_g" in query
    assert "spectral_subclass IS NOT NULL" in query
    assert "ORDER BY source_id ASC" in query
    assert "LIMIT 50" in query


def test_build_gaia_mk_refinement_family_training_query_prefers_random_index_order() -> None:
    query = build_gaia_mk_refinement_family_training_query(
        "lab.v_gaia_mk_refinement_training_a",
        ("source_id", "random_index", "spectral_subclass"),
        limit=None,
    )

    assert "ORDER BY random_index ASC, source_id ASC" in query
