# Тестовый файл `test_load_model_scoring_dataset.py` домена `datasets`.
#
# Этот файл проверяет только:
# - проверку логики домена: loader-слой и shape рабочих dataframe;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `datasets` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from exohost.contracts.feature_contract import ROUTER_FEATURES
from exohost.datasets.load_model_scoring_dataset import (
    build_model_scoring_contract,
    build_model_scoring_query,
)


def test_build_model_scoring_contract_requires_source_id_and_feature_columns() -> None:
    # Контракт model-scoring должен требовать только source_id и признаки модели.
    contract = build_model_scoring_contract(
        "lab.v_gaia_random_stars",
        feature_columns=ROUTER_FEATURES,
    )

    assert contract.relation_name == "lab.v_gaia_random_stars"
    assert contract.required_columns[0] == "source_id"
    assert "teff_gspphot" in contract.required_columns
    assert "parallax" in contract.required_columns
    assert "random_index" in contract.optional_columns


def test_build_model_scoring_query_contains_non_null_feature_filters() -> None:
    # В query должны стоять non-null проверки по всем признакам модели.
    query = build_model_scoring_query(
        "lab.v_gaia_random_stars",
        ("source_id", *ROUTER_FEATURES, "random_index"),
        feature_columns=ROUTER_FEATURES,
        limit=25,
    )

    assert "FROM lab.v_gaia_random_stars" in query
    assert "teff_gspphot IS NOT NULL" in query
    assert "mh_gspphot IS NOT NULL" in query
    assert "ORDER BY random_index ASC, source_id ASC" in query
    assert "LIMIT 25" in query
