# Тестовый файл `test_load_final_decision_input_dataset.py` домена `datasets`.
#
# Этот файл проверяет только:
# - проверку логики домена: loader-слой и shape рабочих dataframe;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `datasets` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest
from sqlalchemy.engine import Engine

from exohost.datasets import load_final_decision_input_dataset as module


def test_load_final_decision_input_dataset_builds_radius_feature_from_radius_flame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_query: dict[str, str] = {}

    monkeypatch.setattr(module, "relation_exists", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        module,
        "relation_columns",
        lambda *args, **kwargs: (
            "source_id",
            "quality_state",
            "random_index",
            "teff_gspphot",
            "radius_flame",
            "quality_reason",
            "review_bucket",
        ),
    )

    def fake_read_sql(query: str, engine: object) -> pd.DataFrame:
        del engine
        captured_query["query"] = query
        return pd.DataFrame(
            {
                "source_id": ["501"],
                "quality_state": ["pass"],
                "random_index": [7],
                "teff_gspphot": [5770.0],
                "radius_flame": [1.04],
                "quality_reason": ["clean"],
                "review_bucket": ["pass"],
            }
        )

    monkeypatch.setattr(module.pd, "read_sql", fake_read_sql)

    loaded_df = module.load_final_decision_input_dataset(
        cast(Engine, object()),
        relation_name="lab.gaia_mk_quality_gated",
        feature_columns=("teff_gspphot", "radius_feature"),
        limit=10,
    )

    assert "quality_state IS NOT NULL" in captured_query["query"]
    assert "radius_flame IS NOT NULL" not in captured_query["query"]
    assert "radius_feature" not in captured_query["query"]
    assert "quality_reason" in captured_query["query"]
    assert "review_bucket" in captured_query["query"]
    assert loaded_df["radius_feature"].tolist() == [1.04]
    assert loaded_df["quality_reason"].tolist() == ["clean"]
    assert loaded_df["review_bucket"].tolist() == ["pass"]


def test_load_final_decision_input_dataset_builds_radius_gspphot_from_radius_flame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "relation_exists", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        module,
        "relation_columns",
        lambda *args, **kwargs: (
            "source_id",
            "quality_state",
            "teff_gspphot",
            "radius_flame",
        ),
    )
    monkeypatch.setattr(
        module.pd,
        "read_sql",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "source_id": ["601"],
                "quality_state": ["pass"],
                "teff_gspphot": [5050.0],
                "radius_flame": [0.91],
            }
        ),
    )

    loaded_df = module.load_final_decision_input_dataset(
        cast(Engine, object()),
        relation_name="lab.gaia_mk_quality_gated",
        feature_columns=("teff_gspphot", "radius_gspphot"),
    )

    assert loaded_df["radius_gspphot"].tolist() == [0.91]


def test_load_final_decision_input_dataset_raises_when_feature_has_no_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "relation_exists", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        module,
        "relation_columns",
        lambda *args, **kwargs: (
            "source_id",
            "quality_state",
            "teff_gspphot",
        ),
    )

    with pytest.raises(RuntimeError, match="radius_feature"):
        module.load_final_decision_input_dataset(
            cast(Engine, object()),
            relation_name="lab.gaia_mk_quality_gated",
            feature_columns=("teff_gspphot", "radius_feature"),
        )
