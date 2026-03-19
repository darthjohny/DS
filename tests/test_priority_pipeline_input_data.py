"""Тесты current V1-контракта загрузки входных кандидатов pipeline."""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy.engine import Engine

from priority_pipeline.input_data import load_input_candidates


@pytest.mark.db_integration
def test_load_input_candidates_filters_incomplete_rows_before_router(
    postgres_test_engine: Engine,
    temp_pg_schema: str,
) -> None:
    """Structurally incomplete строки не должны доходить до router в current V1."""
    relation_name = f"{temp_pg_schema}.gaia_input_candidates"
    df = pd.DataFrame(
        [
            {
                "source_id": 1,
                "ra": 10.0,
                "dec": -5.0,
                "teff_gspphot": 4700.0,
                "logg_gspphot": 4.60,
                "radius_gspphot": 0.80,
                "mh_gspphot": 0.10,
                "parallax": 12.0,
                "parallax_over_error": 18.0,
                "ruwe": 1.10,
            },
            {
                "source_id": 1,
                "ra": 10.1,
                "dec": -4.9,
                "teff_gspphot": 4695.0,
                "logg_gspphot": 4.58,
                "radius_gspphot": 0.79,
                "mh_gspphot": 0.10,
                "parallax": 13.0,
                "parallax_over_error": 24.0,
                "ruwe": 0.95,
            },
            {
                "source_id": 2,
                "ra": 20.0,
                "dec": 1.0,
                "teff_gspphot": None,
                "logg_gspphot": 4.30,
                "radius_gspphot": 1.00,
                "mh_gspphot": -0.10,
                "parallax": 8.0,
                "parallax_over_error": 12.0,
                "ruwe": 1.10,
            },
            {
                "source_id": 3,
                "ra": 30.0,
                "dec": 2.0,
                "teff_gspphot": 5600.0,
                "logg_gspphot": 4.20,
                "radius_gspphot": None,
                "mh_gspphot": 0.00,
                "parallax": 9.0,
                "parallax_over_error": 10.0,
                "ruwe": 1.00,
            },
            {
                "source_id": 4,
                "ra": 40.0,
                "dec": 3.0,
                "teff_gspphot": 5100.0,
                "logg_gspphot": 4.40,
                "radius_gspphot": 0.95,
                "mh_gspphot": 0.05,
                "parallax": 11.0,
                "parallax_over_error": 15.0,
                "ruwe": 1.05,
            },
        ]
    )
    df.to_sql(
        name="gaia_input_candidates",
        schema=temp_pg_schema,
        con=postgres_test_engine,
        if_exists="replace",
        index=False,
        method="multi",
    )

    loaded = load_input_candidates(
        engine=postgres_test_engine,
        source_name=relation_name,
    )
    ra_values = loaded["ra"].to_numpy(dtype=float)
    parallax_error_values = loaded["parallax_over_error"].to_numpy(dtype=float)
    ruwe_values = loaded["ruwe"].to_numpy(dtype=float)

    assert loaded["source_id"].tolist() == [1, 4]
    assert ra_values[0] == 10.1
    assert parallax_error_values[0] == 24.0
    assert ruwe_values[0] == 0.95
    assert "bp_rp" in loaded.columns
    assert "validation_factor" in loaded.columns
    assert loaded["validation_factor"].tolist() == [1.0, 1.0]
    assert loaded["bp_rp"].isna().all()
