"""DB-backed интеграционный тест для боевого pipeline на временной схеме."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine

import priority_pipeline.pipeline as pipeline
from priority_pipeline.constants import PRIORITY_RESULTS_COLUMNS, ROUTER_RESULTS_COLUMNS


def _prepare_result_tables_with_constraints(
    engine: Engine,
    schema_name: str,
    router_table_name: str,
    priority_table_name: str,
) -> None:
    """Создать result-таблицы с CHECK constraints, близкими к production."""
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {schema_name}.{priority_table_name}"))
        conn.execute(text(f"DROP TABLE IF EXISTS {schema_name}.{router_table_name}"))
        conn.execute(
            text(
                f"""
                CREATE TABLE {schema_name}.{router_table_name} (
                    run_id text NOT NULL,
                    source_id bigint NOT NULL,
                    ra double precision NOT NULL,
                    dec double precision NOT NULL,
                    teff_gspphot double precision,
                    logg_gspphot double precision,
                    radius_gspphot double precision,
                    predicted_spec_class text NOT NULL,
                    predicted_evolution_stage text NOT NULL,
                    router_label text NOT NULL,
                    d_mahal_router double precision,
                    router_similarity double precision,
                    router_log_likelihood double precision,
                    router_log_posterior double precision,
                    second_best_label text,
                    margin double precision,
                    posterior_margin double precision,
                    router_model_version text NOT NULL,
                    CONSTRAINT pk_{router_table_name}
                        PRIMARY KEY (run_id, source_id),
                    CONSTRAINT chk_{router_table_name}_spec_class
                        CHECK (
                            predicted_spec_class IN (
                                'A', 'B', 'F', 'G', 'K', 'M', 'O', 'UNKNOWN'
                            )
                        ),
                    CONSTRAINT chk_{router_table_name}_evolution_stage
                        CHECK (
                            predicted_evolution_stage IN (
                                'dwarf', 'evolved', 'unknown'
                            )
                        ),
                    CONSTRAINT chk_{router_table_name}_similarity
                        CHECK (
                            router_similarity IS NULL
                            OR (
                                router_similarity >= 0.0
                                AND router_similarity <= 1.0
                            )
                        ),
                    CONSTRAINT chk_{router_table_name}_posterior_margin
                        CHECK (
                            posterior_margin IS NULL
                            OR posterior_margin >= 0.0
                        )
                )
                """
            )
        )
        conn.execute(
            text(
                f"""
                CREATE TABLE {schema_name}.{priority_table_name} (
                    run_id text NOT NULL,
                    source_id bigint NOT NULL,
                    ra double precision NOT NULL,
                    dec double precision NOT NULL,
                    predicted_spec_class text NOT NULL,
                    predicted_evolution_stage text NOT NULL,
                    router_label text NOT NULL,
                    d_mahal_router double precision,
                    router_similarity double precision,
                    router_log_likelihood double precision,
                    router_log_posterior double precision,
                    gauss_label text,
                    host_log_likelihood double precision,
                    field_log_likelihood double precision,
                    host_log_lr double precision,
                    host_posterior double precision,
                    d_mahal double precision,
                    similarity double precision,
                    class_prior double precision,
                    quality_factor double precision,
                    reliability_factor double precision,
                    followup_factor double precision,
                    metallicity_factor double precision,
                    color_factor double precision,
                    validation_factor double precision,
                    final_score double precision,
                    priority_tier text NOT NULL,
                    reason_code text NOT NULL,
                    posterior_margin double precision,
                    router_model_version text NOT NULL,
                    host_model_version text,
                    CONSTRAINT pk_{priority_table_name}
                        PRIMARY KEY (run_id, source_id),
                    CONSTRAINT chk_{priority_table_name}_spec_class
                        CHECK (
                            predicted_spec_class IN (
                                'A', 'B', 'F', 'G', 'K', 'M', 'O', 'UNKNOWN'
                            )
                        ),
                    CONSTRAINT chk_{priority_table_name}_evolution_stage
                        CHECK (
                            predicted_evolution_stage IN (
                                'dwarf', 'evolved', 'unknown'
                            )
                        ),
                    CONSTRAINT chk_{priority_table_name}_tier
                        CHECK (
                            priority_tier IN ('HIGH', 'MEDIUM', 'LOW')
                        ),
                    CONSTRAINT chk_{priority_table_name}_similarity
                        CHECK (
                            similarity IS NULL
                            OR (similarity >= 0.0 AND similarity <= 1.0)
                        ),
                    CONSTRAINT chk_{priority_table_name}_router_similarity
                        CHECK (
                            router_similarity IS NULL
                            OR (
                                router_similarity >= 0.0
                                AND router_similarity <= 1.0
                            )
                        ),
                    CONSTRAINT chk_{priority_table_name}_final_score
                        CHECK (
                            final_score IS NULL
                            OR (final_score >= 0.0 AND final_score <= 1.0)
                        ),
                    CONSTRAINT chk_{priority_table_name}_posterior_margin
                        CHECK (
                            posterior_margin IS NULL
                            OR posterior_margin >= 0.0
                        ),
                    CONSTRAINT chk_{priority_table_name}_host_posterior
                        CHECK (
                            host_posterior IS NULL
                            OR (
                                host_posterior >= 0.0
                                AND host_posterior <= 1.0
                            )
                        )
                )
                """
            )
        )


@pytest.mark.db_integration
def test_run_pipeline_persists_into_temporary_schema(
    monkeypatch: pytest.MonkeyPatch,
    postgres_test_engine: Engine,
    temp_pg_schema: str,
) -> None:
    """`run_pipeline(..., persist=True)` должен реально читать и писать в Postgres."""
    input_table = "gaia_input_candidates"
    router_table = "gaia_router_results"
    priority_table = "gaia_priority_results"
    input_relation = f"{temp_pg_schema}.{input_table}"
    router_relation = f"{temp_pg_schema}.{router_table}"
    priority_relation = f"{temp_pg_schema}.{priority_table}"

    df_input = pd.DataFrame(
        [
            {
                "source_id": 101,
                "ra": 10.0,
                "dec": -5.0,
                "parallax": 12.0,
                "parallax_over_error": 8.0,
                "ruwe": 1.30,
                "bp_rp": 1.20,
                "teff_gspphot": 4700.0,
                "logg_gspphot": 4.60,
                "radius_gspphot": 0.78,
                "mh_gspphot": 0.05,
                "validation_factor": 0.90,
            },
            {
                "source_id": 101,
                "ra": 10.5,
                "dec": -4.8,
                "parallax": 15.0,
                "parallax_over_error": 25.0,
                "ruwe": 0.95,
                "bp_rp": 1.32,
                "teff_gspphot": 4690.0,
                "logg_gspphot": 4.62,
                "radius_gspphot": 0.77,
                "mh_gspphot": 0.08,
                "validation_factor": 1.00,
            },
            {
                "source_id": 202,
                "ra": 20.0,
                "dec": 1.0,
                "parallax": 9.0,
                "parallax_over_error": 12.0,
                "ruwe": 1.10,
                "bp_rp": 0.25,
                "teff_gspphot": 8200.0,
                "logg_gspphot": 4.10,
                "radius_gspphot": 1.90,
                "mh_gspphot": -0.10,
                "validation_factor": 1.00,
            },
            {
                "source_id": 303,
                "ra": 30.0,
                "dec": 4.0,
                "parallax": 11.0,
                "parallax_over_error": 14.0,
                "ruwe": 1.05,
                "bp_rp": 1.10,
                "teff_gspphot": 5600.0,
                "logg_gspphot": 3.20,
                "radius_gspphot": 4.10,
                "mh_gspphot": 0.00,
                "validation_factor": 1.00,
            },
            {
                "source_id": 404,
                "ra": 40.0,
                "dec": 7.5,
                "parallax": 6.0,
                "parallax_over_error": 5.5,
                "ruwe": 1.40,
                "bp_rp": 0.85,
                "teff_gspphot": 6100.0,
                "logg_gspphot": 4.05,
                "radius_gspphot": 1.25,
                "mh_gspphot": -0.05,
                "validation_factor": 1.00,
            },
        ]
    )
    df_input.to_sql(
        name=input_table,
        schema=temp_pg_schema,
        con=postgres_test_engine,
        if_exists="replace",
        index=False,
        method="multi",
    )
    _prepare_result_tables_with_constraints(
        engine=postgres_test_engine,
        schema_name=temp_pg_schema,
        router_table_name=router_table,
        priority_table_name=priority_table,
    )

    def fake_load_models(
        router_model_path: object,
        host_model_path: object,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return (
            {"meta": {"model_version": "gaussian_router_test_v1"}},
            {
                "meta": {
                    "model_version": "gaussian_host_field_test_v1",
                    "score_mode": "host_vs_field_log_lr_v1",
                    "shrink_alpha": 0.15,
                    "use_m_subclasses": True,
                }
            },
        )

    def fake_run_router(
        df: pd.DataFrame,
        router_model: dict[str, Any],
    ) -> pd.DataFrame:
        assert df["source_id"].tolist() == [101, 202, 303, 404]

        selected = df.loc[df["source_id"] == 101].iloc[0]
        assert float(selected["ra"]) == 10.5
        assert float(selected["parallax_over_error"]) == 25.0
        assert float(selected["ruwe"]) == 0.95

        result = df.copy()
        result["predicted_spec_class"] = ["K", "A", "G", "UNKNOWN"]
        result["predicted_evolution_stage"] = [
            "dwarf",
            "dwarf",
            "evolved",
            "unknown",
        ]
        result["router_label"] = [
            "K_dwarf",
            "A_dwarf",
            "G_evolved",
            "UNKNOWN",
        ]
        result["second_best_label"] = [
            "G_dwarf",
            "F_dwarf",
            "K_dwarf",
            "UNKNOWN",
        ]
        result["d_mahal_router"] = [0.20, 0.90, 0.70, float("nan")]
        result["router_similarity"] = [0.95, 0.10, 0.20, 0.0]
        result["router_log_likelihood"] = [-0.10, -1.90, -1.10, float("nan")]
        result["router_log_posterior"] = [-0.10, -1.90, -1.10, float("nan")]
        result["margin"] = [0.50, 0.10, 0.20, float("nan")]
        result["posterior_margin"] = [0.60, 0.15, 0.25, float("nan")]
        result["router_model_version"] = router_model["meta"]["model_version"]
        return result

    def fake_run_host_similarity(
        df_host: pd.DataFrame,
        host_model: dict[str, Any],
    ) -> pd.DataFrame:
        assert df_host["source_id"].tolist() == [101]

        result = df_host.copy()
        result["gauss_label"] = "K"
        result["host_log_likelihood"] = -0.40
        result["field_log_likelihood"] = -1.10
        result["host_log_lr"] = 0.70
        result["host_posterior"] = 0.82
        result["d_mahal"] = None
        result["similarity"] = None
        result["class_prior"] = 0.95
        result["quality_factor"] = 0.97
        result["reliability_factor"] = 0.97
        result["followup_factor"] = 0.92
        result["metallicity_factor"] = 1.02
        result["color_factor"] = 1.01
        result["validation_factor"] = 1.00
        result["final_score"] = 0.77
        result["priority_tier"] = "HIGH"
        result["reason_code"] = "HOST_SCORING"
        result["host_model_version"] = host_model["meta"]["model_version"]
        return result

    monkeypatch.setattr(pipeline, "load_models", fake_load_models)
    monkeypatch.setattr(pipeline, "run_router", fake_run_router)
    monkeypatch.setattr(
        pipeline,
        "run_host_similarity",
        fake_run_host_similarity,
    )
    monkeypatch.setattr(pipeline, "make_run_id", lambda: "db_it_run_1")

    result = pipeline.run_pipeline(
        engine=postgres_test_engine,
        input_source=input_relation,
        limit=None,
        persist=True,
        router_results_table=router_relation,
        priority_results_table=priority_relation,
    )

    assert result.run_id == "db_it_run_1"
    assert len(result.router_results) == 4
    assert len(result.priority_results) == 4

    persisted_router = pd.read_sql(
        (
            f"SELECT * FROM {router_relation} "
            "ORDER BY source_id ASC"
        ),
        postgres_test_engine,
    )
    persisted_priority = pd.read_sql(
        (
            f"SELECT * FROM {priority_relation} "
            "ORDER BY final_score DESC, router_similarity DESC NULLS LAST"
        ),
        postgres_test_engine,
    )

    assert list(persisted_router.columns) == list(ROUTER_RESULTS_COLUMNS)
    assert list(persisted_priority.columns) == list(PRIORITY_RESULTS_COLUMNS)

    assert persisted_router["run_id"].tolist() == ["db_it_run_1"] * 4
    assert persisted_router["source_id"].tolist() == [101, 202, 303, 404]
    assert persisted_router["router_label"].tolist() == [
        "K_dwarf",
        "A_dwarf",
        "G_evolved",
        "UNKNOWN",
    ]
    assert persisted_router["predicted_spec_class"].tolist() == [
        "K",
        "A",
        "G",
        "UNKNOWN",
    ]

    assert persisted_priority["run_id"].tolist() == ["db_it_run_1"] * 4
    assert persisted_priority["source_id"].tolist() == [101, 303, 202, 404]
    assert persisted_priority["priority_tier"].tolist() == [
        "HIGH",
        "LOW",
        "LOW",
        "LOW",
    ]
    assert persisted_priority["reason_code"].tolist() == [
        "HOST_SCORING",
        "EVOLVED_STAR",
        "HOT_STAR",
        "ROUTER_UNKNOWN",
    ]
    host_posterior_high: Any = persisted_priority.loc[0, "host_posterior"]
    host_posterior_low_evolved: Any = persisted_priority.loc[1, "host_posterior"]
    host_posterior_low_hot: Any = persisted_priority.loc[2, "host_posterior"]
    host_posterior_unknown: Any = persisted_priority.loc[3, "host_posterior"]

    assert float(host_posterior_high) == 0.82
    assert pd.isna(host_posterior_low_evolved)
    assert pd.isna(host_posterior_low_hot)
    assert pd.isna(host_posterior_unknown)
