"""Operational snapshot-layer для comparison моделей на живом Gaia batch.

Модуль строит сравнительный ranking snapshot на одном и том же входном
relation после общего `router + OOD`. Меняется только scoring head
внутри host-ветки:

- `main_contrastive_v1`
- `baseline_legacy_gaussian`
- `baseline_random_forest`
- `baseline_mlp_small`
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy.engine import Engine

from analysis.model_comparison.contracts import (
    DEFAULT_COMPARISON_PROTOCOL,
    ComparisonProtocol,
)
from analysis.model_comparison.contrastive import (
    MAIN_CONTRASTIVE_MODEL_NAME,
    fit_main_contrastive_model_with_search,
)
from analysis.model_comparison.data import get_engine, load_and_split_benchmark_dataset
from analysis.model_comparison.legacy_gaussian import (
    LEGACY_GAUSSIAN_MODEL_NAME,
    fit_legacy_gaussian_baseline_with_search,
)
from analysis.model_comparison.mlp_baseline import (
    MLP_BASELINE_MODEL_NAME,
    fit_mlp_baseline,
    get_mlp_classifier,
)
from analysis.model_comparison.random_forest import (
    RANDOM_FOREST_MODEL_NAME,
    fit_random_forest_baseline,
)
from host_model.artifacts import ContrastiveGaussianModel, LegacyGaussianModel
from host_model.legacy_score import score_df as score_df_legacy
from priority_pipeline.branching import RouterBranchFrames, split_router_branches
from priority_pipeline.constants import DEFAULT_INPUT_SOURCE, ROUTER_MODEL_PATH
from priority_pipeline.decision import (
    HOST_SCORING_REASON,
    apply_common_factors,
    build_low_priority_stub,
    build_unknown_priority_stub,
    clip_unit_interval,
    host_model_version,
    order_priority_results,
    priority_tier_from_score,
    run_host_similarity,
)
from priority_pipeline.input_data import load_input_candidates
from priority_pipeline.pipeline import run_router
from router_model.artifacts import RouterModel, load_router_model

DEFAULT_SNAPSHOT_OUTPUT_DIR = Path("experiments/model_comparison")
DEFAULT_SNAPSHOT_TOP_K = 50


@dataclass(slots=True)
class SnapshotModelRun:
    """Один operational snapshot для конкретной модели."""

    model_name: str
    priority_df: pd.DataFrame
    top_df: pd.DataFrame


@dataclass(slots=True)
class SnapshotComparisonResult:
    """Полный snapshot comparison на одном входном relation."""

    source_name: str
    input_rows: int
    router_df: pd.DataFrame
    branches: RouterBranchFrames
    model_runs: list[SnapshotModelRun]


def score_snapshot_host_with_scalar_head(
    scored_df: pd.DataFrame,
    score_column: str,
    model_name: str,
    model_version_value: str,
) -> pd.DataFrame:
    """Применить общие decision factors к произвольному scalar scoring head."""
    if scored_df.empty:
        return scored_df.copy()

    result = apply_common_factors(scored_df)
    result["model_name"] = model_name
    result["model_score"] = result[score_column].astype(float)

    scoring_rows = result[
        [
            "model_score",
            "class_prior",
            "quality_factor",
            "metallicity_factor",
            "color_factor",
            "validation_factor",
        ]
    ].itertuples(index=False, name=None)
    result["final_score"] = [
        clip_unit_interval(
            float(model_score)
            * float(class_prior_value)
            * float(quality_value)
            * float(metallicity_value)
            * float(color_value)
            * float(validation_value)
        )
        for (
            model_score,
            class_prior_value,
            quality_value,
            metallicity_value,
            color_value,
            validation_value,
        ) in scoring_rows
    ]
    result["priority_tier"] = [
        priority_tier_from_score(float(score))
        for score in result["final_score"]
    ]
    result["reason_code"] = HOST_SCORING_REASON
    result["host_model_version"] = model_version_value
    return result


def legacy_model_version(model: LegacyGaussianModel) -> str:
    """Собрать компактную версию legacy baseline для snapshot-артефактов."""
    meta = model["meta"]
    return (
        "legacy_gaussian_"
        f"msub_{bool(meta['use_m_subclasses'])}_"
        f"shrink_{float(meta['shrink_alpha']):.2f}"
    )


def random_forest_model_version(models_by_class: dict[str, RandomForestClassifier]) -> str:
    """Собрать компактную строку версии RandomForest baseline."""
    example_model = next(iter(models_by_class.values()))
    params = example_model.get_params(deep=False)
    return (
        "random_forest_"
        f"estimators_{int(params['n_estimators'])}_"
        f"leaf_{int(params['min_samples_leaf'])}_"
        f"seed_{int(params['random_state'])}"
    )


def mlp_model_version(models_by_class: dict[str, Pipeline]) -> str:
    """Собрать компактную строку версии class-specific MLP baseline."""
    example_pipeline = next(iter(models_by_class.values()))
    mlp = get_mlp_classifier(
        example_pipeline,
        source="Snapshot comparison MLP baseline",
    )
    params = mlp.get_params(deep=False)
    hidden_sizes = "-".join(str(size) for size in params["hidden_layer_sizes"])
    return (
        "mlp_small_"
        f"hidden_{hidden_sizes}_"
        f"alpha_{float(params['alpha']):.4f}_"
        f"seed_{int(params['random_state'])}"
    )


def score_snapshot_host_main_contrastive(
    df_host: pd.DataFrame,
    model: ContrastiveGaussianModel,
) -> pd.DataFrame:
    """Скорить host-ветку основной contrastive-моделью."""
    result = run_host_similarity(df_host=df_host, host_model=model)
    if result.empty:
        return result
    result["model_name"] = MAIN_CONTRASTIVE_MODEL_NAME
    result["model_score"] = result["host_posterior"].astype(float)
    return result


def score_snapshot_host_legacy_gaussian(
    df_host: pd.DataFrame,
    model: LegacyGaussianModel,
) -> pd.DataFrame:
    """Скорить host-ветку legacy Gaussian baseline."""
    if df_host.empty:
        return df_host.copy()

    scored = score_df_legacy(
        model=model,
        df=df_host,
        spec_class_col="predicted_spec_class",
    ).copy()
    scored["host_log_likelihood"] = None
    scored["field_log_likelihood"] = None
    scored["host_log_lr"] = None
    scored["host_posterior"] = None
    return score_snapshot_host_with_scalar_head(
        scored_df=scored,
        score_column="similarity",
        model_name=LEGACY_GAUSSIAN_MODEL_NAME,
        model_version_value=legacy_model_version(model),
    )


def score_snapshot_host_random_forest(
    df_host: pd.DataFrame,
    models_by_class: dict[str, RandomForestClassifier],
    *,
    protocol: ComparisonProtocol = DEFAULT_COMPARISON_PROTOCOL,
) -> pd.DataFrame:
    """Скорить host-ветку class-specific RandomForest baseline."""
    if df_host.empty:
        return df_host.copy()

    scored_parts: list[pd.DataFrame] = []
    for spec_class in protocol.sources.allowed_classes:
        class_df = df_host[df_host["predicted_spec_class"] == spec_class].copy()
        if class_df.empty:
            continue
        estimator = models_by_class[spec_class]
        features = class_df[list(protocol.sources.feature_columns)].to_numpy()
        positive_proba = estimator.predict_proba(features)[:, 1]
        class_df["gauss_label"] = None
        class_df["host_log_likelihood"] = None
        class_df["field_log_likelihood"] = None
        class_df["host_log_lr"] = None
        class_df["host_posterior"] = None
        class_df["d_mahal"] = None
        class_df["similarity"] = None
        class_df["rf_positive_proba"] = positive_proba.astype(float)
        class_df["rf_predicted_is_host"] = [
            bool(value) for value in estimator.predict(features)
        ]
        scored_parts.append(class_df)

    if not scored_parts:
        return df_host.iloc[0:0].copy()

    scored = pd.concat(scored_parts, ignore_index=True, sort=False)
    return score_snapshot_host_with_scalar_head(
        scored_df=scored,
        score_column="rf_positive_proba",
        model_name=RANDOM_FOREST_MODEL_NAME,
        model_version_value=random_forest_model_version(models_by_class),
    )


def score_snapshot_host_mlp(
    df_host: pd.DataFrame,
    models_by_class: dict[str, Pipeline],
    *,
    protocol: ComparisonProtocol = DEFAULT_COMPARISON_PROTOCOL,
) -> pd.DataFrame:
    """Скорить host-ветку class-specific MLP baseline."""
    if df_host.empty:
        return df_host.copy()

    scored_parts: list[pd.DataFrame] = []
    for spec_class in protocol.sources.allowed_classes:
        class_df = df_host[df_host["predicted_spec_class"] == spec_class].copy()
        if class_df.empty:
            continue
        estimator = models_by_class[spec_class]
        features = class_df[list(protocol.sources.feature_columns)].to_numpy()
        positive_proba = estimator.predict_proba(features)[:, 1]
        class_df["gauss_label"] = None
        class_df["host_log_likelihood"] = None
        class_df["field_log_likelihood"] = None
        class_df["host_log_lr"] = None
        class_df["host_posterior"] = None
        class_df["d_mahal"] = None
        class_df["similarity"] = None
        class_df["rf_positive_proba"] = None
        class_df["mlp_positive_proba"] = positive_proba.astype(float)
        class_df["mlp_predicted_is_host"] = [
            bool(value) for value in estimator.predict(features)
        ]
        scored_parts.append(class_df)

    if not scored_parts:
        return df_host.iloc[0:0].copy()

    scored = pd.concat(scored_parts, ignore_index=True, sort=False)
    return score_snapshot_host_with_scalar_head(
        scored_df=scored,
        score_column="mlp_positive_proba",
        model_name=MLP_BASELINE_MODEL_NAME,
        model_version_value=mlp_model_version(models_by_class),
    )


def attach_snapshot_metadata(
    df_priority: pd.DataFrame,
    *,
    model_name: str,
    model_version_value: str | None,
) -> pd.DataFrame:
    """Добавить общие metadata-поля ко всей snapshot-таблице модели."""
    if df_priority.empty:
        return df_priority.copy()

    result = df_priority.copy()
    result["model_name"] = model_name
    if "model_score" not in result.columns:
        result["model_score"] = None
    if model_version_value is not None:
        if "host_model_version" not in result.columns:
            result["host_model_version"] = model_version_value
        else:
            result["host_model_version"] = result["host_model_version"].fillna(
                model_version_value
            )
    return result


def build_snapshot_top_frame(df_priority: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Собрать top-k таблицу snapshot-результата."""
    columns = [
        "source_id",
        "predicted_spec_class",
        "predicted_evolution_stage",
        "router_label",
        "gauss_label",
        "model_score",
        "host_posterior",
        "host_log_lr",
        "similarity",
        "rf_positive_proba",
        "mlp_positive_proba",
        "final_score",
        "priority_tier",
        "reason_code",
        "router_similarity",
        "router_log_posterior",
        "posterior_margin",
        "ra",
        "dec",
        "teff_gspphot",
        "logg_gspphot",
        "radius_gspphot",
    ]
    existing = [column for column in columns if column in df_priority.columns]
    return df_priority.loc[:, existing].head(top_k).copy()


def build_snapshot_summary_frame(result: SnapshotComparisonResult) -> pd.DataFrame:
    """Собрать summary по всем моделям в snapshot comparison."""
    rows: list[dict[str, object]] = []
    router_rows = int(result.router_df.shape[0])
    host_candidates = int(result.branches.host_df.shape[0])
    low_known_rows = int(result.branches.low_known_df.shape[0])
    unknown_rows = int(result.branches.unknown_df.shape[0])

    for model_run in result.model_runs:
        df_priority = model_run.priority_df
        rows.append(
            {
                "model_name": model_run.model_name,
                "source_name": result.source_name,
                "input_rows": result.input_rows,
                "router_rows": router_rows,
                "host_candidates": host_candidates,
                "low_known_rows": low_known_rows,
                "unknown_rows": unknown_rows,
                "high_rows": int(df_priority["priority_tier"].eq("HIGH").sum()),
                "medium_rows": int(df_priority["priority_tier"].eq("MEDIUM").sum()),
                "low_rows": int(df_priority["priority_tier"].eq("LOW").sum()),
                "top_final_score": float(df_priority["final_score"].max())
                if not df_priority.empty
                else float("nan"),
            }
        )

    return pd.DataFrame.from_records(rows)


def frame_to_text(df: pd.DataFrame) -> str:
    """Преобразовать DataFrame в компактный текстовый блок markdown."""
    if df.empty:
        return "Пусто"
    return df.to_string(index=False)


def build_snapshot_markdown(
    result: SnapshotComparisonResult,
    summary_df: pd.DataFrame,
    *,
    top_k: int,
    note: str = "",
) -> str:
    """Собрать markdown-отчёт для snapshot comparison."""
    created_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    note_text = note.strip() or "-"
    sections = [
        "# Model Comparison Snapshot",
        "",
        f"Дата: {created_at}",
        f"Source: `{result.source_name}`",
        f"input_rows: `{result.input_rows}`",
        f"router_rows: `{len(result.router_df)}`",
        f"host_candidates: `{len(result.branches.host_df)}`",
        f"low_known_rows: `{len(result.branches.low_known_df)}`",
        f"unknown_rows: `{len(result.branches.unknown_df)}`",
        "",
        "## Summary",
        frame_to_text(summary_df),
    ]

    for model_run in result.model_runs:
        sections.extend(
            [
                "",
                f"## Top-{top_k}: `{model_run.model_name}`",
                frame_to_text(model_run.top_df),
            ]
        )

    sections.extend(
        [
            "",
            "## Примечание",
            note_text,
        ]
    )
    return "\n".join(sections)


def save_snapshot_artifacts(
    run_name: str,
    result: SnapshotComparisonResult,
    *,
    output_dir: Path = DEFAULT_SNAPSHOT_OUTPUT_DIR,
    top_k: int = DEFAULT_SNAPSHOT_TOP_K,
    note: str = "",
) -> Path:
    """Сохранить markdown- и CSV-артефакты snapshot comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"{run_name}_snapshot"
    summary_df = build_snapshot_summary_frame(result)
    markdown_path = prefix.with_suffix(".md")
    summary_path = prefix.with_name(f"{prefix.name}_summary.csv")

    markdown_path.write_text(
        build_snapshot_markdown(
            result,
            summary_df,
            top_k=top_k,
            note=note,
        ),
        encoding="utf-8",
    )
    summary_df.to_csv(summary_path, index=False)

    result.router_df.to_csv(
        prefix.with_name(f"{prefix.name}_router.csv"),
        index=False,
    )
    for model_run in result.model_runs:
        model_run.priority_df.to_csv(
            prefix.with_name(f"{prefix.name}_{model_run.model_name}_priority.csv"),
            index=False,
        )
        model_run.top_df.to_csv(
            prefix.with_name(f"{prefix.name}_{model_run.model_name}_top.csv"),
            index=False,
        )

    return markdown_path


def run_snapshot_comparison(
    *,
    protocol: ComparisonProtocol = DEFAULT_COMPARISON_PROTOCOL,
    source_name: str = DEFAULT_INPUT_SOURCE,
    limit: int | None = None,
    top_k: int = DEFAULT_SNAPSHOT_TOP_K,
    engine: Engine | None = None,
    router_model_path: Path = ROUTER_MODEL_PATH,
) -> SnapshotComparisonResult:
    """Построить snapshot comparison на одном входном Gaia relation."""
    actual_engine = engine or get_engine()
    split = load_and_split_benchmark_dataset(
        engine=actual_engine,
        protocol=protocol,
    )

    main_model, _ = fit_main_contrastive_model_with_search(
        split.train_df,
        sources=protocol.sources,
        cv_config=protocol.cv,
        search_config=protocol.search,
    )
    legacy_model, _ = fit_legacy_gaussian_baseline_with_search(
        split.train_df,
        sources=protocol.sources,
        cv_config=protocol.cv,
        search_config=protocol.search,
    )
    mlp_models = fit_mlp_baseline(
        split.train_df,
        sources=protocol.sources,
        cv_config=protocol.cv,
        search_config=protocol.search,
    )
    rf_models = fit_random_forest_baseline(
        split.train_df,
        sources=protocol.sources,
        cv_config=protocol.cv,
        search_config=protocol.search,
    )

    df_input = load_input_candidates(
        engine=actual_engine,
        source_name=source_name,
        limit=limit,
    )
    router_model: RouterModel = load_router_model(str(router_model_path))
    df_router = run_router(df=df_input, router_model=router_model)
    branches = split_router_branches(df_router)

    low_stub = build_low_priority_stub(branches.low_known_df)
    unknown_stub = build_unknown_priority_stub(branches.unknown_df)

    model_runs: list[SnapshotModelRun] = []
    for model_name, host_df, version_value in (
        (
            MAIN_CONTRASTIVE_MODEL_NAME,
            score_snapshot_host_main_contrastive(branches.host_df, main_model),
            host_model_version(main_model),
        ),
        (
            LEGACY_GAUSSIAN_MODEL_NAME,
            score_snapshot_host_legacy_gaussian(branches.host_df, legacy_model),
            legacy_model_version(legacy_model),
        ),
        (
            RANDOM_FOREST_MODEL_NAME,
            score_snapshot_host_random_forest(
                branches.host_df,
                rf_models,
                protocol=protocol,
            ),
            random_forest_model_version(rf_models),
        ),
        (
            MLP_BASELINE_MODEL_NAME,
            score_snapshot_host_mlp(
                branches.host_df,
                mlp_models,
                protocol=protocol,
            ),
            mlp_model_version(mlp_models),
        ),
    ):
        parts = [
            frame
            for frame in (
                host_df,
                attach_snapshot_metadata(
                    low_stub,
                    model_name=model_name,
                    model_version_value=version_value,
                ),
                attach_snapshot_metadata(
                    unknown_stub,
                    model_name=model_name,
                    model_version_value=version_value,
                ),
            )
            if not frame.empty
        ]
        priority_df = order_priority_results(
            pd.concat(parts, ignore_index=True, sort=False)
        )
        model_runs.append(
            SnapshotModelRun(
                model_name=model_name,
                priority_df=priority_df,
                top_df=build_snapshot_top_frame(priority_df, top_k=top_k),
            )
        )

    return SnapshotComparisonResult(
        source_name=source_name,
        input_rows=int(df_input.shape[0]),
        router_df=df_router,
        branches=branches,
        model_runs=model_runs,
    )


__all__ = [
    "DEFAULT_SNAPSHOT_OUTPUT_DIR",
    "DEFAULT_SNAPSHOT_TOP_K",
    "SnapshotComparisonResult",
    "SnapshotModelRun",
    "attach_snapshot_metadata",
    "build_snapshot_markdown",
    "build_snapshot_summary_frame",
    "build_snapshot_top_frame",
    "mlp_model_version",
    "run_snapshot_comparison",
    "save_snapshot_artifacts",
    "score_snapshot_host_legacy_gaussian",
    "score_snapshot_host_mlp",
    "score_snapshot_host_main_contrastive",
    "score_snapshot_host_random_forest",
]
