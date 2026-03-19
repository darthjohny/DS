"""Экспорт versioned operational-артефактов production pipeline.

Модуль не запускает сравнение моделей и не смешивает research snapshot
с боевым результатом. Он получает `PipelineRunResult` от `run_pipeline()`
и сохраняет отдельный operational-layer:

- router-таблицу;
- итоговую priority-таблицу;
- summary по tier;
- summary по классам;
- shortlist для follow-up наблюдений;
- markdown-сводку запуска.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Engine

from priority_pipeline.constants import DEFAULT_INPUT_SOURCE, PROJECT_ROOT
from priority_pipeline.contracts import PipelineRunResult
from priority_pipeline.pipeline import run_pipeline
from router_model import make_engine_from_env

DEFAULT_OPERATIONAL_EXPORT_DIR = (
    PROJECT_ROOT / "experiments" / "QA" / "production_runs"
)
SHORTLIST_PRIORITY_MAP: dict[str, int] = {
    "K": 1,
    "M": 2,
    "G": 3,
}
OPERATIONAL_SHORTLIST_COLUMNS: tuple[str, ...] = (
    "observation_priority",
    "rank_in_priority",
    "source_id",
    "ra",
    "dec",
    "predicted_spec_class",
    "predicted_evolution_stage",
    "host_like_percent",
    "host_like_profile",
    "final_score",
    "reason_code",
)


def default_operational_run_name(*, limit: int | None) -> str:
    """Собрать versioned имя operational run без привязки к comparison-layer."""
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d")
    suffix = f"limit{limit}" if limit is not None else "full"
    return f"production_priority_{timestamp}_v1_calibrated_{suffix}"


def empty_operational_shortlist_frame() -> pd.DataFrame:
    """Вернуть пустой shortlist frame в каноническом column-order."""
    return pd.DataFrame(columns=OPERATIONAL_SHORTLIST_COLUMNS)


def build_tier_summary_frame(df_priority: pd.DataFrame) -> pd.DataFrame:
    """Собрать краткую сводку по HIGH/MEDIUM/LOW слоям."""
    rows: list[dict[str, object]] = []
    for tier in ("HIGH", "MEDIUM", "LOW"):
        tier_df = df_priority[df_priority["priority_tier"] == tier].copy()
        rows.append(
            {
                "priority_tier": tier,
                "n_rows": int(tier_df.shape[0]),
                "top_final_score": float(tier_df["final_score"].max())
                if not tier_df.empty
                else float("nan"),
            }
        )
    return pd.DataFrame.from_records(rows)


def build_class_summary_frame(df_priority: pd.DataFrame) -> pd.DataFrame:
    """Собрать summary по спектральным классам и tier-распределению."""
    if df_priority.empty:
        return pd.DataFrame(
            columns=[
                "predicted_spec_class",
                "n_rows",
                "high_rows",
                "medium_rows",
                "low_rows",
                "top_final_score",
            ]
        )

    rows: list[dict[str, object]] = []
    grouped = df_priority.groupby("predicted_spec_class", dropna=False)
    for spec_class, class_df in grouped:
        rows.append(
            {
                "predicted_spec_class": spec_class,
                "n_rows": int(class_df.shape[0]),
                "high_rows": int(class_df["priority_tier"].eq("HIGH").sum()),
                "medium_rows": int(
                    class_df["priority_tier"].eq("MEDIUM").sum()
                ),
                "low_rows": int(class_df["priority_tier"].eq("LOW").sum()),
                "top_final_score": float(class_df["final_score"].max()),
            }
        )
    return (
        pd.DataFrame.from_records(rows)
        .sort_values(["high_rows", "medium_rows", "top_final_score"], ascending=False)
        .reset_index(drop=True)
    )


def build_operational_shortlist(
    df_priority: pd.DataFrame,
) -> pd.DataFrame:
    """Собрать production shortlist из канонического runtime-результата.

    Для текущей `V1` shortlist строится только по operational `HIGH`
    из целевых dwarf-классов, без опоры на comparison snapshot.
    """
    if df_priority.empty:
        return empty_operational_shortlist_frame()

    candidate_df = df_priority.loc[
        (df_priority["priority_tier"] == "HIGH")
        & (df_priority["predicted_evolution_stage"] == "dwarf")
        & (df_priority["predicted_spec_class"].isin(SHORTLIST_PRIORITY_MAP))
    ].copy()

    if candidate_df.empty:
        return empty_operational_shortlist_frame()

    candidate_df["source_id"] = candidate_df["source_id"].astype(int)
    candidate_df["observation_priority"] = candidate_df[
        "predicted_spec_class"
    ].map(SHORTLIST_PRIORITY_MAP)
    candidate_df["host_like_percent"] = (
        candidate_df["host_posterior"].astype(float) * 100.0
    ).round(2)
    candidate_df["host_like_profile"] = candidate_df["gauss_label"].fillna("-")

    shortlist_df = candidate_df.sort_values(
        ["observation_priority", "final_score", "host_posterior"],
        ascending=[True, False, False],
        ignore_index=True,
    ).copy()
    shortlist_df["rank_in_priority"] = (
        shortlist_df.groupby("observation_priority").cumcount() + 1
    )
    return shortlist_df.loc[:, OPERATIONAL_SHORTLIST_COLUMNS].copy()


def build_shortlist_summary_frame(df_shortlist: pd.DataFrame) -> pd.DataFrame:
    """Собрать компактную summary по physical priorities shortlist-а."""
    if df_shortlist.empty:
        return pd.DataFrame(
            columns=["observation_priority", "n_rows"]
        )
    return (
        df_shortlist.groupby("observation_priority")
        .size()
        .to_frame("n_rows")
        .reset_index()
    )


def build_operational_markdown(
    *,
    run_name: str,
    input_source: str,
    limit: int | None,
    result: PipelineRunResult,
    tier_summary_df: pd.DataFrame,
    class_summary_df: pd.DataFrame,
    shortlist_summary_df: pd.DataFrame,
    note: str,
) -> str:
    """Собрать markdown-сводку канонического production run."""
    created_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    limit_text = "full" if limit is None else str(limit)
    note_text = note.strip() or "-"

    sections = [
        f"# {run_name}",
        "",
        f"Дата: {created_at}",
        f"Источник: `{input_source}`",
        f"limit: `{limit_text}`",
        f"run_id: `{result.run_id}`",
        "",
        "## Operational semantics",
        "- Это production-like result `run_pipeline()`.",
        "- Этот артефакт не является comparison snapshot.",
        "- Shortlist строится только из runtime `priority_results`.",
        "",
        "## Tier summary",
        tier_summary_df.to_string(index=False)
        if not tier_summary_df.empty
        else "Пусто",
        "",
        "## Class summary",
        class_summary_df.to_string(index=False)
        if not class_summary_df.empty
        else "Пусто",
        "",
        "## Shortlist summary",
        shortlist_summary_df.to_string(index=False)
        if not shortlist_summary_df.empty
        else "Пусто",
        "",
        "## Note",
        note_text,
    ]
    return "\n".join(sections)


def save_operational_artifacts(
    *,
    run_name: str,
    input_source: str,
    limit: int | None,
    result: PipelineRunResult,
    output_dir: Path = DEFAULT_OPERATIONAL_EXPORT_DIR,
    note: str = "",
) -> Path:
    """Сохранить versioned production artifact в отдельный operational-контур."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / run_name

    tier_summary_df = build_tier_summary_frame(result.priority_results)
    class_summary_df = build_class_summary_frame(result.priority_results)
    shortlist_df = build_operational_shortlist(result.priority_results)
    shortlist_summary_df = build_shortlist_summary_frame(shortlist_df)

    router_path = prefix.with_name(f"{prefix.name}_router.csv")
    priority_path = prefix.with_name(f"{prefix.name}_priority.csv")
    tier_summary_path = prefix.with_name(f"{prefix.name}_tier_summary.csv")
    class_summary_path = prefix.with_name(f"{prefix.name}_class_summary.csv")
    shortlist_path = prefix.with_name(f"{prefix.name}_shortlist.csv")
    shortlist_summary_path = prefix.with_name(
        f"{prefix.name}_shortlist_summary.csv"
    )
    markdown_path = prefix.with_suffix(".md")

    result.router_results.to_csv(router_path, index=False)
    result.priority_results.to_csv(priority_path, index=False)
    tier_summary_df.to_csv(tier_summary_path, index=False)
    class_summary_df.to_csv(class_summary_path, index=False)
    shortlist_df.to_csv(shortlist_path, index=False)
    shortlist_summary_df.to_csv(shortlist_summary_path, index=False)
    markdown_path.write_text(
        build_operational_markdown(
            run_name=run_name,
            input_source=input_source,
            limit=limit,
            result=result,
            tier_summary_df=tier_summary_df,
            class_summary_df=class_summary_df,
            shortlist_summary_df=shortlist_summary_df,
            note=note,
        ),
        encoding="utf-8",
    )
    return markdown_path


def parse_args() -> Namespace:
    """Разобрать CLI-аргументы operational export слоя."""
    parser = ArgumentParser(
        description="Запустить production pipeline и сохранить operational artifacts."
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Имя versioned operational run. По умолчанию генерируется автоматически.",
    )
    parser.add_argument(
        "--input-source",
        default=DEFAULT_INPUT_SOURCE,
        help="Relation/view для production-like запуска.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Необязательный LIMIT для входного relation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OPERATIONAL_EXPORT_DIR,
        help="Каталог для operational CSV/markdown артефактов.",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Дополнительно записать runtime results в production DB tables.",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Необязательное примечание к markdown-сводке operational run.",
    )
    return parser.parse_args()


def main() -> None:
    """Запустить production pipeline и сохранить versioned operational artifacts."""
    args = parse_args()
    actual_limit = int(args.limit) if args.limit is not None else None
    run_name = args.run_name or default_operational_run_name(limit=actual_limit)

    engine: Engine = make_engine_from_env()
    result = run_pipeline(
        engine=engine,
        input_source=str(args.input_source),
        limit=actual_limit,
        persist=bool(args.persist),
    )
    markdown_path = save_operational_artifacts(
        run_name=run_name,
        input_source=str(args.input_source),
        limit=actual_limit,
        result=result,
        output_dir=Path(args.output_dir),
        note=str(args.note),
    )
    print("Operational report:", markdown_path)


__all__ = [
    "DEFAULT_OPERATIONAL_EXPORT_DIR",
    "OPERATIONAL_SHORTLIST_COLUMNS",
    "SHORTLIST_PRIORITY_MAP",
    "build_class_summary_frame",
    "build_operational_markdown",
    "build_operational_shortlist",
    "build_shortlist_summary_frame",
    "build_tier_summary_frame",
    "default_operational_run_name",
    "empty_operational_shortlist_frame",
    "main",
    "parse_args",
    "save_operational_artifacts",
]


if __name__ == "__main__":
    main()
