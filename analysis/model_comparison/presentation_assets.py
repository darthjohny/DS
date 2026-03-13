"""Сборка slide-ready ассетов по comparison-artifacts.

Что делает модуль:
    - читает benchmark и snapshot CSV-артефакты comparative benchmark;
    - формирует компактные таблицы для защиты и записки;
    - сохраняет PNG-графики для benchmark, classwise и snapshot-сводок.

Где находится основная логика или откуда приходят данные:
    - входные CSV берутся из experiments/model_comparison;
    - выходные PNG и CSV сохраняются в docs/presentation/assets/<run_name>.

Что модуль не делает:
    - не пересчитывает benchmark и snapshot;
    - не меняет production pipeline и его артефакты.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.container import BarContainer

MODEL_ORDER = [
    "main_contrastive_v1",
    "baseline_random_forest",
    "baseline_mlp_small",
    "baseline_legacy_gaussian",
]

MODEL_LABELS = {
    "main_contrastive_v1": "Contrastive V1",
    "baseline_random_forest": "RandomForest",
    "baseline_mlp_small": "MLP small",
    "baseline_legacy_gaussian": "Legacy Gaussian",
}

MODEL_COLORS = {
    "main_contrastive_v1": "#1f77b4",
    "baseline_random_forest": "#2ca02c",
    "baseline_mlp_small": "#ff7f0e",
    "baseline_legacy_gaussian": "#7f7f7f",
}

METRIC_LABELS = {
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "precision_at_k": "Precision@50",
}

TIER_COLUMNS = ["high_rows", "medium_rows", "low_rows"]
TIER_LABELS = {
    "high_rows": "HIGH",
    "medium_rows": "MEDIUM",
    "low_rows": "LOW",
}

TIER_COLORS = {
    "high_rows": "#bf360c",
    "medium_rows": "#ef6c00",
    "low_rows": "#b0bec5",
}


@dataclass(slots=True)
class PresentationAssetConfig:
    """Параметры сборки slide-ready ассетов.

    Параметры:
        benchmark_run_name: run_name benchmark-артефактов без суффикса `_summary.csv`.
        snapshot_run_name: run_name snapshot-артефактов без суффикса `_snapshot_summary.csv`.
        input_dir: каталог с comparison CSV-артефактами.
        output_dir: каталог, куда сохраняются итоговые PNG и CSV.
        top_rank_limit: число top-строк для curve-графика и contrastive shortlist.
    """

    benchmark_run_name: str
    snapshot_run_name: str
    input_dir: Path
    output_dir: Path
    top_rank_limit: int = 10


@dataclass(slots=True)
class PresentationFrames:
    """Набор DataFrame, из которых строятся презентационные ассеты."""

    summary_df: pd.DataFrame
    classwise_df: pd.DataFrame
    search_summary_df: pd.DataFrame
    snapshot_summary_df: pd.DataFrame
    top_frames: dict[str, pd.DataFrame]


def find_project_root(start: Path) -> Path:
    """Найти корень репозитория по стандартным маркерам проекта.

    Параметры:
        start: каталог, от которого начинается поиск вверх по дереву.

    Возвращает:
        Path: путь до корня репозитория.

    Исключения:
        FileNotFoundError: если не удалось найти README.md и pyproject.toml.
    """

    for candidate in (start, *start.parents):
        if (candidate / "README.md").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(
        "Project root was not found. Expected README.md and pyproject.toml in parent directories."
    )


def parse_args() -> Namespace:
    """Разобрать CLI-аргументы генератора презентационных ассетов."""

    project_root = find_project_root(Path.cwd().resolve())
    default_input_dir = project_root / "experiments" / "model_comparison"
    default_output_dir = project_root / "docs" / "presentation" / "assets"

    parser = ArgumentParser(
        description="Собрать PNG и CSV-ассеты для презентации по comparison-artifacts.",
    )
    parser.add_argument(
        "--benchmark-run-name",
        default="baseline_comparison_2026-03-13_vkr30_cv10",
        help="Имя benchmark run без суффикса `_summary.csv`.",
    )
    parser.add_argument(
        "--snapshot-run-name",
        default="baseline_comparison_2026-03-13_vkr30_cv10_limit5000",
        help="Имя snapshot run без суффикса `_snapshot_summary.csv`.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help="Каталог с benchmark/snapshot CSV-артефактами.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Базовый каталог сохранения slide-ready ассетов.",
    )
    parser.add_argument(
        "--top-rank-limit",
        type=int,
        default=10,
        help="Число top-строк для curve-графика и contrastive shortlist.",
    )
    return parser.parse_args()


def build_config(args: Namespace) -> PresentationAssetConfig:
    """Собрать типизированный конфиг генератора из CLI-аргументов."""

    return PresentationAssetConfig(
        benchmark_run_name=str(args.benchmark_run_name),
        snapshot_run_name=str(args.snapshot_run_name),
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir) / str(args.benchmark_run_name),
        top_rank_limit=int(args.top_rank_limit),
    )


def load_frames(config: PresentationAssetConfig) -> PresentationFrames:
    """Прочитать comparison-artifacts, нужные для презентации.

    Источник данных:
        CSV-артефакты из experiments/model_comparison.
    """

    summary_path = config.input_dir / f"{config.benchmark_run_name}_summary.csv"
    classwise_path = config.input_dir / f"{config.benchmark_run_name}_classwise.csv"
    search_path = config.input_dir / f"{config.benchmark_run_name}_search_summary.csv"
    snapshot_path = config.input_dir / f"{config.snapshot_run_name}_snapshot_summary.csv"
    top_paths = {
        model_name: config.input_dir / f"{config.snapshot_run_name}_snapshot_{model_name}_top.csv"
        for model_name in MODEL_ORDER
    }

    required_paths = [summary_path, classwise_path, search_path, snapshot_path, *top_paths.values()]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Presentation artifacts were not found:\n{missing_text}")

    return PresentationFrames(
        summary_df=pd.read_csv(summary_path, low_memory=False),
        classwise_df=pd.read_csv(classwise_path, low_memory=False),
        search_summary_df=pd.read_csv(search_path, low_memory=False),
        snapshot_summary_df=pd.read_csv(snapshot_path, low_memory=False),
        top_frames={
            model_name: pd.read_csv(path, low_memory=False)
            for model_name, path in top_paths.items()
        },
    )


def _apply_model_order(frame: pd.DataFrame, column_name: str = "model_label") -> pd.DataFrame:
    ordered_labels = [MODEL_LABELS[name] for name in MODEL_ORDER]
    frame[column_name] = pd.Categorical(frame[column_name], categories=ordered_labels, ordered=True)
    return frame.sort_values(column_name).reset_index(drop=True)


def build_test_summary_table(frames: PresentationFrames) -> pd.DataFrame:
    """Собрать компактную test-таблицу benchmark-метрик для презентации."""

    test_df = frames.summary_df.loc[frames.summary_df["split_name"] == "test"].copy()
    test_df["model_label"] = test_df["model_name"].map(MODEL_LABELS)
    test_df = _apply_model_order(test_df)
    table = test_df[
        ["model_label", "n_rows", "n_host", "n_field", "roc_auc", "pr_auc", "brier", "precision_at_k"]
    ].rename(
        columns={
            "model_label": "Модель",
            "n_rows": "N test",
            "n_host": "Host",
            "n_field": "Field",
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
            "brier": "Brier",
            "precision_at_k": "Precision@50",
        }
    )
    numeric_columns = ["ROC-AUC", "PR-AUC", "Brier", "Precision@50"]
    table[numeric_columns] = table[numeric_columns].round(4)
    return table


def build_search_summary_table(frames: PresentationFrames) -> pd.DataFrame:
    """Собрать presentation-view по search_summary benchmark-контура."""

    search_df = frames.search_summary_df.copy()
    search_df["model_label"] = search_df["model_name"].map(MODEL_LABELS)
    search_df["spec_class"] = search_df["spec_class"].fillna("all")
    search_df = _apply_model_order(search_df)
    table = search_df[
        [
            "model_label",
            "search_scope",
            "spec_class",
            "refit_metric",
            "cv_folds",
            "candidate_count",
            "best_cv_score",
            "best_params_json",
        ]
    ].rename(
        columns={
            "model_label": "Модель",
            "search_scope": "Масштаб",
            "spec_class": "Spec class",
            "refit_metric": "Refit metric",
            "cv_folds": "CV folds",
            "candidate_count": "Кандидатов",
            "best_cv_score": "Лучший CV score",
            "best_params_json": "Best params",
        }
    )
    table["Лучший CV score"] = table["Лучший CV score"].round(4)
    return table


def build_snapshot_summary_table(frames: PresentationFrames) -> pd.DataFrame:
    """Собрать presentation-view по snapshot preview для защиты."""

    snapshot_df = frames.snapshot_summary_df.copy()
    snapshot_df["model_label"] = snapshot_df["model_name"].map(MODEL_LABELS)
    snapshot_df = _apply_model_order(snapshot_df)
    table = snapshot_df[
        [
            "model_label",
            "input_rows",
            "host_candidates",
            "high_rows",
            "medium_rows",
            "low_rows",
            "top_final_score",
        ]
    ].rename(
        columns={
            "model_label": "Модель",
            "input_rows": "Input rows",
            "host_candidates": "Host candidates",
            "high_rows": "HIGH",
            "medium_rows": "MEDIUM",
            "low_rows": "LOW",
            "top_final_score": "Top final_score",
        }
    )
    table["Top final_score"] = table["Top final_score"].round(4)
    return table


def build_contrastive_top_table(
    frames: PresentationFrames,
    *,
    top_rank_limit: int,
) -> pd.DataFrame:
    """Собрать shortlist top-кандидатов для основной contrastive модели."""

    contrastive_top = frames.top_frames["main_contrastive_v1"].copy()
    table = contrastive_top.head(top_rank_limit)[
        [
            "source_id",
            "predicted_spec_class",
            "host_posterior",
            "final_score",
            "priority_tier",
            "teff_gspphot",
            "logg_gspphot",
            "radius_gspphot",
        ]
    ].rename(
        columns={
            "source_id": "source_id",
            "predicted_spec_class": "spec_class",
            "host_posterior": "host_posterior",
            "final_score": "final_score",
            "priority_tier": "priority_tier",
            "teff_gspphot": "teff_gspphot",
            "logg_gspphot": "logg_gspphot",
            "radius_gspphot": "radius_gspphot",
        }
    )
    table[["host_posterior", "final_score"]] = table[["host_posterior", "final_score"]].round(4)
    table[["teff_gspphot", "logg_gspphot", "radius_gspphot"]] = table[
        ["teff_gspphot", "logg_gspphot", "radius_gspphot"]
    ].round(3)
    return table


def save_table(frame: pd.DataFrame, output_path: Path) -> None:
    """Сохранить табличный ассет в CSV.

    Побочные эффекты:
        Пишет CSV-файл на диск.
    """

    frame.to_csv(output_path, index=False)


def plot_benchmark_metrics(test_table: pd.DataFrame, output_path: Path) -> None:
    """Построить grouped bar chart по ключевым benchmark-метрикам.

    Побочные эффекты:
        Пишет PNG-файл на диск.
    """

    metric_columns = ["ROC-AUC", "PR-AUC", "Precision@50"]
    x_positions = np.arange(len(metric_columns), dtype=float)
    bar_width = 0.18

    fig, ax = plt.subplots(figsize=(11, 6))
    containers: list[BarContainer] = []
    for index, model_name in enumerate(MODEL_ORDER):
        model_label = MODEL_LABELS[model_name]
        model_row = test_table.loc[test_table["Модель"] == model_label]
        if model_row.empty:
            continue

        values = [
            float(model_row.iloc[0][metric_name])
            for metric_name in metric_columns
        ]
        offset = (index - (len(MODEL_ORDER) - 1) / 2) * bar_width
        container = ax.bar(
            x_positions + offset,
            values,
            width=bar_width,
            label=model_label,
            color=MODEL_COLORS[model_name],
        )
        containers.append(container)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(metric_columns)
    ax.set_ylim(0, 1.0)
    ax.set_title("Benchmark на test split (30%): ключевые метрики")
    ax.set_xlabel("")
    ax.set_ylabel("Значение метрики")
    for container in containers:
        ax.bar_label(container, fmt="%.2f", fontsize=9, padding=2)
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_classwise_heatmap(frames: PresentationFrames, output_path: Path) -> None:
    """Построить heatmap classwise ROC-AUC на test split."""

    classwise_test = frames.classwise_df.loc[frames.classwise_df["split_name"] == "test"].copy()
    classwise_test["model_label"] = classwise_test["model_name"].map(MODEL_LABELS)
    pivot = (
        classwise_test.pivot(index="model_label", columns="spec_class", values="roc_auc")
        .reindex([MODEL_LABELS[name] for name in MODEL_ORDER])
        .loc[:, ["F", "G", "K", "M"]]
    )
    plt.figure(figsize=(8, 5))
    ax = sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0.70, vmax=1.00)
    ax.set_title("Classwise ROC-AUC на test split")
    ax.set_xlabel("Спектральный класс")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_snapshot_priority_mix(snapshot_table: pd.DataFrame, output_path: Path) -> None:
    """Построить stacked bar chart по HIGH/MEDIUM/LOW на snapshot preview."""

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    bottoms = pd.Series(0, index=snapshot_table.index, dtype="int64")
    for tier_column in TIER_COLUMNS:
        ax.bar(
            snapshot_table["Модель"],
            snapshot_table[TIER_LABELS[tier_column]],
            bottom=bottoms,
            color=TIER_COLORS[tier_column],
            label=TIER_LABELS[tier_column],
        )
        bottoms = bottoms + snapshot_table[TIER_LABELS[tier_column]]
    ax.set_title("Snapshot preview (limit=5000): распределение приоритетов")
    ax.set_xlabel("")
    ax.set_ylabel("Число объектов")
    ax.legend(title="")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_top_score_curves(frames: PresentationFrames, output_path: Path, *, top_rank_limit: int) -> None:
    """Построить кривые `final_score` по top-rank для всех моделей."""

    curve_rows: list[dict[str, float | int | str]] = []
    for model_name in MODEL_ORDER:
        top_frame = frames.top_frames[model_name].copy().head(top_rank_limit)
        for rank, (_, row) in enumerate(top_frame.iterrows(), start=1):
            curve_rows.append(
                {
                    "rank": rank,
                    "final_score": float(row["final_score"]),
                    "model_label": MODEL_LABELS[model_name],
                }
            )
    curve_df = pd.DataFrame(curve_rows)
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=curve_df,
        x="rank",
        y="final_score",
        hue="model_label",
        palette=[MODEL_COLORS[name] for name in MODEL_ORDER],
        marker="o",
    )
    ax.set_title(f"Top-{top_rank_limit} final_score на snapshot preview")
    ax.set_xlabel("Rank")
    ax.set_ylabel("final_score")
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def generate_assets(config: PresentationAssetConfig) -> list[Path]:
    """Собрать все CSV и PNG-ассеты для презентации.

    Возвращает:
        list[Path]: список созданных артефактов.

    Побочные эффекты:
        Создаёт каталог и пишет PNG/CSV на диск.
    """

    sns.set_theme(style="whitegrid", context="talk")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    frames = load_frames(config)

    test_table = build_test_summary_table(frames)
    search_table = build_search_summary_table(frames)
    snapshot_table = build_snapshot_summary_table(frames)
    contrastive_top_table = build_contrastive_top_table(
        frames,
        top_rank_limit=config.top_rank_limit,
    )

    output_paths = [
        config.output_dir / "benchmark_test_table.csv",
        config.output_dir / "search_summary_table.csv",
        config.output_dir / "snapshot_summary_table.csv",
        config.output_dir / "contrastive_top_table.csv",
        config.output_dir / "benchmark_metrics.png",
        config.output_dir / "classwise_roc_auc_heatmap.png",
        config.output_dir / "snapshot_priority_mix.png",
        config.output_dir / "top_score_curves.png",
    ]

    save_table(test_table, output_paths[0])
    save_table(search_table, output_paths[1])
    save_table(snapshot_table, output_paths[2])
    save_table(contrastive_top_table, output_paths[3])

    plot_benchmark_metrics(test_table, output_paths[4])
    plot_classwise_heatmap(frames, output_paths[5])
    plot_snapshot_priority_mix(snapshot_table, output_paths[6])
    plot_top_score_curves(frames, output_paths[7], top_rank_limit=config.top_rank_limit)
    return output_paths


def main() -> None:
    """Собрать slide-ready ассеты и вывести пути сохранённых файлов."""

    config = build_config(parse_args())
    created_paths = generate_assets(config)
    print("Generated presentation assets:")
    for path in created_paths:
        print(path)


if __name__ == "__main__":
    main()
