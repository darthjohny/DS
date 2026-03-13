"""CLI-обвязка для retrain и preview Gaussian router."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    # При прямом запуске файла `src/router_model/cli.py` Python кладёт
    # каталог `src/router_model` в начало `sys.path`, и локальный
    # `math.py` начинает перекрывать stdlib `math`. Переключаем импортный
    # корень на `src`, чтобы CLI одинаково работал и как модуль, и как файл.
    _script_path = Path(__file__).resolve()
    _script_dir = str(_script_path.parent)
    _src_root = str(_script_path.parents[1])
    if sys.path and sys.path[0] == _script_dir:
        sys.path.pop(0)
    if _src_root not in sys.path:
        sys.path.insert(0, _src_root)

from router_model.artifacts import save_router_model
from router_model.db import (
    ROUTER_VIEW,
    load_router_training_from_db,
    make_engine_from_env,
)
from router_model.fit import fit_router_model
from router_model.score import score_router_df

DEFAULT_OUTPUT_PATH = Path("data/router_gaussian_params.json")
DEFAULT_PREVIEW_LIMIT = 10


def parse_args(argv: Sequence[str] | None = None) -> Namespace:
    """Разобрать CLI-аргументы retrain-сценария router."""
    parser = ArgumentParser(
        description="Переобучение и preview Gaussian router."
    )
    parser.add_argument(
        "--source-view",
        default=ROUTER_VIEW,
        help="Relation/view для router training source.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Путь сохранения JSON artifact.",
    )
    parser.add_argument(
        "--shrink-alpha",
        type=float,
        default=0.15,
        help="Коэффициент shrinkage для class covariance.",
    )
    parser.add_argument(
        "--min-class-size",
        type=int,
        default=3,
        help="Минимальный размер класса для включения в artifact.",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=DEFAULT_PREVIEW_LIMIT,
        help="Сколько строк training source показать в preview.",
    )
    parser.add_argument(
        "--allow-unknown",
        action="store_true",
        help="Включить open-set reject-option в metadata artifact.",
    )
    parser.add_argument(
        "--ood-policy-version",
        default=None,
        help="Явная версия OOD policy в metadata artifact.",
    )
    parser.add_argument(
        "--min-router-log-posterior",
        type=float,
        default=None,
        help="Нижний порог reject по router_log_posterior.",
    )
    parser.add_argument(
        "--min-posterior-margin",
        type=float,
        default=None,
        help="Нижний порог reject по posterior_margin.",
    )
    parser.add_argument(
        "--min-router-similarity",
        type=float,
        default=None,
        help="Нижний порог reject по router_similarity.",
    )
    return parser.parse_args(argv)


def validate_ood_args(args: Namespace) -> None:
    """Проверить согласованность OOD-аргументов retrain CLI."""
    thresholds_specified = any(
        value is not None
        for value in (
            args.min_router_log_posterior,
            args.min_posterior_margin,
            args.min_router_similarity,
        )
    )
    if thresholds_specified and not bool(args.allow_unknown):
        raise ValueError(
            "OOD thresholds require explicit --allow-unknown."
        )
    if (
        args.min_posterior_margin is None
    ) != (
        args.min_router_similarity is None
    ):
        raise ValueError(
            "posterior_margin and router_similarity thresholds must be "
            "set together."
        )
    if bool(args.allow_unknown) and not thresholds_specified:
        raise ValueError(
            "When --allow-unknown is enabled, provide at least one "
            "threshold rule."
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Переобучить router-модель и напечатать короткий preview.

    Побочные эффекты
    ----------------
    - читает reference-layer из Postgres;
    - пересобирает artifact router;
    - сохраняет JSON в `data/router_gaussian_params.json`;
    - печатает preview scoring для первых строк обучающей выборки.
    """
    args = parse_args(argv)
    validate_ood_args(args)

    engine = make_engine_from_env()
    df_router = load_router_training_from_db(
        engine,
        view_name=str(args.source_view),
    )
    model = fit_router_model(
        df_router=df_router,
        shrink_alpha=float(args.shrink_alpha),
        min_class_size=int(args.min_class_size),
        source_view=str(args.source_view),
        allow_unknown=bool(args.allow_unknown),
        ood_policy_version=args.ood_policy_version,
        min_router_log_posterior=args.min_router_log_posterior,
        min_posterior_margin=args.min_posterior_margin,
        min_router_similarity=args.min_router_similarity,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_router_model(model, str(output_path))
    print(f"Saved router model to {output_path}")
    print(f"Training rows: {len(df_router)}")
    print(f"Router classes: {len(model['classes'])}")
    print(f"OOD enabled: {bool(model['meta']['allow_unknown'])}")
    print(f"OOD policy version: {model['meta']['ood_policy_version']}")

    preview_limit = max(0, int(args.preview_limit))
    sample = score_router_df(model=model, df=df_router.head(preview_limit))
    preview = sample[
        [
            "spec_class",
            "evolution_stage",
            "predicted_spec_class",
            "predicted_evolution_stage",
            "router_label",
            "d_mahal_router",
            "router_similarity",
        ]
    ]
    if not sample.empty:
        unknown_rows = int(
            (sample["predicted_spec_class"].astype(str) == "UNKNOWN").sum()
        )
        print(f"Preview rows: {len(sample)}")
        print(f"Preview unknown rows: {unknown_rows}")
    print(preview.to_string(index=False))


__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_PREVIEW_LIMIT",
    "main",
    "parse_args",
    "validate_ood_args",
]


if __name__ == "__main__":
    main()
