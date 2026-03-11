"""CLI-точки входа для переобучения артефактов host-модели."""

from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace
from collections.abc import Mapping
from typing import Any

from host_model.artifacts import save_model
from host_model.constants import CONTRASTIVE_POPULATION_COLUMN
from host_model.db import (
    load_contrastive_training_from_db,
    load_default_contrastive_training_from_db,
    load_dwarfs_from_db,
    make_engine_from_env,
    resolve_contrastive_view_name,
)
from host_model.fit import fit_contrastive_gaussian_model, fit_gaussian_model


def parse_args() -> Namespace:
    """Разобрать CLI-аргументы для legacy или contrastive retraining."""
    parser = ArgumentParser(
        description="Переобучение legacy или contrastive артефакта host-модели."
    )
    parser.add_argument(
        "--mode",
        choices=("contrastive", "legacy"),
        default="contrastive",
        help="Режим обучения артефакта host-модели.",
    )
    parser.add_argument(
        "--view",
        default=None,
        help=(
            "Имя DB view. Для contrastive режима источник должен содержать "
            "spec_class, teff/logg/radius и бинарную колонку host/field. "
            "Если аргумент не задан, используются стандартные project sources."
        ),
    )
    parser.add_argument(
        "--population-col",
        default=CONTRASTIVE_POPULATION_COLUMN,
        help="Имя бинарной host/field колонки для contrastive режима.",
    )
    parser.add_argument(
        "--output",
        default="data/model_gaussian_params.json",
        help="Путь к итоговому JSON artifact.",
    )
    parser.add_argument(
        "--shrink-alpha",
        type=float,
        default=0.15,
        help="Коэффициент shrinkage для ковариаций.",
    )
    parser.add_argument(
        "--min-population-size",
        type=int,
        default=2,
        help="Минимальный размер `host/field` популяции на класс в contrastive режиме.",
    )
    parser.add_argument(
        "--min-legacy-class-size",
        type=int,
        default=3,
        help="Параметр сохранён для совместимости; legacy fitting использует внутренний guard.",
    )
    parser.add_argument(
        "--no-m-subclasses",
        action="store_true",
        help="Отключить разбиение M-класса на подклассы.",
    )
    return parser.parse_args()


def main() -> None:
    """Запустить CLI переобучения host-модели.

    Сценарий выбирает один из двух training-path:
    - `contrastive` для текущего production artifact;
    - `legacy` для обратной совместимости.

    Побочные эффекты
    ----------------
    - читает обучающие данные из Postgres;
    - сохраняет JSON artifact на диск;
    - печатает путь к итоговой модели.
    """
    args = parse_args()
    engine = make_engine_from_env()
    use_m_subclasses = not bool(args.no_m_subclasses)

    if args.mode == "contrastive":
        view_name = resolve_contrastive_view_name(args.view)
        if view_name:
            df_training = load_contrastive_training_from_db(
                engine=engine,
                view_name=view_name,
                population_col=args.population_col,
            )
        else:
            df_training = load_default_contrastive_training_from_db(
                engine=engine,
                population_col=args.population_col,
            )
        model: Mapping[str, Any] = fit_contrastive_gaussian_model(
            df_training=df_training,
            population_col=args.population_col,
            use_m_subclasses=use_m_subclasses,
            shrink_alpha=float(args.shrink_alpha),
            min_population_size=int(args.min_population_size),
        )
    else:
        legacy_view = args.view or "lab.v_nasa_gaia_train_dwarfs"
        df_dwarfs = load_dwarfs_from_db(engine, view_name=legacy_view)
        model = fit_gaussian_model(
            df_dwarfs=df_dwarfs,
            use_m_subclasses=use_m_subclasses,
            shrink_alpha=float(args.shrink_alpha),
        )

    output_path = str(args.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_model(model, output_path)
    print(f"Saved Gaussian model to {output_path}")


__all__ = ["main", "parse_args"]
