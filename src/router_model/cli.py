"""CLI-обвязка для retrain и preview Gaussian router."""

from __future__ import annotations

import os

from router_model.artifacts import save_router_model
from router_model.db import load_router_training_from_db, make_engine_from_env
from router_model.fit import fit_router_model
from router_model.score import score_router_df


def main() -> None:
    """Переобучить router-модель и напечатать короткий preview.

    Побочные эффекты
    ----------------
    - читает reference-layer из Postgres;
    - пересобирает artifact router;
    - сохраняет JSON в `data/router_gaussian_params.json`;
    - печатает preview scoring для первых строк обучающей выборки.
    """
    engine = make_engine_from_env()
    df_router = load_router_training_from_db(engine)
    model = fit_router_model(df_router=df_router)

    os.makedirs("data", exist_ok=True)
    output_path = "data/router_gaussian_params.json"
    save_router_model(model, output_path)
    print(f"Saved router model to {output_path}")

    sample = score_router_df(model=model, df=df_router.head(10))
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
    print(preview.to_string(index=False))


__all__ = ["main"]
