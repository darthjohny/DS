"""CLI-запуск офлайн-калибровки decision layer."""

from __future__ import annotations

import pandas as pd

from decision_calibration.config import (
    load_calibration_config,
    parse_args,
)
from decision_calibration.reporting import (
    build_iteration_summary,
    print_summary,
    save_iteration_artifacts,
    top_candidates_frame,
)
from decision_calibration.runtime import (
    load_ready_input_dataset,
    make_run_id,
    run_base_scoring,
)
from decision_calibration.scoring import (
    apply_calibration_config,
    build_low_priority_preview,
    build_unknown_preview,
)
from input_layer import make_engine_from_env
from logbooks.decision_layer import ensure_logbook_dir, next_iteration_number
from priority_pipeline import (
    host_model_version,
    load_models,
    order_priority_results,
)


def main() -> None:
    """Запустить одну итерацию офлайн-калибровки decision layer.

    Сценарий:
    - загружает validated dataset со статусом `READY`;
    - выполняет base scoring через production helpers;
    - применяет offline calibration formula;
    - собирает markdown и CSV артефакты итерации;
    - печатает краткую summary в stdout.
    """
    args = parse_args()
    engine = make_engine_from_env()

    dataset, df_input = load_ready_input_dataset(
        engine=engine,
        relation_name=args.relation,
        limit=args.limit,
    )
    router_model, host_model = load_models()
    config = load_calibration_config(args.config)
    run_id = make_run_id()

    base_result = run_base_scoring(
        df_input=df_input,
        router_model=router_model,
        host_model=host_model,
    )
    host_version = host_model_version(host_model)
    router_score_mode = str(router_model["meta"]["score_mode"])
    host_score_mode = str(host_model["meta"].get("score_mode", "legacy"))
    scored = apply_calibration_config(
        df_scored=base_result.host_scored_df,
        config=config,
        host_model_version_value=host_version,
    )
    low_known_preview = build_low_priority_preview(
        base_result.low_known_input_df
    )
    unknown_preview = build_unknown_preview(base_result.unknown_input_df)
    combined = pd.concat(
        [scored, low_known_preview, unknown_preview],
        ignore_index=True,
        sort=False,
    )
    ordered_results = order_priority_results(combined)

    summary = build_iteration_summary(
        run_id=run_id,
        dataset=dataset,
        base_result=base_result,
        ordered_results=ordered_results,
        top_n=args.top_n,
        router_score_mode=router_score_mode,
        host_score_mode=host_score_mode,
        host_model_version_value=host_version,
    )
    logbook_dir = ensure_logbook_dir()
    markdown_path = save_iteration_artifacts(
        logbook_dir=logbook_dir,
        config=config,
        summary=summary,
        ordered_results=ordered_results,
        top_n=args.top_n,
        iteration_note=args.iteration_note,
        next_iteration_number=next_iteration_number(logbook_dir),
    )
    top_candidates = top_candidates_frame(ordered_results, args.top_n)
    print_summary(summary, markdown_path, top_candidates)


__all__ = ["main"]
