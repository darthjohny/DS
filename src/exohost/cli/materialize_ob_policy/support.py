# Файл `support.py` слоя `cli`.
#
# Этот файл отвечает только за:
# - CLI-команды и orchestration entrypoints;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - CLI-команды или support-модули этого же домена;
# - пользовательский запуск через `python -m exohost.cli.main`.

from __future__ import annotations

from exohost.db.coarse_ob_boundary_policy import CoarseObBoundaryPolicySummary


def print_materialize_ob_policy_stage(message: str) -> None:
    # Печатаем короткий статус materialize-ob-policy шага.
    print(f"[materialize-ob-policy] {message}")


def print_coarse_ob_boundary_policy_summary(
    summary: CoarseObBoundaryPolicySummary,
) -> None:
    # Печатаем компактную сводку после materialization O/B policy relations.
    print(
        "[materialize-ob-policy] summary "
        f"source_relation={summary.source_relation_name} "
        f"secure_o_like_rows={summary.secure_o_like_rows} "
        f"ob_boundary_rows={summary.ob_boundary_rows} "
        f"summary_rows_loaded={summary.summary_rows_loaded}"
    )
