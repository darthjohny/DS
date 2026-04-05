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

from exohost.db.coarse_ob_review_pool import CoarseObReviewPoolSummary


def print_materialize_ob_review_stage(message: str) -> None:
    # Печатаем короткий статус materialize-ob-review шага.
    print(f"[materialize-ob-review] {message}")


def print_coarse_ob_review_pool_summary(
    summary: CoarseObReviewPoolSummary,
) -> None:
    # Печатаем компактную сводку после materialization O/B review-pool.
    print(
        "[materialize-ob-review] summary "
        f"source_relation={summary.source_relation_name} "
        f"review_rows_loaded={summary.review_rows_loaded} "
        f"summary_rows_loaded={summary.summary_rows_loaded}"
    )
