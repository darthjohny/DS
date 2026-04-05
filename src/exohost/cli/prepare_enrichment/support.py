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

import argparse
from datetime import UTC, datetime
from pathlib import Path

from exohost.db.bmk_enrichment import BmkGaiaEnrichmentExportSummary

DEFAULT_GAIA_ENRICHMENT_OUTPUT_DIR = Path("artifacts/gaia_enrichment")


def print_prepare_enrichment_stage(message: str) -> None:
    # Печатаем короткий статус локального batch export шага.
    print(f"[prepare-enrichment] {message}")


def resolve_gaia_enrichment_output_dir(namespace: argparse.Namespace) -> Path:
    # Создаем отдельный run_dir под один batch export для Gaia enrichment.
    base_dir = Path(namespace.output_dir)
    run_stamp = datetime.now(UTC).strftime("%Y_%m_%d_%H%M%S_%f")
    return base_dir / f"bmk_gaia_enrichment__{run_stamp}"


def print_gaia_enrichment_export_summary(
    summary: BmkGaiaEnrichmentExportSummary,
) -> None:
    # Печатаем compact summary после локального batch export шага.
    print(
        "[prepare-enrichment] summary "
        f"relation={summary.relation_name} "
        f"xmatch_batch_id={summary.xmatch_batch_id} "
        f"only_conflict_free={summary.only_conflict_free} "
        f"total_rows_exported={summary.total_rows_exported} "
        f"total_batches={summary.total_batches} "
        f"batch_size={summary.batch_size}"
    )
    print(f"[artifacts] gaia_enrichment_dir={summary.output_dir}")
    print(f"[artifacts] gaia_enrichment_manifest={summary.manifest_path}")
    print(f"[artifacts] gaia_enrichment_query_template={summary.query_template_path}")
