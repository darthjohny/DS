# Файл `markdown_tables.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

import pandas as pd


def frame_to_code_block(df: pd.DataFrame, *, max_rows: int | None = None) -> str:
    # Преобразуем DataFrame в простой text-блок внутри markdown.
    if df.empty:
        return "```text\n(empty)\n```"

    prepared_frame = df.head(max_rows) if max_rows is not None else df
    table_text = prepared_frame.to_string(index=False)
    return f"```text\n{table_text}\n```"


def format_scalar_line(label: str, value: object) -> str:
    # Форматируем одну строку key-value для markdown summary.
    return f"- {label}: `{value}`"
