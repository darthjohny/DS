"""DataFrame-level helpers для общего контракта priority pipeline.

Модуль отвечает за:

- добавление отсутствующих decision-layer колонок;
- выравнивание входных DataFrame под общий schema contract pipeline.
"""

from __future__ import annotations

import pandas as pd


def ensure_decision_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить отсутствующие decision-layer колонки с нейтральными значениями.

    Функция готовит входной DataFrame к общему контракту боевого
    pipeline, даже если часть soft-factor колонок отсутствует в исходной
    relation.
    """
    result = df.copy()
    defaults: dict[str, float] = {
        "parallax": float("nan"),
        "parallax_over_error": float("nan"),
        "ruwe": float("nan"),
        "bp_rp": float("nan"),
        "mh_gspphot": float("nan"),
        "validation_factor": 1.0,
    }
    for column, default in defaults.items():
        if column not in result.columns:
            result[column] = default
    return result


__all__ = [
    "ensure_decision_columns",
]
