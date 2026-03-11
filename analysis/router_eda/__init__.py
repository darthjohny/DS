"""Пакет исследовательского EDA для router-слоя.

Содержит загрузку reference-выборки, readiness-метрики, графики,
export helpers и CLI для анализа физического router. При импорте
добавляет `src/` в `sys.path`, чтобы использовать production-модули
и контракты проекта.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
