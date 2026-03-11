"""Пакет исследовательского EDA для host-модели.

Содержит загрузку данных, статистики, графики, export helpers и CLI
для анализа выборок `host vs field`. При импорте добавляет `src/` в
`sys.path`, чтобы исследовательский контур мог использовать production
модули проекта.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
