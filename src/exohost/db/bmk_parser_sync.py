# Файл `bmk_parser_sync.py` слоя `db`.
#
# Этот файл отвечает только за:
# - relation-layer, materialization и SQL-обвязку;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `db` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from exohost.db.bmk_parser_sync_execution import sync_bmk_parser_fields_downstream

__all__ = ["sync_bmk_parser_fields_downstream"]
