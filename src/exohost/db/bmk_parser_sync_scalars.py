# Файл `bmk_parser_sync_scalars.py` слоя `db`.
#
# Этот файл отвечает только за:
# - relation-layer, materialization и SQL-обвязку;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `db` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations


def cursor_rowcount(cursor: object) -> int:
    # Приводим rowcount DB-курсора к неотрицательному целому.
    rowcount = getattr(cursor, "rowcount", -1)
    if isinstance(rowcount, bool):
        return int(rowcount)
    if isinstance(rowcount, int):
        return max(rowcount, 0)
    if isinstance(rowcount, float):
        return max(int(rowcount), 0)
    if isinstance(rowcount, str) and rowcount.strip():
        return max(int(rowcount.strip()), 0)
    return 0


def scalar_to_int(value: object, *, relation_name: str) -> int:
    # Безопасно приводим DB-скаляр к int и даем понятную ошибку на мусорном типе.
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        return int(value.strip())
    raise RuntimeError(
        "Unable to convert parser sync scalar to int: "
        f"{relation_name} -> {value!r}"
    )


__all__ = [
    "cursor_rowcount",
    "scalar_to_int",
]
