# Файл `support.py` слоя `ui/pages`.
#
# Этот файл отвечает только за:
# - мелкие page-helper функции для Streamlit-страниц интерфейса;
# - устранение дублирования простой навигационной логики между страницами.
#
# Следующий слой:
# - сами страницы `run_browser`, `candidate` и `csv_decide`;
# - их scenario- и smoke-проверки.

from __future__ import annotations


def resolve_selected_index(
    *,
    options: tuple[str, ...],
    selected_value: object,
) -> int:
    # Selectbox-страницы должны одинаково восстанавливать последний выбор пользователя.
    if not options:
        return 0
    selected_key = str(selected_value) if selected_value is not None else None
    if selected_key is None:
        return 0
    try:
        return options.index(selected_key)
    except ValueError:
        return 0


__all__ = ["resolve_selected_index"]
