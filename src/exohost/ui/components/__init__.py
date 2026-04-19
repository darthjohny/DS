# Пакет `components` слоя `ui`.
#
# Этот пакет хранит только:
# - повторно используемые визуальные блоки интерфейса;
# - тонкие render-helper без бизнес-логики и без прямого доступа к БД.
#
# Следующий слой:
# - страницы интерфейса Streamlit;
# - unit-тесты helper-слоя и smoke entrypoint.

from __future__ import annotations

