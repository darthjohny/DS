# Пакетный файл слоя `cli`.
#
# Этот файл отвечает только за:
# - пакетный marker/export-layer для домена `cli`;
# - короткую навигацию по модульной структуре CLI-команды и orchestration entrypoints.
#
# Следующий слой:
# - конкретные модули этого пакета;
# - слои выше, которые импортируют этот пакет дальше.

from .command import register_train_parser

__all__ = ["register_train_parser"]
