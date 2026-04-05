# Базовые smoke-тесты для стартового каркаса V2.

from exohost import __version__
from exohost.cli.main import build_parser, main


def test_package_exposes_version() -> None:
    # Проверяем, что пакет импортируется и экспортирует версию.
    assert __version__ == "0.1.0"


def test_cli_parser_accepts_known_command() -> None:
    # Проверяем, что CLI принимает зафиксированные команды верхнего уровня.
    parser = build_parser()
    namespace = parser.parse_args(["train"])

    assert namespace.command == "train"


def test_cli_main_returns_zero_for_known_command() -> None:
    # Проверяем, что минимальный CLI-контур завершается успешно.
    assert main(["train"]) == 0
