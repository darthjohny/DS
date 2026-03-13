"""Точечные тесты bootstrap-helper-ов подключения к Postgres."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import infra.db as infra_db


def test_load_dotenv_local_sets_missing_values_without_overwriting_existing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`.env` должен заполнять только отсутствующие переменные окружения."""
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            [
                "# comment",
                "PGHOST=env-host",
                "PGUSER='env-user'",
                'PGPASSWORD="env-password"',
                "BROKEN_LINE",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PGHOST", "existing-host")
    monkeypatch.delenv("PGUSER", raising=False)
    monkeypatch.delenv("PGPASSWORD", raising=False)

    infra_db.load_dotenv_local(str(dotenv_path))

    assert infra_db.os.environ["PGHOST"] == "existing-host"
    assert infra_db.os.environ["PGUSER"] == "env-user"
    assert infra_db.os.environ["PGPASSWORD"] == "env-password"


def test_make_engine_from_env_uses_database_url_and_connect_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """При наличии `DATABASE_URL` helper должен использовать его как приоритетный источник."""
    captured: dict[str, Any] = {}

    def fake_create_engine(url: str, connect_args: dict[str, Any] | None = None) -> str:
        captured["url"] = url
        captured["connect_args"] = connect_args
        return "engine-from-url"

    monkeypatch.setattr(infra_db, "create_engine", fake_create_engine)
    monkeypatch.setattr(infra_db, "load_dotenv_local", lambda dotenv_path=".env": None)
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://user:pass@host:5432/db")
    monkeypatch.delenv("PGHOST", raising=False)
    monkeypatch.delenv("PGDATABASE", raising=False)
    monkeypatch.delenv("PGUSER", raising=False)
    monkeypatch.delenv("PGPASSWORD", raising=False)

    engine = infra_db.make_engine_from_env(connect_timeout=7)

    assert engine == "engine-from-url"
    assert captured == {
        "url": "postgresql+psycopg2://user:pass@host:5432/db",
        "connect_args": {"connect_timeout": 7},
    }


def test_make_engine_from_env_rejects_placeholder_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Placeholder DSN должен отклоняться до попытки создать engine."""
    monkeypatch.setattr(infra_db, "load_dotenv_local", lambda dotenv_path=".env": None)
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME",
    )

    with pytest.raises(RuntimeError, match="looks like a placeholder"):
        infra_db.make_engine_from_env(reject_placeholder_url=True)


def test_make_engine_from_env_falls_back_to_pg_variables_with_default_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Без `DATABASE_URL` helper должен собрать DSN из `PG*` и дефолтного порта."""
    captured: dict[str, Any] = {}

    def fake_create_engine(url: str, connect_args: dict[str, Any] | None = None) -> str:
        captured["url"] = url
        captured["connect_args"] = connect_args
        return "engine-from-pg-vars"

    monkeypatch.setattr(infra_db, "create_engine", fake_create_engine)
    monkeypatch.setattr(infra_db, "load_dotenv_local", lambda dotenv_path=".env": None)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("PGHOST", "db-host")
    monkeypatch.setenv("PGDATABASE", "gaia")
    monkeypatch.setenv("PGUSER", "codex")
    monkeypatch.setenv("PGPASSWORD", "secret")
    monkeypatch.delenv("PGPORT", raising=False)

    engine = infra_db.make_engine_from_env()

    assert engine == "engine-from-pg-vars"
    assert captured == {
        "url": "postgresql+psycopg2://codex:secret@db-host:5432/gaia",
        "connect_args": None,
    }


def test_make_engine_from_env_raises_custom_missing_message_when_config_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """При отсутствии конфигурации helper должен выдавать понятную ошибку."""
    monkeypatch.setattr(infra_db, "load_dotenv_local", lambda dotenv_path=".env": None)
    for name in ("DATABASE_URL", "PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD", "PGPORT"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="custom missing message"):
        infra_db.make_engine_from_env(missing_message="custom missing message")


def test_looks_like_placeholder_dsn_detects_template_tokens() -> None:
    """Helper placeholder-detection должен распознавать незаполненный шаблон DSN."""
    assert infra_db._looks_like_placeholder_dsn("postgresql://USER:PASSWORD@HOST/DBNAME")
    assert not infra_db._looks_like_placeholder_dsn(
        "postgresql://real_user:real_password@real-host/real_db"
    )
