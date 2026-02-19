# db_test.py
# Назначение:
# - Проверить, что Python (venv) подключается к Postgres
# - Проверить, что схема `lab` и ключевая вьюха доступны (используется в пайплайне)

import os
from sqlalchemy import create_engine, text

# Параметры по умолчанию (как в DBeaver)
USER = "postgres"
PASSWORD = "1234"
HOST = "127.0.0.1"
PORT = 5432
DB = "dspro_vkr_research"

# Единый источник правды для подключения:
# - если в окружении задан DATABASE_URL, используем его
# - иначе собираем DSN из параметров выше
database_url = os.getenv("DATABASE_URL")
if not database_url:
    database_url = f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}"

engine = create_engine(
    database_url,
    connect_args={"connect_timeout": 5},
)

with engine.connect() as conn:
    # 0) Connection sanity
    print(conn.execute(text("select 1")).fetchone())
    print(conn.execute(text("select current_database(), current_user, current_schema()")).fetchone())

    # 1) Schema existence + view access
    print(conn.execute(text("select schema_name from information_schema.schemata where schema_name='lab'")).fetchall())
    print(conn.execute(text("select count(*) from lab.v_nasa_gaia_train_classified")).fetchone())