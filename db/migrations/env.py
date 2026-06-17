import os
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool, text

load_dotenv()

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Use the async URL from .env but swap driver to psycopg (sync) for Alembic.
# asyncpg uses ssl=require; psycopg uses sslmode=require.
_db_url = (
    os.environ["DATABASE_URL"]
    .replace("postgresql+asyncpg://", "postgresql+psycopg://")
    .replace("ssl=require", "sslmode=require")
)
config.set_main_option("sqlalchemy.url", _db_url)

from db.models import Base  # noqa: E402 — import after path is configured

target_metadata = Base.metadata

# LangGraph manages its own checkpoint tables — exclude them from autogenerate.
_LANGGRAPH_TABLES = {
    "checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations"
}


def _include_object(obj, name, type_, reflected, compare_to):  # type: ignore[no-untyped-def]
    if type_ == "table" and name in _LANGGRAPH_TABLES:
        return False
    return True


def run_migrations_offline() -> None:
    context.configure(
        url=_db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=_include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args={"connect_timeout": 10},
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=_include_object,
        )
        with context.begin_transaction():
            # Fail fast if a DDL statement can't acquire its lock.
            connection.execute(text("SET lock_timeout = '10s'"))
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
