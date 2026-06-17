"""enable pgvector extension for LlamaIndex RAG (Layer 4 long-term memory)

The report_chunks table is managed by LlamaIndex's PGVectorStore (perform_setup=True)
and is created automatically on first use. Only the extension needs to be pre-installed.

Revision ID: cce81f2a4b9d
Revises: 7a885b67938e
Create Date: 2026-06-17
"""
from alembic import op

revision = "cce81f2a4b9d"
down_revision = "7a885b67938e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS vector")
