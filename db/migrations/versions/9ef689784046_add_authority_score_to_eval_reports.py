"""add authority_score to eval_reports

Revision ID: 9ef689784046
Revises: cce81f2a4b9d
Create Date: 2026-07-03 09:05:17.402525

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '9ef689784046'
down_revision: Union[str, None] = 'cce81f2a4b9d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # data_report_chunks (LlamaIndex vector store, created outside SQLAlchemy
    # metadata) must never be touched by autogenerate — see env.py's
    # include_object filter.
    op.add_column(
        'eval_reports',
        sa.Column('authority_score', sa.Float(), server_default='0', nullable=False),
    )


def downgrade() -> None:
    op.drop_column('eval_reports', 'authority_score')
