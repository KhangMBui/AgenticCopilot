"""enable pgvector extension

Revision ID: cad7872da278
Revises: c92851cc0354
Create Date: 2026-03-01 08:51:03.421727

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "cad7872da278"
down_revision: Union[str, Sequence[str], None] = "c92851cc0354"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    # Drop extension (careful: drops all vector columns)
    op.execute("DROP EXTENSION IF EXISTS vector")
