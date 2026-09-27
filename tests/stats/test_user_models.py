"""Testes para os modelos User e ApiToken do armazem de estatisticas."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.stats.models import ApiToken, User


def test_user_table_created(make_engine):
    """Insere um User, faz commit e lê de volta por username."""
    engine = make_engine()

    with Session(engine) as session:
        user = User(
            username="testuser",
            password_hash="hashed_password_123",
            created_at=datetime.now(UTC),
        )
        session.add(user)
        session.commit()

    with Session(engine) as session:
        stmt = select(User).where(User.username == "testuser")
        stored = session.scalar(stmt)
        assert stored is not None
        assert stored.username == "testuser"
        assert stored.password_hash == "hashed_password_123"


def test_username_unique(make_engine):
    """Inserir dois Users com o mesmo username levanta IntegrityError no segundo commit."""
    engine = make_engine()

    with Session(engine) as session:
        user1 = User(
            username="duplicate",
            password_hash="hash1",
            created_at=datetime.now(UTC),
        )
        session.add(user1)
        session.commit()

        user2 = User(
            username="duplicate",
            password_hash="hash2",
            created_at=datetime.now(UTC),
        )
        session.add(user2)
        with pytest.raises(IntegrityError):
            session.commit()


def test_api_token_roundtrip(make_engine):
    """Insere um ApiToken com token_hash + expires_at, lê de volta, campos preservados."""
    engine = make_engine()

    with Session(engine) as session:
        token = ApiToken(
            name="test-token",
            token_hash="a" * 64,
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC),
        )
        session.add(token)
        session.commit()

    with Session(engine) as session:
        stmt = select(ApiToken).where(ApiToken.name == "test-token")
        stored = session.scalar(stmt)
        assert stored is not None
        assert stored.name == "test-token"
        assert stored.token_hash == "a" * 64
        assert stored.created_at is not None
        assert stored.expires_at is not None


def test_user_last_login_nullable(make_engine):
    """Um novo User tem last_login_at None por padrao; pode setar e persiste."""
    engine = make_engine()

    with Session(engine) as session:
        user = User(
            username="nologin",
            password_hash="hash",
            created_at=datetime.now(UTC),
        )
        session.add(user)
        session.commit()

        # Por padrao, last_login_at e None
        assert user.last_login_at is None

        # Seta e persiste
        user.last_login_at = datetime.now(UTC)
        session.commit()

    with Session(engine) as session:
        stmt = select(User.last_login_at).where(User.username == "nologin")
        result = session.scalar(stmt)
        assert result is not None
