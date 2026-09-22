"""Testes para os modelos User e ApiToken do armazem de estatisticas."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.stats.models import ApiToken, Base, User


def _make_engine(tmp_path):
    """Cria um engine SQLite de arquivo temporario e cria todas as tabelas."""
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'user_models.db'}")
    Base.metadata.create_all(engine)
    return engine


def test_user_table_created(tmp_path):
    """Insere um User, faz commit e lê de volta por username."""
    engine = _make_engine(tmp_path)
    try:
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
    finally:
        engine.dispose()


def test_username_unique(tmp_path):
    """Inserir dois Users com o mesmo username levanta IntegrityError no segundo commit."""
    engine = _make_engine(tmp_path)
    try:
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
    finally:
        engine.dispose()


def test_api_token_roundtrip(tmp_path):
    """Insere um ApiToken com token_hash + expires_at, lê de volta, campos preservados."""
    engine = _make_engine(tmp_path)
    try:
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
    finally:
        engine.dispose()


def test_user_last_login_nullable(tmp_path):
    """Um novo User tem last_login_at None por padrao; pode setar e persiste."""
    engine = _make_engine(tmp_path)
    try:
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
    finally:
        engine.dispose()