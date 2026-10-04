"""Troca a senha de um usuario do painel direto no banco.

Uso:
    uv run python -m scripts.change_pass iglesias 'nova-senha'
    make change_pass USER=iglesias PASS='nova-senha'

O alvo e operacional, nao codigo de request: a unica porta de entrada fora
da tela de login (que exige a senha atual, inutil se o operador a perdeu).
Roda contra o `TURSO_DATABASE_URL` do `.env` quando nenhum argumento extra
e dado, e mostra o hash ANTES de sobrescrever para auditoria.
"""

from __future__ import annotations

import argparse
import os
import sys

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.config.config import load_dotenv
from app.core.security import hash_password, verify_password
from app.stats.models import User


def change_password(database_url: str, username: str, new_password: str) -> dict:
    """Aplica a nova senha e devolve um relatorio para o operador.

    Levanta `LookupError` quando o usuario nao existe: silenciar criaria a
    senha de um usuario novo por um caminho que nao e a tela de login.
    """
    if not new_password:
        raise ValueError("a nova senha nao pode ser vazia")
    engine = create_engine(database_url)
    with Session(engine) as session:
        user = session.execute(
            select(User).where(User.username == username)
        ).scalar_one_or_none()
        if user is None:
            raise LookupError(f"usuario {username!r} nao encontrado")
        previous = user.password_hash
        user.password_hash = hash_password(new_password)
        session.commit()
        return {
            "id": user.id,
            "username": user.username,
            "changed": True,
            "previous_hash": previous,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Troca a senha de um administrador do painel Shunt.",
    )
    parser.add_argument("username")
    parser.add_argument(
        "password",
        help="Nova senha. Se omitida, lida de SHUNT_NEW_PASS (nao vira historico).",
        nargs="?",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Override do TURSO_DATABASE_URL (padrao: o do .env).",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    database_url = args.database_url or os.environ.get("TURSO_DATABASE_URL")
    if not database_url:
        print("erro: TURSO_DATABASE_URL ausente", file=sys.stderr)
        return 2

    password = args.password if args.password is not None else os.environ.get("SHUNT_NEW_PASS")
    if password is None:
        print("erro: forneça a senha como argumento ou SHUNT_NEW_PASS", file=sys.stderr)
        return 2

    try:
        report = change_password(database_url, args.username, password)
    except LookupError as err:
        print(f"erro: {err}", file=sys.stderr)
        return 1
    except ValueError as err:
        print(f"erro: {err}", file=sys.stderr)
        return 2

    print(
        f"senha trocada: id={report['id']} user={report['username']}\n"
        f"hash anterior: {report['previous_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
