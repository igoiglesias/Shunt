"""Tests for security utilities: password hashing, JWT issuance, and decoding."""

from app.core.security import (
    decode_jwt,
    hash_password,
    issue_jwt,
    verify_password,
)


class TestHashPassword:
    """Tests for hash_password function."""

    def test_hash_verify_roundtrip(self):
        """Hash a password and verify it returns True."""
        password = "semenha_secreta"
        hashed = hash_password(password)
        assert verify_password(password, hashed) is True

    def test_verify_wrong_password_false(self):
        """Wrong password should not verify."""
        password = "semenha_secreta"
        wrong_password = "outro_senha"
        hashed = hash_password(password)
        assert verify_password(wrong_password, hashed) is False

    def test_verify_tampered_hash_false(self):
        """Modificando os parametros do hash a verificacao deve falhar.

        O formato Argon2 e `$argon2id$v=19$m=<custo>,t=..,p=..$<salt>$<hash>`;
        alterar o custo (`m=`) muda a derivacao esperada, entao a verificacao
        nao confere. Corromper um `:` nao adianta -- o hash nao usa `:`.
        """
        password = "senha_segura"
        hashed = hash_password(password)
        # Troca o digito do custo `m=`: a verificacao rederiva com o novo custo
        # embutido e o hash guardado ja nao confere.
        pos = hashed.index("m=") + 1
        digit = hashed[pos]
        new_digit = "9" if digit != "9" else "0"
        tampered = hashed[:pos] + new_digit + hashed[pos + 1 :]
        assert verify_password(password, tampered) is False

    def test_hash_tampered_returns_false(self):
        """Tampered password hash should not verify."""
        password = "senha_segura"
        hashed = hash_password(password)
        # Flip a character in the hash to simulate tampering
        tampered = hashed[:-1] + ("0" if hashed[-1] != "0" else "1")
        assert verify_password(password, tampered) is False


class TestIssueJwt:
    """Tests for issue_jwt function."""

    def test_issue_valid_token(self):
        """Issuing a token with valid parameters should return a non-empty string."""
        user_id = 123
        secret = "my-super-secret"
        token = issue_jwt(user_id, secret, ttl_seconds=3600)
        assert isinstance(token, str)
        assert len(token) > 0

    def test_issue_token_with_zero_ttl(self):
        """Issuing a token with zero TTL should still produce a valid token."""
        user_id = 456
        secret = "my-super-secret"
        token = issue_jwt(user_id, secret, ttl_seconds=0)
        assert isinstance(token, str)


class TestDecodeJwt:
    """Tests for decode_jwt function."""

    def test_decode_valid_token(self):
        """Decoding a valid token should return a non-None dictionary."""
        from app.core.security import issue_jwt

        user_id = 789
        secret = "my-super-secret"
        token = issue_jwt(user_id, secret, ttl_seconds=3600)
        payload = decode_jwt(token, secret)
        assert payload is not None
        assert payload["sub"] == user_id
        assert "exp" in payload

    def test_decode_expired_token_returns_none(self):
        """An expired token should decode to None."""
        from app.core.security import issue_jwt

        user_id = 999
        secret = "my-super-secret"
        # Create a token that expires immediately
        token = issue_jwt(user_id, secret, ttl_seconds=-1)
        payload = decode_jwt(token, secret)
        assert payload is None

    def test_decode_invalid_token_returns_none(self):
        """An invalid token should decode to None."""

        secret = "my-super-secret"
        token = "invalid.token.here"
        payload = decode_jwt(token, secret)
        assert payload is None

    def test_decode_wrong_secret_returns_none(self):
        """Using a wrong secret should return None."""
        from app.core.security import issue_jwt

        user_id = 777
        secret = "correct-secret"
        token = issue_jwt(user_id, secret, ttl_seconds=3600)
        payload = decode_jwt(token, "wrong-secret")
        assert payload is None

    def test_decode_non_numeric_sub_returns_none(self):
        """`sub` que nao e numero: token forjado/de outra app -> nao autenticado.

        Antes da correcao, um `sub` como `"admin"` passava pelo `decode_jwt`
        e fazia o `require_admin` cair em `int("admin")` -> ValueError -> 500.
        Agora cai como qualquer decode ruim: `None`.
        """
        import jwt as pyjwt

        from app.core.security import _ALGORITHM

        secret = "segredo-de-teste"
        token = pyjwt.encode(
            {"sub": "admin", "exp": 9999999999},
            secret,
            algorithm=_ALGORITHM,
        )
        assert decode_jwt(token, secret) is None
