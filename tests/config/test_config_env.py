"""Testes de que as knobs operacionais lêem SHUNT_* do ambiente com defaults."""

import os
import sys
from importlib import reload

# Garante que o path do projeto esteja disponível
sys.path.insert(0, "/home/iglesias/Documents/Projetos_Pessoais/Shunt")


def test_env_overrides_default():
    """SHUNT_MAX_ATTEMPTS override do default 3."""
    os.environ["SHUNT_MAX_ATTEMPTS"] = "7"
    # reload força a leitura do ambiente no topo do módulo
    import app.config.config as cfg
    reload(cfg)
    assert cfg.MAX_ATTEMPTS == 7
    # limpa
    del os.environ["SHUNT_MAX_ATTEMPTS"]
    reload(cfg)
    assert cfg.MAX_ATTEMPTS == 3


def test_all_env_knobs_readable():
    """Cada knob tem sua env var e default correto."""
    import app.config.config as cfg
    reload(cfg)

    # Mapa de atributo do config -> (default_esperado, tipo)
    checks = {
        "MAX_ATTEMPTS": (3, int),
        "RETRY_AFTER_BUDGET": (5.0, float),
        "TOTAL_DEADLINE": (120.0, float),
        "FIRST_EVENT_DEADLINE": (20.0, float),
        "ENGINE_BOOT_TIMEOUT": (10.0, float),
        "REACH_TIMEOUT": (2.0, float),
        "BUSY_TIMEOUT_MS": (5000, int),
        "TIMEOUT_CONNECT": (10.0, float),
        "TIMEOUT_READ": (60.0, float),
        "TIMEOUT_WRITE": (30.0, float),
        "TIMEOUT_POOL": (10.0, float),
        "DEFAULT_MAX_OUTPUT_TOKENS": (4096, int),
        "PING_INTERVAL": (5.0, float),
        "ADMIN_COOKIE_MAX_AGE": (12 * 60 * 60, int),
        "MAX_QUEUE": (10000, int),
        "BATCH_SIZE": (200, int),
        "INTERVAL": (1.0, float),
        "SUBSCRIBER_QUEUE": (100, int),
        "DRAIN_TIMEOUT": (5.0, float),
        "RECONNECT_SECONDS": (30.0, float),
        "DEFAULT_LIMIT": (64000, int),
        "DEFAULT_HOURS": (24.0, float),
        "MAX_HOURS": (24.0 * 30, float),
        "EXPORT_LIMIT": (5000, int),
        "SEARCH_LIMIT": (50, int),
        "MAX_SEARCH_LIMIT": (500, int),
        "TOOL_LIMIT": (8, int),
        "ROW_LIMIT": (5000, int),
        "ANSWER_PIECES": (20000, int),
        "ANALYSIS_MAX_OUTPUT_TOKENS": (4000, int),
    }

    # Verifica defaults
    for attr, (expected_default, typ) in checks.items():
        val = getattr(cfg, attr)
        assert val == expected_default, f"{attr}: got {val}, expected {expected_default}"
        assert isinstance(val, typ), f"{attr}: type {type(val)} != {typ}"


def test_each_knob_reads_its_own_env_var():
    """Nome errado da env var em qualquer knob deixa o default no lugar:
    sobrecarregar a var correta tem de virar 1."""
    import app.config.config as cfg

    env_names = {
        "MAX_ATTEMPTS": "SHUNT_MAX_ATTEMPTS",
        "RETRY_AFTER_BUDGET": "SHUNT_RETRY_AFTER_BUDGET",
        "TOTAL_DEADLINE": "SHUNT_TOTAL_DEADLINE",
        "FIRST_EVENT_DEADLINE": "SHUNT_FIRST_EVENT_DEADLINE",
        "ENGINE_BOOT_TIMEOUT": "SHUNT_ENGINE_BOOT_TIMEOUT",
        "REACH_TIMEOUT": "SHUNT_REACH_TIMEOUT",
        "BUSY_TIMEOUT_MS": "SHUNT_BUSY_TIMEOUT_MS",
        "TIMEOUT_CONNECT": "SHUNT_TIMEOUT_CONNECT",
        "TIMEOUT_READ": "SHUNT_TIMEOUT_READ",
        "TIMEOUT_WRITE": "SHUNT_TIMEOUT_WRITE",
        "TIMEOUT_POOL": "SHUNT_TIMEOUT_POOL",
        "DEFAULT_MAX_OUTPUT_TOKENS": "SHUNT_DEFAULT_MAX_OUTPUT_TOKENS",
        "PING_INTERVAL": "SHUNT_PING_INTERVAL",
        "ADMIN_COOKIE_MAX_AGE": "SHUNT_ADMIN_COOKIE_MAX_AGE",
        "MAX_QUEUE": "SHUNT_MAX_QUEUE",
        "BATCH_SIZE": "SHUNT_BATCH_SIZE",
        "INTERVAL": "SHUNT_INTERVAL",
        "SUBSCRIBER_QUEUE": "SHUNT_SUBSCRIBER_QUEUE",
        "DRAIN_TIMEOUT": "SHUNT_DRAIN_TIMEOUT",
        "RECONNECT_SECONDS": "SHUNT_RECONNECT_SECONDS",
        "DEFAULT_LIMIT": "SHUNT_BODY_LIMIT",
        "DEFAULT_HOURS": "SHUNT_DEFAULT_HOURS",
        "MAX_HOURS": "SHUNT_MAX_HOURS",
        "EXPORT_LIMIT": "SHUNT_EXPORT_LIMIT",
        "SEARCH_LIMIT": "SHUNT_SEARCH_LIMIT",
        "MAX_SEARCH_LIMIT": "SHUNT_MAX_SEARCH_LIMIT",
        "TOOL_LIMIT": "SHUNT_TOOL_LIMIT",
        "ROW_LIMIT": "SHUNT_DOSIER_ROW_LIMIT",
        "ANSWER_PIECES": "SHUNT_ANSWER_PIECES",
        "ANALYSIS_MAX_OUTPUT_TOKENS": "SHUNT_ANALYSIS_MAX_OUTPUT_TOKENS",
    }
    for attr, env in env_names.items():
        os.environ[env] = "1"
        try:
            reload(cfg)
            assert getattr(cfg, attr) == 1, f"{attr} nao leu {env}"
        finally:
            del os.environ[env]
            reload(cfg)


def test_admin_cookie_constants_not_env():
    """ADMIN_COOKIE, ADMIN_COOKIE_PATH, LOGIN_URL são constantes fixas, não lidas de env."""
    import app.config.config as cfg
    reload(cfg)
    assert cfg.ADMIN_COOKIE == "shunt_admin"
    assert cfg.ADMIN_COOKIE_PATH == "/"
    assert cfg.LOGIN_URL == "/admin/login"