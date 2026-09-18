providers = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "protocol": "openai",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "protocol": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
}

models = {
    "free": {
        "provider": "openrouter",
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "supports": {"tools": True, "streaming": True, "vision": False},
        "context_window": 64000,
        "max_output_tokens": 8192,
    },
}

routes = [
    ("haiku", ["free"]),
]

default_model = None

# TODO(task-4): app/routers/v1.py still reads `model_sources`; that router is
# rewritten in a later task to consume `load_settings()` instead. Kept here
# only to avoid breaking the existing import until then.
model_sources = [
    {
        "provider": "Openrouter",
        "model": models["free"]["model"],
        "api_key": "",
    }
]
