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
