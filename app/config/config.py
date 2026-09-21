"""O catalogo desta instalacao: provedores, modelos e rotas.

Este e o unico arquivo que muda de maquina para maquina. Nenhuma credencial
mora aqui -- so o NOME da variavel de ambiente que a carrega (`api_key_env`),
lida do `.env`, que esta no `.gitignore`.

A cadeia de cada rota e ordenada: o primeiro candidato que responde vence, e
os seguintes so entram quando o anterior falha de forma que vale repetir. Por
isso o modelo local vem primeiro -- nao custa token nem depende de rede -- e o
remoto e a rede de seguranca.
"""

providers = {
    "local": {
        "base_url": "http://127.0.0.1:8080/v1",
        "protocol": "openai",
        "api_key_env": "LOCAL_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "protocol": "openai",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "protocol": "openai",
        "api_key_env": "GROQ_API_KEY",
    },
}

models = {
    "qwen-local": {
        "provider": "local",
        "model": "qwen3.8-27b",
        "supports": {"tools": True, "streaming": True, "vision": True},
        "context_window": 163840,
        "max_output_tokens": 16384,
    },
    "open-free": {
        "provider": "openrouter",
        "model": "openrouter/free",
        "supports": {"tools": True, "streaming": True, "vision": False},
        "context_window": 262144,
        "max_output_tokens": 8192,
    },
    "open-nemotron-ultra": {
        "provider": "openrouter",
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "supports": {"tools": True, "streaming": True, "vision": False},
        "context_window": 262144,
        "max_output_tokens": 16384,
    },
    "open-deepseek-v4.1-flash": {
        "provider": "openrouter",
        "model": "deepseek/deepseek-v4.1-flash",
        "supports": {"tools": True, "streaming": True, "vision": True},
        "context_window": 262144,
        "max_output_tokens": 16384,
    },
    "open-gpt-oss-120": {
        "provider": "openrouter",
        "model": "openai/gpt-oss-120b",
        "supports": {"tools": True, "streaming": True, "vision": False},
        "context_window": 131072,
        "max_output_tokens": 16384,
    },
    "groq-free": {
        "provider": "groq",
        "model": "groq/compound",
        "supports": {"tools": True, "streaming": True, "vision": False},
        "context_window": 131072,
        "max_output_tokens": 8192,
    },
}

routes = [
    ("groq", ["groq-free"]),
    ("local", ["qwen-local"]),
    ("nemotron", ["open-nemotron-ultra"]),
    ("deepseek", ["open-deepseek-v4.1-flash"]),
    ("gpt-oss", ["open-gpt-oss-120"]),
    ("free", ["open-free", "groq-free", "open-nemotron-ultra"]),
    ("fable", ["open-nemotron-ultra", "open-free", "groq-free"]),
    ("haiku", ["groq-free", "open-free", "open-nemotron-ultra"]),
    ("sonnet", ["open-free", "open-nemotron-ultra", "groq-free"]),
    ("opus", ["qwen-local", "open-nemotron-ultra", "open-free"]),
]

default_model = "open-gpt-oss-120"
