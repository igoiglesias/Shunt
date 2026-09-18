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
    # llama.cpp servindo local, dialeto OpenAI. A chave existe porque o
    # `llama serve` foi subido com `--api-key`, e nao porque haja segredo:
    # um proxy que fale com ele sem o cabecalho leva 401.
    "local": {
        "base_url": "http://127.0.0.1:8181/v1",
        "protocol": "openai",
        "api_key_env": "LOCAL_API_KEY",
    },
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
    # `--alias qwen3.8-27b` no llama.cpp: o nome tem de bater com o que o
    # servidor anuncia, e nao com o arquivo .gguf.
    "qwen-local": {
        "provider": "local",
        "model": "qwen3.8-27b",
        "supports": {"tools": True, "streaming": True, "vision": True},
        "context_window": 32000,
        "max_output_tokens": 8192,
    },
    "free": {
        "provider": "openrouter",
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "supports": {"tools": True, "streaming": True, "vision": False},
        "context_window": 64000,
        "max_output_tokens": 8192,
    },
}

# Os tres nomes que o Claude Code pede. O padrao casa por substring, entao
# `haiku` pega `claude-haiku-4-5` e qualquer outra versao do mesmo porte.
routes = [
    ("haiku", ["qwen-local", "free"]),
    ("sonnet", ["qwen-local", "free"]),
    ("opus", ["qwen-local", "free"]),
]

default_model = "qwen-local"
