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
    # Groq: dialeto OpenAI puro, com tool calling e streaming. O plano gratis
    # limita por minuto e por dia, e nao pede cartao.
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "protocol": "openai",
        "api_key_env": "GROQ_API_KEY",
    },
}

models = {
    # `--alias qwen3.8-27b` no llama.cpp: o nome tem de bater com o que o
    # servidor anuncia, e nao com o arquivo .gguf.
    "qwen-local": {
        "provider": "local",
        "model": "qwen3.8-27b",
        "supports": {"tools": True, "streaming": True, "vision": True},
        "context_window": 262144,
        "max_output_tokens": 8192,
    },
    "free": {
        "provider": "openrouter",
        "model": "openrouter/free",
        "supports": {"tools": True, "streaming": True, "vision": False},
        "context_window": 262144,
        "max_output_tokens": 8192,
    },
    "groq-free": {
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
        "supports": {"tools": True, "streaming": True, "vision": False},
        "context_window": 131072,
        "max_output_tokens": 65536,
    },
}

# Os tres nomes que o Claude Code pede. O padrao casa por substring, entao
# `haiku` pega `claude-haiku-4-5` e qualquer outra versao do mesmo porte.
routes = [
    # Escotilha para forcar o Groq. O nome NAO pode conter o padrao de
    # nenhuma outra rota: a resolucao por familia e por substring, na ordem
    # desta lista, entao um alias chamado `free-groq` casaria com a rota
    # `free` e nunca chegaria no Groq.
    ("groq", ["groq-free"]),
    # Escotilha para forcar o remoto: o alias do modelo tambem e um padrao de
    # rota, entao `model: "free"` sobe direto para o OpenRouter, sem passar
    # pelo local. Serve para conferir a cadeia de fallback sem derrubar nada.
    ("free", ["free"]),
    ("haiku", ["free", "qwen-local", "groq-free"]),
    ("sonnet", ["free", "qwen-local", "groq-free"]),
    ("opus", ["qwen-local", "free", "groq-free"]),
]

default_model = "qwen-local"
