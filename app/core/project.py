"""De qual projeto veio a requisicao.

O Claude Code nao manda o diretorio num cabecalho -- ele manda no CORPO, num
bloco de ambiente que vive ora numa mensagem `system` no meio de `messages`, ora
dentro de um `<system-reminder>` da primeira mensagem do usuario. Medido no banco
real: 117 de 261 corpos gravados trazem o bloco, nas duas posicoes, na mesma
versao do cliente. Por isso a varredura passa por tudo em vez de olhar um indice.

Duas decisoes que valem mais que o codigo:

- **A extracao roda na INGESTAO.** Medido: 249 dos 261 `request_json` gravados
  estao cortados em 64.000 chars e nao parseiam como JSON. O corpo inteiro so
  existe no caminho da requisicao, entao o projeto vira coluna ali -- e continua
  funcionando com a gravacao de conversa DESLIGADA, que e o padrao.
- **Nada aqui levanta.** Um rotulo que mude de nome, um corpo de outro cliente,
  um `messages` que nao e lista: tudo vira `None`, e a linha do painel "sem
  projeto" mostra a perda em vez de esconde-la. O proxy nao pode recusar uma
  requisicao porque nao reconheceu o formato do prompt de quem a mandou.
"""

import json
import re
from collections import OrderedDict

# O rotulo do bloco de ambiente. O corte e na quebra de linha, e nao no espaco:
# caminho com espaco existe. `re.MULTILINE` nao e necessario -- o `(?:\r?\n|$)`
# ja fecha no fim da linha ou no fim do texto.
LABEL = re.compile(r"Primary working directory:[ \t]*(.+?)[ \t]*(?:\r?\n|$)")
# Teto do que vai para a coluna. Caminho legitimo nao passa disso; o que passa e
# prompt adversario ou texto colado sem quebra de linha.
MAX_PATH = 512
# Teto do cache de sessao. Sem ele, um proxy ligado por semanas guarda uma
# entrada por sessao para sempre.
SESSION_LIMIT = 1024


def _texts(value) -> list[str]:
    """Todo texto que houver dentro de um campo de conteudo, em ordem."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        found = []
        for block in value:
            if isinstance(block, str):
                found.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                found.append(block["text"])
        return found
    return []


def project_of(body) -> str | None:
    """O diretorio de trabalho declarado no corpo, ou None."""
    if not isinstance(body, dict):
        return None
    candidates = list(_texts(body.get("system")))
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                candidates.extend(_texts(message.get("content")))
    for text in candidates:
        found = LABEL.search(text)
        if found:
            return found.group(1)[:MAX_PATH]
    return None


def session_of(body) -> str | None:
    """A sessao do harness, que o Anthropic carrega como JSON dentro de `user_id`."""
    if not isinstance(body, dict):
        return None
    metadata = body.get("metadata")
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("user_id")
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except ValueError:
        return None
    session = parsed.get("session_id") if isinstance(parsed, dict) else None
    return session if isinstance(session, str) and session else None


class SessionProjects:
    """O projeto de cada sessao vista, para as requisicoes que nao repetem o bloco.

    O bloco de ambiente vem na primeira requisicao de uma sessao do harness; as
    seguintes, nao. Guardar por sessao e o que evita metade das linhas ficarem
    "sem projeto" numa conversa longa.

    Vive no processo: com varios workers, a heranca so vale quando o mesmo
    worker atende as duas requisicoes. Degradacao visivel -- a linha fica "sem
    projeto" -- e nao resultado errado.
    """

    def __init__(self, limit: int = SESSION_LIMIT) -> None:
        self._limit = limit
        self._seen: OrderedDict[str, str] = OrderedDict()

    def remember(self, session: str | None, project: str | None) -> None:
        if not session or not project:
            return
        self._seen[session] = project
        self._seen.move_to_end(session)
        while len(self._seen) > self._limit:
            self._seen.popitem(last=False)

    def recall(self, session: str | None) -> str | None:
        if not session:
            return None
        project = self._seen.get(session)
        if project is not None:
            self._seen.move_to_end(session)
        return project


# A instancia do processo. Uma so, porque o ponto e justamente compartilhar o
# que foi visto entre requisicoes.
SESSIONS = SessionProjects()


def project_and_session(body) -> tuple[str | None, str | None]:
    """O par que o gravador registra, ja com a heranca por sessao aplicada."""
    session = session_of(body)
    project = project_of(body)
    if project:
        SESSIONS.remember(session, project)
    else:
        project = SESSIONS.recall(session)
    return project, session
