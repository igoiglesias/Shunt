import json

from fastapi.testclient import TestClient

from tools.harness_stub import build_stub, mask_headers


def test_credentials_are_masked_but_names_kept():
    out = mask_headers({"Authorization": "Bearer sk-x", "x-api-key": "k", "x-shunt-token": "t", "user-agent": "codex/1"})
    assert out["authorization"] == "***" and out["x-api-key"] == "***" and out["x-shunt-token"] == "***"
    assert out["user-agent"] == "codex/1"

def test_every_request_is_logged_with_method_path_query_and_masked_headers(tmp_path):
    log = tmp_path / "stub.jsonl"
    with TestClient(build_stub(log)) as c:
        r = c.post("/t/abc/v1/responses?token=q", json={"model": "gpt-5", "input": "oi", "stream": True},
                   headers={"authorization": "Bearer sk"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    line = json.loads(log.read_text().splitlines()[0])
    assert line["method"] == "POST" and line["path"] == "/t/***/v1/responses" and line["query"] == "token=***"
    assert line["headers"]["authorization"] == "***"
    assert "response.output_item.added" in r.text and "response.completed" in r.text
