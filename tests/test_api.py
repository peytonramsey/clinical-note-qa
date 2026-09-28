from fastapi.testclient import TestClient

import src.serve as serve


class FakeLLM:
    def __init__(self):
        self.calls = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": " amoxicillin \n"}}]}


def client_with_fake(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(serve, "get_llm", lambda: fake)
    return TestClient(serve.app), fake


def test_answer_endpoint(monkeypatch):
    client, fake = client_with_fake(monkeypatch)
    r = client.post("/answer", json={"note": "Discharge Medications: amoxicillin",
                                     "question": "What medication?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "amoxicillin"
    assert body["sources"] == ["Discharge Medications: amoxicillin"]
    assert body["latency_ms"] >= 0
    assert fake.calls[0]["temperature"] == 0


def test_retrieval_sends_only_top_sections(monkeypatch):
    client, fake = client_with_fake(monkeypatch)
    note = "\n\n".join(f"Section {i}: filler text about topic number {i} for the record" for i in range(6))
    note += "\n\nDischarge Medications: amoxicillin 500 mg three times daily for five days"
    r = client.post("/answer", json={"note": note, "question": "What discharge medications?",
                                     "use_retrieval": True})
    assert r.status_code == 200
    sources = r.json()["sources"]
    assert len(sources) == serve.TOP_K
    assert any("amoxicillin" in s for s in sources)
    prompt = fake.calls[0]["messages"][1]["content"]
    assert all(s in prompt for s in sources) and len(prompt) < len(note)  # only the top sections are sent


def test_empty_question_is_rejected(monkeypatch):
    client, _ = client_with_fake(monkeypatch)
    r = client.post("/answer", json={"note": "Some note", "question": ""})
    assert r.status_code == 422


def test_health(monkeypatch):
    client, _ = client_with_fake(monkeypatch)
    assert client.get("/health").json()["status"] == "ok"
