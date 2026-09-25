"""Admin long-term memory facts tests — mocked Chroma collection (no live embeddings)."""
import pytest
from fastapi.testclient import TestClient

import src.admin.tracing_router as tr


class FakeMemoryCollection:
    """Minimal chromadb Collection stub for memory facts."""

    def __init__(self):
        self.calls = []
        self.docs = []
        # 12 facts for user u1 (mixed types), 3 for u2
        types = ["medication", "symptom", "condition", "allergy", "lifestyle", "general"]
        for i in range(12):
            ft = types[i % len(types)]
            text = f"fact {i} about diabetes {ft}"
            if i % 4 == 0:
                text += " HbA1c monitoring"
            meta = {
                "fact_id": f"f{i}",
                "user_id": "u1",
                "fact_type": ft,
                "entities": "metformin,headache" if i % 3 == 0 else "",
                "source": "conversation",
                "timestamp": 1700000000.0 + i,
                "confidence": 0.8,
            }
            if i == 5:
                # missing metadata fail-open case
                meta = {"user_id": "u1"}
            self.docs.append({"id": f"f{i}", "doc": text * 30, "meta": meta})
        for i in range(3):
            self.docs.append({
                "id": f"g{i}", "doc": f"other user fact {i}",
                "meta": {"fact_id": f"g{i}", "user_id": "u2", "fact_type": "general",
                         "entities": "", "source": "conversation",
                         "timestamp": 1700000000.0, "confidence": 0.9},
            })

    def get(self, ids=None, where=None, limit=None, offset=None, where_document=None, include=None):
        self.calls.append({"where": where, "where_document": where_document,
                           "limit": limit, "offset": offset, "include": include})
        rows = self.docs
        if where:
            conds = where.get("$and") if isinstance(where, dict) and "$and" in where else None
            if conds:
                for c in conds:
                    for k, v in c.items():
                        rows = [d for d in rows if d["meta"].get(k) == v]
            else:
                for k, v in where.items():
                    rows = [d for d in rows if d["meta"].get(k) == v]
        if where_document and "$contains" in where_document:
            needle = where_document["$contains"]
            rows = [d for d in rows if needle in d["doc"]]
        if include == []:
            return {"ids": [d["id"] for d in rows]}
        start = offset or 0
        end = start + limit if limit is not None else None
        page = rows[start:end]
        return {"ids": [d["id"] for d in page],
                "documents": [d["doc"] for d in page],
                "metadatas": [dict(d["meta"]) for d in page]}


@pytest.fixture
def fake_memory(monkeypatch):
    fake = FakeMemoryCollection()
    monkeypatch.setattr(tr, "_get_memory_collection", lambda: fake)
    return fake


def _admin_hdr(admin_client):
    return {"Authorization": f"Bearer {admin_client['token']}"}


def test_facts_pagination_shape(client: TestClient, admin_client, fake_memory):
    resp = client.get("/v1/admin/memory/facts?user_id=u1&page=1&limit=10", headers=_admin_hdr(admin_client))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["total"] == 12
    assert data["page"] == 1 and data["limit"] == 10
    assert data["total_pages"] == 2
    assert len(data["facts"]) == 10
    assert data["user_id"] == "u1"
    f0 = data["facts"][0]
    for field in ("fact_id", "text_snippet", "text_full", "fact_type", "entities",
                  "confidence", "source", "timestamp", "temporal_weight"):
        assert field in f0, f"missing {field}"
    assert len(f0["text_snippet"]) <= 500
    assert len(f0["text_full"]) <= 4000
    r2 = client.get("/v1/admin/memory/facts?user_id=u1&page=2&limit=10", headers=_admin_hdr(admin_client))
    assert len(r2.json()["facts"]) == 2


def test_facts_only_requested_user(client: TestClient, admin_client, fake_memory):
    data = client.get("/v1/admin/memory/facts?user_id=u2", headers=_admin_hdr(admin_client)).json()
    assert data["total"] == 3
    assert all(f["fact_id"].startswith("g") for f in data["facts"])


def test_facts_fact_type_where_mapping(client: TestClient, admin_client, fake_memory):
    resp = client.get("/v1/admin/memory/facts?user_id=u1&fact_type=medication", headers=_admin_hdr(admin_client))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["total"] == 2  # indices 0, 6
    assert all(f["fact_type"] == "medication" for f in data["facts"])
    paged = [c for c in fake_memory.calls if c["include"] != []]
    assert paged and paged[0]["where"] == {"$and": [{"user_id": "u1"}, {"fact_type": "medication"}]}


def test_facts_fact_type_all_no_filter(client: TestClient, admin_client, fake_memory):
    fake_memory.calls.clear()
    resp = client.get("/v1/admin/memory/facts?user_id=u1&fact_type=all", headers=_admin_hdr(admin_client))
    assert resp.status_code == 200, resp.text
    paged = [c for c in fake_memory.calls if c["include"] != []]
    assert paged and paged[0]["where"] == {"user_id": "u1"}


def test_facts_q_where_document_mapping(client: TestClient, admin_client, fake_memory):
    resp = client.get("/v1/admin/memory/facts?user_id=u1&q=HbA1c", headers=_admin_hdr(admin_client))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["total"] == 3  # every 4th of 12
    assert data["q"] == "HbA1c"
    assert any(call["where_document"] == {"$contains": "HbA1c"} for call in fake_memory.calls)


def test_facts_q_stripped_empty_no_filter(client: TestClient, admin_client, fake_memory):
    fake_memory.calls.clear()
    resp = client.get("/v1/admin/memory/facts?user_id=u1&q=%20%20%20", headers=_admin_hdr(admin_client))
    assert resp.status_code == 200, resp.text
    paged = [c for c in fake_memory.calls if c["include"] != []]
    assert paged and paged[0]["where_document"] is None


def test_facts_missing_user_id_422(client: TestClient, admin_client, fake_memory):
    resp = client.get("/v1/admin/memory/facts", headers=_admin_hdr(admin_client))
    assert resp.status_code == 422
    resp2 = client.get("/v1/admin/memory/facts?user_id=%20%20", headers=_admin_hdr(admin_client))
    assert resp2.status_code == 422


def test_facts_requires_auth(client: TestClient, fake_memory):
    client.cookies.clear()
    assert client.get("/v1/admin/memory/facts?user_id=u1").status_code == 401


def test_facts_forbidden_non_admin(client: TestClient, auth_header, fake_memory):
    hdr = {"Authorization": f"Bearer {auth_header['token']}"}
    assert client.get("/v1/admin/memory/facts?user_id=u1", headers=hdr).status_code == 403


def test_facts_missing_collection_empty(client: TestClient, admin_client, monkeypatch):
    monkeypatch.setattr(tr, "_get_memory_collection", lambda: None)
    resp = client.get("/v1/admin/memory/facts?user_id=u1", headers=_admin_hdr(admin_client))
    assert resp.status_code == 200, resp.text
    assert resp.json()["total"] == 0 and resp.json()["facts"] == []


def test_facts_entities_empty_and_missing_metadata_fail_open(client: TestClient, admin_client, fake_memory):
    data = client.get("/v1/admin/memory/facts?user_id=u1&page=1&limit=10", headers=_admin_hdr(admin_client)).json()
    by_id = {f["fact_id"]: f for f in data["facts"]}
    # f1 has entities "" -> []
    assert by_id["f1"]["entities"] == []
    # f0 has csv entities
    assert by_id["f0"]["entities"] == ["metformin", "headache"]
    # f5 missing metadata -> defaults, no crash
    assert by_id["f5"]["entities"] == []
    assert by_id["f5"]["confidence"] is None
    assert by_id["f5"]["fact_type"] is None


def test_ragas_excludes_memory_chunks(client: TestClient, admin_client, monkeypatch):
    import uuid as _uuid
    from src.shared.observability import tracing_db as tdb
    trace_id = tdb.create_trace(user_id="u1", username="u1", query="q?", tone="balanced",
                                model="m", embedding_model="e", chunking_strategy="structure", top_k=5)
    span = tdb.create_span(trace_id=trace_id, name="retrieve_context")
    tdb.add_span_chunks(span, trace_id, [
        {"source_id": "doc-1", "content": "real guideline content", "dataset": "byt"},
        {"source_id": "short_term_memory", "content": "memory stuff", "dataset": "short_term_memory"},
        {"source_id": "long_term_memory", "content": "memory facts", "dataset": "long_term_memory"},
    ])
    tdb.update_trace(trace_id, answer="answer text here")
    seen = {}

    def fake_eval(query, answer, ctxs):
        seen["ctxs"] = ctxs
        return {"metrics": {"faithfulness": 4.0, "context_precision": 4.0, "context_recall": 4.0,
                            "answer_relevance": 4.0},
                "comments": {}, "failed_metrics": [], "confidence": 4.0,
                "raw": {}, "evaluator_model": "test", "is_low_confidence": False, "routed_role": "doctor"}

    monkeypatch.setattr(tr, "evaluate_rag", fake_eval)
    resp = client.post(f"/v1/admin/traces/{trace_id}/ragas", headers=_admin_hdr(admin_client))
    assert resp.status_code == 200, resp.text
    ctxs = seen["ctxs"]
    assert len(ctxs) == 1
    assert all("memory" not in (c.get("source_id") or "") for c in ctxs)
