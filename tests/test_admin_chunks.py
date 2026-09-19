"""Admin chunk viewer tests — mocked Chroma collection (no live embeddings)."""
import pytest
from fastapi.testclient import TestClient

import src.admin.tracing_router as tr


class FakeCollection:
    """Minimal chromadb Collection stub: count() + get(where, where_document, limit, offset, include)."""

    def __init__(self, n=25):
        self.calls = []
        self.docs = []
        for i in range(n):
            ds = "byt" if i % 2 == 0 else "ada"
            text = f"chunk {i} about diabetes care"
            if i % 5 == 0:
                text += " HbA1c monitoring guideline"
            self.docs.append({
                "id": f"vec-{i}",
                "doc": text * 20,  # long enough to exceed snippet 500
                "meta": {
                    "source_id": f"doc-{i}",
                    "dataset": ds,
                    "section_path": "A|||B",
                    "page_numbers": "1|||2",
                    "chunk_index": i,
                    "chunk_hash": f"h{i}",
                    "chunking_strategy": "structure",
                },
            })

    def count(self):
        return len(self.docs)

    def get(self, ids=None, where=None, limit=None, offset=None, where_document=None, include=None):
        self.calls.append({"where": where, "where_document": where_document,
                           "limit": limit, "offset": offset, "include": include})
        rows = self.docs
        if where and "dataset" in where:
            rows = [d for d in rows if d["meta"].get("dataset") == where["dataset"]]
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
def fake_collection(monkeypatch):
    fake = FakeCollection(n=25)
    monkeypatch.setattr(tr, "_get_chroma_collection", lambda s: fake)
    tr._COLLECTIONS_CACHE.clear()
    return fake


def _admin_hdr(admin_client):
    return {"Authorization": f"Bearer {admin_client['token']}"}


def test_collections_returns_4_strategies(client: TestClient, admin_client, fake_collection):
    resp = client.get("/v1/admin/collections", headers=_admin_hdr(admin_client))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert [c["strategy"] for c in data["collections"]] == ["structure", "sliding", "semantic", "hybrid_section_semantic"]
    for c in data["collections"]:
        assert c["exists"] is True
        assert c["count"] == 25
        assert sorted(c["datasets"]) == ["ada", "byt"]


def test_chunks_pagination_shape(client: TestClient, admin_client, fake_collection):
    resp = client.get("/v1/admin/collections/structure/chunks?page=1&limit=10", headers=_admin_hdr(admin_client))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["total"] == 25
    assert data["page"] == 1 and data["limit"] == 10
    assert data["total_pages"] == 3
    assert len(data["chunks"]) == 10
    assert data["collection"] == "structure"
    c0 = data["chunks"][0]
    for field in ("id", "vector_id", "content_snippet", "content_full", "source_id", "file_name",
                  "dataset", "section_path", "page_numbers", "chunk_index", "chunk_hash",
                  "chunking_strategy", "embedding_model", "updated_at"):
        assert field in c0, f"missing {field}"
    assert len(c0["content_snippet"]) <= 500
    assert len(c0["content_full"]) <= 4000
    assert c0["section_path"] == ["A", "B"]  # deserialized from "|||"
    assert c0["page_numbers"] == [1, 2]
    # page 3 has remainder
    r3 = client.get("/v1/admin/collections/structure/chunks?page=3&limit=10", headers=_admin_hdr(admin_client))
    assert len(r3.json()["chunks"]) == 5


def test_chunks_dataset_where_mapping(client: TestClient, admin_client, fake_collection):
    resp = client.get("/v1/admin/collections/structure/chunks?dataset=byt&limit=10", headers=_admin_hdr(admin_client))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["total"] == 13  # even indices of 25
    assert all(c["dataset"] == "byt" for c in data["chunks"])
    assert any(call["where"] == {"dataset": "byt"} for call in fake_collection.calls)


def test_chunks_q_where_document_mapping(client: TestClient, admin_client, fake_collection):
    resp = client.get("/v1/admin/collections/structure/chunks?q=HbA1c&limit=10", headers=_admin_hdr(admin_client))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["total"] == 5  # every 5th of 25
    assert data["q"] == "HbA1c"
    assert any(call["where_document"] == {"$contains": "HbA1c"} for call in fake_collection.calls)


def test_chunks_q_stripped_and_dataset_all_no_filter(client: TestClient, admin_client, fake_collection):
    fake_collection.calls.clear()
    resp = client.get("/v1/admin/collections/structure/chunks?q=%20%20HbA1c%20%20&dataset=all",
                      headers=_admin_hdr(admin_client))
    assert resp.status_code == 200, resp.text
    assert resp.json()["q"] == "HbA1c"
    paged = [c for c in fake_collection.calls if c["include"] != []]
    assert paged and paged[0]["where"] is None
    assert paged[0]["where_document"] == {"$contains": "HbA1c"}


def test_chunks_requires_auth(client: TestClient, fake_collection):
    # session client persists login cookies from other tests -> clear to simulate anonymous
    client.cookies.clear()
    resp = client.get("/v1/admin/collections/structure/chunks")
    assert resp.status_code == 401
    resp2 = client.get("/v1/admin/collections")
    assert resp2.status_code == 401


def test_chunks_forbidden_non_admin(client: TestClient, auth_header, fake_collection):
    hdr = {"Authorization": f"Bearer {auth_header['token']}"}
    assert client.get("/v1/admin/collections", headers=hdr).status_code == 403
    assert client.get("/v1/admin/collections/structure/chunks", headers=hdr).status_code == 403


def test_chunks_unknown_strategy_404(client: TestClient, admin_client, fake_collection):
    resp = client.get("/v1/admin/collections/nonexistent/chunks", headers=_admin_hdr(admin_client))
    assert resp.status_code == 404


def test_chunks_missing_collection_empty(client: TestClient, admin_client, monkeypatch):
    monkeypatch.setattr(tr, "_get_chroma_collection", lambda s: None)
    tr._COLLECTIONS_CACHE.clear()
    resp = client.get("/v1/admin/collections/structure/chunks", headers=_admin_hdr(admin_client))
    assert resp.status_code == 200, resp.text
    assert resp.json()["total"] == 0 and resp.json()["chunks"] == []
