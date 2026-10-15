"""Admin trace feedback + list filters (sort/from-to/q-id) tests.

Sort-whitelist policy (documented choice): invalid `sort_by`/`order` -> 422
(strict whitelist). `list_traces()` itself falls back to defaults defensively;
the API layer enforces the 422.
"""
from fastapi.testclient import TestClient

import src.shared.observability.tracing_db as tdb


def _admin_hdr(admin_client):
    return {"Authorization": f"Bearer {admin_client['token']}"}


def _mk_trace(query="q", answer="a", latency=None, created_at=None):
    tid = tdb.create_trace(user_id="u1", username="u1", query=query, tone="diabetes",
                           model="m", embedding_model="e", chunking_strategy="s", top_k=5)
    fields = {"answer": answer}
    if latency is not None:
        fields["total_latency_ms"] = latency
    if created_at is not None:
        fields["created_at"] = created_at
    tdb.update_trace(tid, **fields)
    if created_at is not None:
        # update_trace only touches updated_at + given fields; force created_at
        # via direct SQL since update_trace allowlists nothing (passes through).
        import sqlite3
        conn = sqlite3.connect(str(tdb.TRACING_DB_PATH))
        conn.execute("UPDATE traces SET created_at=? WHERE id=?", (created_at, tid))
        conn.commit()
        conn.close()
    return tid


def test_feedback_post_201_and_listed(client: TestClient, admin_client):
    tid = _mk_trace(query="fb q unique 1", answer="fb a")
    resp = client.post(f"/v1/admin/traces/{tid}/feedback",
                       json={"score": 5, "comment": "great answer"},
                       headers=_admin_hdr(admin_client))
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["trace_id"] == tid and data["score"] == 5
    # newest-first list in detail response
    client.post(f"/v1/admin/traces/{tid}/feedback", json={"score": 2, "comment": "meh"},
                headers=_admin_hdr(admin_client))
    detail = client.get(f"/v1/admin/traces/{tid}", headers=_admin_hdr(admin_client))
    assert detail.status_code == 200, detail.text
    fb = detail.json()["feedback"]
    assert len(fb) == 2
    assert fb[0]["score"] == 2 and fb[1]["score"] == 5  # DESC


def test_feedback_bad_score_422(client: TestClient, admin_client):
    tid = _mk_trace(query="fb q unique 2", answer="fb a")
    for bad in (0, 6, -1, 10):
        resp = client.post(f"/v1/admin/traces/{tid}/feedback", json={"score": bad},
                           headers=_admin_hdr(admin_client))
        assert resp.status_code == 422, (bad, resp.text)


def test_feedback_requires_auth(client: TestClient):
    tid = _mk_trace(query="fb q unique 3", answer="fb a")
    client.cookies.clear()
    assert client.post(f"/v1/admin/traces/{tid}/feedback", json={"score": 4}).status_code == 401


def test_feedback_forbidden_non_admin(client: TestClient, auth_header):
    tid = _mk_trace(query="fb q unique 4", answer="fb a")
    hdr = {"Authorization": f"Bearer {auth_header['token']}"}
    assert client.post(f"/v1/admin/traces/{tid}/feedback", json={"score": 4}, headers=hdr).status_code == 403


def test_feedback_unknown_trace_404(client: TestClient, admin_client):
    resp = client.post("/v1/admin/traces/does-not-exist/feedback", json={"score": 4},
                       headers=_admin_hdr(admin_client))
    assert resp.status_code == 404


def test_from_to_filter(client: TestClient, admin_client):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    iso = lambda d: d.isoformat()
    from datetime import timedelta
    t_old = _mk_trace(query="fromto marker old", answer="x",
                      created_at=iso(now - timedelta(days=5)))
    t_new = _mk_trace(query="fromto marker new", answer="x",
                      created_at=iso(now - timedelta(days=1)))
    mid_from = iso(now - timedelta(days=3))
    resp = client.get("/v1/admin/traces",
                      params={"q": "fromto marker", "from": mid_from, "limit": 10},
                      headers=_admin_hdr(admin_client))
    assert resp.status_code == 200, resp.text
    ids = [t["id"] for t in resp.json()["traces"]]
    assert t_new in ids and t_old not in ids
    # from > to -> 422
    bad = client.get("/v1/admin/traces",
                     params={"from": iso(now), "to": iso(now - timedelta(days=1))},
                     headers=_admin_hdr(admin_client))
    assert bad.status_code == 422


def test_q_matches_id_exact(client: TestClient, admin_client):
    tid = _mk_trace(query="idmatch query zzz-unique", answer="idmatch answer")
    resp = client.get("/v1/admin/traces", params={"q": tid, "limit": 10},
                      headers=_admin_hdr(admin_client))
    assert resp.status_code == 200, resp.text
    ids = [t["id"] for t in resp.json()["traces"]]
    assert tid in ids


def test_sort_by_latency(client: TestClient, admin_client):
    _mk_trace(query="sortmark low", answer="x", latency=10)
    _mk_trace(query="sortmark high", answer="x", latency=900)
    _mk_trace(query="sortmark mid", answer="x", latency=100)
    resp = client.get("/v1/admin/traces",
                      params={"q": "sortmark", "sort_by": "total_latency_ms", "order": "asc", "limit": 10},
                      headers=_admin_hdr(admin_client))
    assert resp.status_code == 200, resp.text
    lats = [t["total_latency_ms"] for t in resp.json()["traces"]]
    assert lats == sorted(lats), lats


def test_sort_whitelist_invalid_422(client: TestClient, admin_client):
    r1 = client.get("/v1/admin/traces", params={"sort_by": "query"},
                    headers=_admin_hdr(admin_client))
    assert r1.status_code == 422
    r2 = client.get("/v1/admin/traces", params={"order": "sideways"},
                    headers=_admin_hdr(admin_client))
    assert r2.status_code == 422
