"""Span nesting via ContextVar stack (Phase B) — tmp DB, never touches dev DB."""
import pytest


@pytest.fixture()
def tmp_tracing(monkeypatch, tmp_path):
    db = tmp_path / "tracing_test.db"
    mods = []
    import src.shared.observability.tracing_db as t1
    mods.append(t1)
    try:
        import shared.observability.tracing_db as t2
        if t2 is not t1:
            mods.append(t2)
    except ImportError:
        pass
    for m in mods:
        monkeypatch.setattr(m, "TRACING_DB_PATH", db)
    t1.init_tracing_db()
    # reset contextvars (tests share process)
    import src.shared.observability.local_tracing as lt
    lt._trace_ctx.set(None)
    lt._span_ctx.set(())
    return t1


def _spans(t1, trace_id):
    rows = t1._conn().execute(
        "SELECT id, name, parent_id FROM spans WHERE trace_id=? ORDER BY start_at",
        (trace_id,)).fetchall()
    return [(r[1], r[2]) for r in rows]


def test_nested_spans_get_parent_ids(tmp_tracing):
    import src.shared.observability.local_tracing as lt
    tid = lt.start_trace(user_id="u", username=None, query="q")
    a = lt.start_span("a")
    b = lt.start_span("b")
    lt.finish_span(b)
    c = lt.start_span("c")
    lt.finish_span(c)
    lt.finish_span(a)
    lt.end_trace(tid, answer="ok")
    got = dict()
    for name, parent in _spans(tmp_tracing, tid):
        got[name] = parent
    assert got["a"] is None
    assert got["b"] == _id_of(tmp_tracing, tid, "a")
    assert got["c"] == _id_of(tmp_tracing, tid, "a")


def _id_of(t1, trace_id, name):
    return t1._conn().execute(
        "SELECT id FROM spans WHERE trace_id=? AND name=?", (trace_id, name)).fetchone()[0]


def test_out_of_order_finish_safe(tmp_tracing):
    import src.shared.observability.local_tracing as lt
    tid = lt.start_trace(user_id="u", username=None, query="q")
    a = lt.start_span("a")
    b = lt.start_span("b")
    lt.finish_span(a)  # out-of-order: đóng cha trước
    lt.finish_span(b)  # vẫn end_span bình thường, không crash
    lt.end_trace(tid)
    rows = _spans(tmp_tracing, tid)
    assert len(rows) == 2


def test_noop_without_trace(tmp_tracing):
    import src.shared.observability.local_tracing as lt
    sid = lt.start_span("x")
    assert sid.startswith("noop_")
    lt.finish_span(sid)  # no-op, không crash


def test_end_trace_resets_stack(tmp_tracing):
    import src.shared.observability.local_tracing as lt
    tid = lt.start_trace(user_id="u", username=None, query="q")
    lt.start_span("a")
    lt.end_trace(tid)
    assert lt._span_ctx.get() == ()
    tid2 = lt.start_trace(user_id="u", username=None, query="q2")
    lt.start_span("b")
    lt.end_trace(tid2)
    rows = _spans(tmp_tracing, tid2)
    assert rows == [("b", None)]
