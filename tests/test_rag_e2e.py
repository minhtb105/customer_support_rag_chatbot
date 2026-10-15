"""E2E rag_chat: cau hoi so lieu (GHO tool) + cau hoi guideline (BYT RAG).

Mock `generate_answer` (LLM) de test deterministic; giu retrieval that
(Chroma + OpenAI embeddings) va GHO SQLite tool that de verify wiring.
"""
import uuid

from src.chat import rag_pipeline as rp
from src.shared.models.llm_io import LLMOutput


def _run(question: str, monkeypatch):
    captured = {}

    def fake_generate(llm_input, model=None):
        captured["contexts"] = list(llm_input.contexts)
        as_dicts = [c.model_dump() if hasattr(c, "model_dump") else c
                    for c in llm_input.contexts]
        return LLMOutput(answer="mocked", cited_sources=[], contexts=as_dicts)

    monkeypatch.setattr(rp, "generate_answer", fake_generate)
    uid = uuid.uuid4().hex[:8]
    res = rp.rag_chat(f"{question} [{uid}]", top_k=5)
    assert res.get("trace_id")
    captured["trace_id"] = res["trace_id"]
    return captured


def test_e2e_stats_question_injects_gho(monkeypatch):
    captured = _run("Ty le dai thao duong o Viet Nam hien nay la bao nhieu?", monkeypatch)
    first = captured["contexts"][0]
    assert first.source_id == "gho_stats"
    assert "%" in first.content and "WHO" in first.content


def test_e2e_guideline_question_hits_byt(monkeypatch):
    captured = _run(
        "Theo Bo Y te, glucose huyet tuong luc doi bao nhieu thi chan doan dai thao duong?",
        monkeypatch,
    )
    assert not any(c.source_id == "gho_stats" for c in captured["contexts"][:1])
    assert any(
        "qd3319" in (c.file_name or "").lower() or "qd5481" in (c.file_name or "").lower()
        for c in captured["contexts"]
    ), [c.file_name for c in captured["contexts"]]


def test_e2e_spans_nest_under_root(monkeypatch):
    """Phase B: pipeline spans con nest duoi root rag_chat (flat cu khong co)."""
    captured = _run("Duong huyet luc doi bao nhieu la binh thuong?", monkeypatch)
    import sqlite3
    from src.shared.config import TRACING_DB_PATH
    con = sqlite3.connect(str(TRACING_DB_PATH))
    rows = con.execute(
        "SELECT id, name, parent_id FROM spans WHERE trace_id=?",
        (captured["trace_id"],)).fetchall()
    con.close()
    by_id = {r[0]: (r[1], r[2]) for r in rows}
    roots = [i for i, (_, p) in by_id.items() if p is None]
    assert len(roots) == 1 and by_id[roots[0]][0] == "rag_chat"
    for i, (_, p) in by_id.items():
        if i != roots[0]:
            assert p in by_id, f"span {i} tro toi parent khong ton tai"
