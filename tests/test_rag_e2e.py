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
