"""Chunking strategy registry + overrides + winner selection (TDD, cheap)."""
import numpy as np
import pytest


def _fake_embed(texts):
    rng = np.random.RandomState(0)
    return rng.rand(len(texts), 8)


TEXT = ("Diabetes care requires monitoring blood glucose every day. "
        "Patients should follow BYT guideline QD5481 for HbA1c targets. "
        "Metformin is the first line therapy for type 2 diabetes. "
        "Complications include retinopathy neuropathy and nephropathy. "
        "Lifestyle change with diet and exercise helps control glucose. " * 4)


def test_registry_resolves_bases():
    from src.shared.chunk_strategies import VARIANTS, ChunkingStrategy, resolve_strategy
    for name in ("structure", "sentence", "sliding", "semantic",
                 "hybrid_section_semantic"):
        assert name in VARIANTS
        base, overrides = resolve_strategy(name)
        assert isinstance(base, ChunkingStrategy)
        assert isinstance(overrides, dict)


def test_registry_unknown_name_raises():
    from src.shared.chunk_strategies import resolve_strategy
    with pytest.raises((KeyError, ValueError)):
        resolve_strategy("structur_typo")


def test_sentence_override_changes_output():
    from src.shared.chunk_strategies import sentence_chunk
    a = sentence_chunk(TEXT, "s", group_size=2)
    b = sentence_chunk(TEXT, "s", group_size=6)
    assert len(a) > len(b) > 0


def test_sliding_override_changes_output():
    from src.shared.chunk_strategies import sliding_window_chunk
    a = sliding_window_chunk(TEXT, "s", window_tokens=50, overlap=10)
    b = sliding_window_chunk(TEXT, "s", window_tokens=200, overlap=10)
    assert len(a) > len(b) > 0


def test_semantic_override_changes_output():
    from src.shared.chunk_strategies import semantic_chunk
    from src.shared import chunk_strategies as cs
    tok = cs.tokenizer.tokenizer
    loose = semantic_chunk(TEXT, "s", embed_fn=_fake_embed,
                           atomic_tokenizer=tok, sim_threshold=0.0)
    tight = semantic_chunk(TEXT, "s", embed_fn=_fake_embed,
                           atomic_tokenizer=tok, sim_threshold=0.9999)
    assert len(tight) >= len(loose) > 0


def test_chunk_document_accepts_overrides():
    from src.shared.chunk_strategies import chunk_document, ChunkingStrategy
    a = chunk_document(None, "doc.pdf", ChunkingStrategy.SENTENCE,
                       raw_text=TEXT, group_size=2)
    b = chunk_document(None, "doc.pdf", ChunkingStrategy.SENTENCE,
                       raw_text=TEXT, group_size=6)
    assert len(a) > len(b) > 0


def test_semantic_blank_atomics_skipped_no_embed_call():
    from src.shared.chunk_strategies import semantic_chunk
    from src.shared import chunk_strategies as cs

    def _boom(texts):
        raise AssertionError("embed_fn must not be called on blank-only text")

    out = semantic_chunk("   \n  \n", "s", embed_fn=_boom,
                         atomic_tokenizer=cs.tokenizer.tokenizer)
    assert out == []


def test_registry_resolves_stage2_variants():
    from src.shared.chunk_strategies import VARIANTS, ChunkingStrategy, resolve_strategy
    for name in ("hybrid__t256", "hybrid__t768", "hybrid__thr080",
                 "hybrid__thr095", "sentence__g2", "sentence__g6",
                 "sentence__g8"):
        assert name in VARIANTS
    base, ov = resolve_strategy("hybrid__t256")
    assert base == ChunkingStrategy.HYBRID_SECTION_SEMANTIC and ov == {"max_tokens": 256}
    base, ov = resolve_strategy("sentence__g6")
    assert base == ChunkingStrategy.SENTENCE and ov == {"group_size": 6}


def test_chunk_document_rejects_bad_override_key():
    import pytest
    from src.shared.chunk_strategies import chunk_document, ChunkingStrategy
    with pytest.raises(ValueError):
        chunk_document(None, "doc.pdf", ChunkingStrategy.SENTENCE,
                       raw_text="Some text here. Second sentence.", bogus_param=1)


def test_pick_winner_recall_then_mrr_then_hit():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from run_benchmarks import pick_winner
    results = {
        "a": {"Recall@K": 0.3, "Hit@K": 0.9, "MRR": 0.9},
        "b": {"Recall@K": 0.3, "Hit@K": 0.5, "MRR": 0.8},
    }
    assert pick_winner(results) == "a"  # tie recall -> higher MRR wins
    results2 = {
        "a": {"Recall@K": 0.3, "Hit@K": 0.9, "MRR": 0.8},
        "b": {"Recall@K": 0.3, "Hit@K": 0.5, "MRR": 0.8},
    }
    assert pick_winner(results2) == "a"  # tie recall+MRR -> higher Hit wins
    results3 = {
        "a": {"Recall@K": 0.2, "Hit@K": 0.9, "MRR": 0.9},
        "b": {"Recall@K": 0.4, "Hit@K": 0.1, "MRR": 0.1},
    }
    assert pick_winner(results3) == "b"  # recall dominates
