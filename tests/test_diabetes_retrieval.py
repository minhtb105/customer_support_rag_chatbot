"""Retrieval evaluation for 4 diseases — diabetes + hypertension + respiratory + mental.

- diabetes: data/evaluation/diabetes_retrieval_evaluation.json (15 Qs)
- hypertension: data/evaluation/hypertension_retrieval_evaluation.json (10 Qs)
- respiratory: data/evaluation/respiratory_retrieval_evaluation.json (10 Qs)
- mental: data/evaluation/mental_retrieval_evaluation.json (10 Qs)
Legacy generic Q1-Q20 kept as `benchmark_questions.json` (skip via `pytest.mark.legacy`).
"""

import json
import pytest
from pathlib import Path

from src.evaluation import recall_at_k, hit_at_k, mean_reciprocal_rank


DIABETES_DATASET = Path("data/evaluation/diabetes_retrieval_evaluation.json")
HYPERTENSION_DATASET = Path("data/evaluation/hypertension_retrieval_evaluation.json")
RESPIRATORY_DATASET = Path("data/evaluation/respiratory_retrieval_evaluation.json")
MENTAL_DATASET = Path("data/evaluation/mental_retrieval_evaluation.json")
LEGACY_DATASET = Path("data/evaluation/retrieval_evaluation.json")
BENCHMARK_LEGACY = Path("data/evaluation/benchmark_questions.json")


def load_dataset(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def test_diabetes_retrieval_files_exist():
    assert DIABETES_DATASET.exists(), f"Missing {DIABETES_DATASET} — run create script"
    data = load_dataset(DIABETES_DATASET)
    assert len(data) >= 10
    for item in data:
        assert "question_id" in item
        assert "question" in item
        assert "relevant_documents" in item
        assert any("diabetes" in d.lower() or "pen" in d.lower() or "classification" in d.lower() or "hearts" in d.lower() for d in item["relevant_documents"]), f"Non-diabetes doc in {item['question_id']}: {item['relevant_documents']}"


def test_diabetes_topics_are_diabetes_only():
    data = load_dataset(DIABETES_DATASET)
    allowed = {"diabetes_classification", "diabetes_diagnosis", "diabetes_management", "diabetes_complications", "diabetes_adherence", "diabetes_market", "diabetes_policy", "diabetes_guideline", "ncd_pen", "ada_soc"}
    for item in data:
        assert item.get("topic") in allowed or "diabetes" in item.get("topic", ""), f"Unexpected topic {item.get('topic')} in {item['question_id']}"


def test_hypertension_retrieval_files_exist():
    assert HYPERTENSION_DATASET.exists(), f"Missing {HYPERTENSION_DATASET}"
    data = load_dataset(HYPERTENSION_DATASET)
    assert len(data) >= 8
    for item in data:
        assert "question_id" in item and item["question_id"].startswith("HQ")
        assert "relevant_documents" in item
        assert any("hypertension" in d.lower() or "aha" in d.lower() or "hearts" in d.lower() or "pen" in d.lower() or "3192" in d.lower() for d in item["relevant_documents"]), f"Non-hypertension doc in {item['question_id']}"
    topics = {i["topic"] for i in data}
    assert any("hypertension" in t for t in topics)


def test_respiratory_retrieval_files_exist():
    assert RESPIRATORY_DATASET.exists(), f"Missing {RESPIRATORY_DATASET}"
    data = load_dataset(RESPIRATORY_DATASET)
    assert len(data) >= 8
    for item in data:
        assert "question_id" in item and item["question_id"].startswith("RQ")
        assert any("gold" in d.lower() or "gina" in d.lower() or "copd" in d.lower() or "pen" in d.lower() for d in item["relevant_documents"]), f"Non-respiratory doc in {item['question_id']}"
    assert any("respiratory" in i["topic"] or "gold" in i["topic"] or "gina" in i["topic"] for i in data)


def test_mental_retrieval_files_exist():
    assert MENTAL_DATASET.exists(), f"Missing {MENTAL_DATASET}"
    data = load_dataset(MENTAL_DATASET)
    assert len(data) >= 8
    for item in data:
        assert "question_id" in item and item["question_id"].startswith("MQ")
        assert any("mhgap" in d.lower() or "mental" in d.lower() or "pen" in d.lower() for d in item["relevant_documents"]), f"Non-mental doc in {item['question_id']}"
    assert any("mental" in i["topic"] for i in data)


def test_all_four_diseases_covered():
    """Coverage gate: 4 diseases must have retrieval sets."""
    for path in [DIABETES_DATASET, HYPERTENSION_DATASET, RESPIRATORY_DATASET, MENTAL_DATASET]:
        assert path.exists(), f"Missing {path}"
        data = load_dataset(path)
        assert len(data) >= 8, f"{path} too small: {len(data)}"
    total = sum(len(load_dataset(p)) for p in [DIABETES_DATASET, HYPERTENSION_DATASET, RESPIRATORY_DATASET, MENTAL_DATASET])
    assert total >= 40, f"Expected >=40 Qs across 4 diseases, got {total}"


def test_recall_metrics_sanity():
    # Synthetic sanity: perfect retrieval should give 1.0 (requires ContextItem list)
    from src.models.llm_io import ContextItem
    retrieved = [ContextItem(source_id="a", content="x"), ContextItem(source_id="b", content="y")]
    assert recall_at_k(retrieved, ["a.pdf"], k=5) == 1.0
    assert hit_at_k(retrieved, ["c.pdf"], k=5) == 0
    assert mean_reciprocal_rank([ContextItem(source_id="doc", content="x")], ["doc.pdf"], k=5) == 1.0


@pytest.mark.legacy
def test_legacy_generic_metrics_still_importable():
    if not LEGACY_DATASET.exists():
        pytest.skip("Legacy dataset not present")
    data = load_dataset(LEGACY_DATASET)
    assert len(data) == 20


@pytest.mark.legacy
def test_benchmark_questions_legacy_kept_skipped():
    """P0 decision: benchmark_questions.json 5 Qs generic kept as legacy (skip unless explicitly run)."""
    if not BENCHMARK_LEGACY.exists():
        pytest.skip("Legacy benchmark not present")
    data = load_dataset(BENCHMARK_LEGACY)
    assert len(data) == 5
    # ensure legacy topics are not in new 4-disease allowlist
    for item in data:
        assert item.get("id", "").startswith("q")
