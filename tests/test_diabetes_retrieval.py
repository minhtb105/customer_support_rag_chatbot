"""Diabetes-specific retrieval evaluation — replaces generic Q1-Q20.

Uses data/evaluation/diabetes_retrieval_evaluation.json (15 diabetes Qs).
Falls back to legacy with pytest.mark.legacy if diabetes dataset missing.
"""

import json
import pytest
from pathlib import Path

from src.evaluation import recall_at_k, hit_at_k, mean_reciprocal_rank


DIABETES_DATASET = Path("data/evaluation/diabetes_retrieval_evaluation.json")
LEGACY_DATASET = Path("data/evaluation/retrieval_evaluation.json")


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
