"""Single medical entity extractor shared by short-term/long-term/episodic.

Unifies the three keyword/regex variants that lived in
short_term.py / long_term.py / episodic.py (same matching semantics:
case-insensitive substring over a keyword union incl. Vietnamese diabetes
terms). Old class names remain importable as aliases for compat.
"""
from __future__ import annotations

from typing import Dict, List


class MedicalEntityExtractor:
    """Keyword medical entity extractor + importance/fact-type/fact helpers."""

    medical_keywords: Dict[str, List[str]] = {
        "medications": [
            "paracetamol", "ibuprofen", "aspirin", "antibiotic", "insulin",
            "metformin", "blood pressure", "cholesterol", "antidepressant",
            "painkiller", "medication", "drug", "treatment", "prescription",
            "medicine", "dose", "dosage", "pill", "tablet", "capsule",
            "injection",
            # Vietnamese (diabetes patients)
            "thuốc", "liều", "viên", "tiêm", "uống thuốc", "đơn thuốc",
        ],
        "symptoms": [
            "headache", "fever", "cough", "pain", "nausea", "dizziness",
            "fatigue", "shortness of breath", "chest pain", "rash",
            "symptom", "discomfort", "ache", "illness", "condition",
            "feeling", "problem", "issue", "complaint",
            # Vietnamese
            "mệt", "chóng mặt", "khát nước", "tiểu nhiều", "mờ mắt",
            "tê bì", "đau đầu", "hạ đường huyết", "tăng đường huyết",
            "đo đường huyết",
        ],
        "conditions": [
            "diabetes", "hypertension", "asthma", "arthritis", "depression",
            "anxiety", "cancer", "heart disease", "stroke", "infection",
            "disease", "disorder", "syndrome", "diagnosis", "medical issue",
            "health problem",
            # Vietnamese
            "tiểu đường", "đái tháo đường", "huyết áp", "mỡ máu", "béo phì",
            "suy thận", "tim mạch", "gan nhiễm mỡ", "tiền tiểu đường",
        ],
        "allergies": [
            "allergy", "allergic", "sensitivity", "reaction",
            "penicillin", "peanut", "shellfish", "pollen",
            # Vietnamese
            "dị ứng", "mẩn ngứa", "phát ban",
        ],
        "lifestyle": [
            "diet", "exercise", "sleep", "stress", "smoking", "alcohol",
            "lifestyle", "habits", "routine", "wellness",
            # Vietnamese
            "ăn ngọt", "ăn kiêng", "tập thể dục", "chạy bộ", "đi bộ",
            "mất ngủ", "rượu", "bia", "thuốc lá", "trà sữa", "chế độ ăn",
            "giảm cân", "bỏ bữa",
        ],
        "medical_terms": [
            "dose", "side effect", "allergy", "prescription", "diagnosis",
            "treatment", "symptom", "condition", "medication", "drug",
        ],
        "temporal": [
            "yesterday", "today", "tomorrow", "last week", "recently",
            "when", "since", "for", "ago", "always", "sometimes", "never",
            "usually",
        ],
    }

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Case-insensitive substring match per category (deduped, ordered)."""
        text_lower = (text or "").lower()
        entities: Dict[str, List[str]] = {}
        for category, keywords in self.medical_keywords.items():
            found = [kw for kw in dict.fromkeys(keywords) if kw in text_lower]
            if found:
                entities[category] = found
        return entities

    def calculate_importance_score(self, text: str) -> float:
        """Importance 0.5 base + per-entity boosts, capped at 1.0."""
        entities = self.extract_entities(text)
        score = 0.5
        if "medications" in entities:
            score += 0.3 * len(entities["medications"])
        if "symptoms" in entities:
            score += 0.2 * len(entities["symptoms"])
        if "conditions" in entities:
            score += 0.4 * len(entities["conditions"])
        if "medical_terms" in entities:
            score += 0.1 * len(entities["medical_terms"])
        if "allergies" in entities:
            score += 0.2 * len(entities["allergies"])
        if "lifestyle" in entities:
            score += 0.1 * len(entities["lifestyle"])
        return min(score, 1.0)

    def classify_fact_type(self, entities: Dict[str, List[str]]) -> str:
        """medication > symptom > condition > allergy > lifestyle > general."""
        if "medications" in entities:
            return "medication"
        if "symptoms" in entities:
            return "symptom"
        if "conditions" in entities:
            return "condition"
        if "allergies" in entities:
            return "allergy"
        if "lifestyle" in entities:
            return "lifestyle"
        return "general"

    def extract_facts(self, text: str) -> List[Dict]:
        """Structured medication/symptom/condition facts (episodic shape)."""
        entities = self.extract_entities(text)
        specs = (("medications", "medication", 0.8),
                 ("symptoms", "symptom", 0.7),
                 ("conditions", "condition", 0.9))
        facts = []
        for category, ftype, conf in specs:
            if entities.get(category):
                facts.append({
                    "type": ftype,
                    "entities": entities[category],
                    "text": text[:200] + "..." if len(text) > 200 else text,
                    "confidence": conf,
                })
        return facts


# Compat alias: episodic.py exposed MedicalFactExtractor; same implementation.
MedicalFactExtractor = MedicalEntityExtractor
