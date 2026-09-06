"""Features package — Diabetes (glucose) + Hypertension (BP) + Respiratory + Mental (PHQ-9/GAD-7) + SOAP."""

from .glucose_tracker import (
    init_glucose_db, classify_glucose, add_log, get_logs, get_stats, should_escalate_to_doctor,
)
from .bp_tracker import (
    init_bp_db, classify_bp, add_bp_log, get_bp_logs, get_bp_stats, should_escalate_bp,
)
from .respiratory_tracker import (
    init_respiratory_db, classify_peak_flow, add_respiratory_log, get_respiratory_logs, get_respiratory_stats, should_escalate_respiratory,
)
from .mood_tracker import (
    init_mood_db, redact_pii, classify_phq9, classify_gad7, contains_crisis_keywords,
    add_mood_log, get_mood_logs, get_mood_stats, should_escalate_mood,
)

__all__ = [
    "init_glucose_db", "classify_glucose", "add_log", "get_logs", "get_stats", "should_escalate_to_doctor",
    "init_bp_db", "classify_bp", "add_bp_log", "get_bp_logs", "get_bp_stats", "should_escalate_bp",
    "init_respiratory_db", "classify_peak_flow", "add_respiratory_log", "get_respiratory_logs", "get_respiratory_stats", "should_escalate_respiratory",
    "init_mood_db", "redact_pii", "classify_phq9", "classify_gad7", "contains_crisis_keywords", "add_mood_log", "get_mood_logs", "get_mood_stats", "should_escalate_mood",
]
