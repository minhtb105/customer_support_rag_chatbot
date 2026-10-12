"""
Memory Management Module for RAG Chatbot

Provides comprehensive memory management including:
- Short-term Memory (Context Window)
- Episodic Memory (Conversation Summaries)
- Long-term Memory (Semantic Storage)
"""

from ._extract import MedicalEntityExtractor, MedicalFactExtractor

from .short_term import (
    ShortTermMemory,
    ContextCompressor,
    get_short_term_memory
)

from .episodic import (
    EpisodicMemory,
    ConversationSummary,
    ConversationSummarizer,
    get_episodic_memory
)
EpisodicMedicalFactExtractor = MedicalFactExtractor

from .long_term import (
    LongTermMemory,
    MedicalFact,
    TemporalWeighting,
    get_long_term_memory
)
LongTermMedicalEntityExtractor = MedicalEntityExtractor

__all__ = [
    'ShortTermMemory',
    'MedicalEntityExtractor',
    'ContextCompressor',
    'get_short_term_memory',
    'EpisodicMemory',
    'ConversationSummary',
    'MedicalFactExtractor',
    'ConversationSummarizer',
    'get_episodic_memory',
    'LongTermMemory',
    'MedicalFact',
    'LongTermMedicalEntityExtractor',
    'TemporalWeighting',
    'get_long_term_memory',
]
