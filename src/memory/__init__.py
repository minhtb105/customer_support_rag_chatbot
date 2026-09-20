"""
Memory Management Module for RAG Chatbot

Provides comprehensive memory management including:
- Short-term Memory (Context Window)
- Episodic Memory (Conversation Summaries)
- Long-term Memory (Semantic Storage)
"""

from .short_term import (
    ShortTermMemory,
    MedicalEntityExtractor,
    ContextCompressor,
    get_short_term_memory
)

from .episodic import (
    EpisodicMemory,
    ConversationSummary,
    MedicalFactExtractor as EpisodicMedicalFactExtractor,
    ConversationSummarizer,
    get_episodic_memory
)

from .long_term import (
    LongTermMemory,
    MedicalFact,
    MedicalEntityExtractor as LongTermMedicalEntityExtractor,
    TemporalWeighting,
    get_long_term_memory
)

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
