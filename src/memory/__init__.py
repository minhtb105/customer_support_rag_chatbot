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

from .evaluation import (
    MemoryEvaluator,
    EvaluationResult,
    get_memory_evaluator
)

from .database_config import (
    get_database_url, get_database_config, create_database_engine,
    SessionLocal, Base, init_database, test_connection
)

from .models import (
    MemoryFact, UserProfile, SessionSummary, MemoryStats,
    get_user_facts, get_session_summary, update_user_profile
)

from .adapters import (
    RedisShortTermAdapter, SQLAlchemyEpisodicAdapter, SQLAlchemyLongTermAdapter,
    MemoryStatsAdapter, get_redis_adapter, get_episodic_adapter, 
    get_long_term_adapter, get_stats_adapter
)

from .monitoring import (
    MemoryMonitor, MemoryBackup, MemoryMetrics,
    get_memory_monitor, get_memory_backup
)

__all__ = [
    # Original memory classes
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
    'MemoryEvaluator',
    'EvaluationResult',
    'get_memory_evaluator',
    
    # Database configuration
    'get_database_url', 'get_database_config', 'create_database_engine',
    'SessionLocal', 'Base', 'init_database', 'test_connection',
    
    # Database models
    'MemoryFact', 'UserProfile', 'SessionSummary', 'MemoryStats',
    'get_user_facts', 'get_session_summary', 'update_user_profile',
    
    # Adapters
    'RedisShortTermAdapter', 'SQLAlchemyEpisodicAdapter', 'SQLAlchemyLongTermAdapter',
    'MemoryStatsAdapter', 'get_redis_adapter', 'get_episodic_adapter', 
    'get_long_term_adapter', 'get_stats_adapter',
    
    # Monitoring
    'MemoryMonitor', 'MemoryBackup', 'MemoryMetrics',
    'get_memory_monitor', 'get_memory_backup'
]
