"""
SQLAlchemy Models for Memory Storage

Implements unified memory storage with pgvector support for semantic search.
Supports both PostgreSQL (with pgvector) and SQLite (fallback).
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, Integer, String, JSON, DateTime, Text, Float, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from src.config import PDF_DB_DIR
from memory.database_config import Base, ENV, Environment


class MemoryFact(Base):
    """Unified memory fact table supporting both episodic summaries and long-term facts."""
    
    __tablename__ = "memory_facts"
    
    # Primary keys and identifiers
    id = Column(Integer, primary_key=True, autoincrement=True)
    fact_id = Column(String(64), unique=True, index=True)
    user_id = Column(String(100), index=True, nullable=False)
    session_id = Column(String(100), index=True)
    
    # Content and type
    fact_type = Column(String(50), nullable=False)  # 'summary', 'fact', 'entity'
    content = Column(Text, nullable=False)
    
    # Vector embedding (pgvector)
    if ENV != Environment.LOCAL:
        # PostgreSQL with pgvector
        embedding = Column(TSVECTOR(384))  # 384-dim vector for all-MiniLM-L6-v2
    else:
        # SQLite fallback - store as JSON string
        embedding = Column(String(2000))  # Store vector as JSON string
    
    # Metadata and tracking
    # NOTE: "metadata" is a reserved attribute name in SQLAlchemy Declarative,
    # so the mapped attribute is "fact_meta" while the DB column stays "metadata".
    fact_meta = Column("metadata", JSONB if ENV != Environment.LOCAL else JSON)
    confidence = Column(Float, default=0.8)
    source = Column(String(100), default="conversation")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # NOTE: no relationship() to UserProfile here on purpose: the two tables
    # are linked only by the string user_id (no FK), and SQLAlchemy 2.x fails
    # mapper configuration for relationships without a join condition.
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_memory_facts_user_type", "user_id", "fact_type"),
        Index("idx_memory_facts_session", "session_id"),
        Index("idx_memory_facts_created", "created_at"),
    )
    
    def __repr__(self):
        return f"<MemoryFact(id={self.fact_id}, user={self.user_id}, type={self.fact_type})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "fact_id": self.fact_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "fact_type": self.fact_type,
            "content": self.content,
            "metadata": self.fact_meta,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class UserProfile(Base):
    """User profile table for aggregated medical information."""
    
    __tablename__ = "user_profiles"
    
    user_id = Column(String(100), primary_key=True)
    profile_data = Column(JSONB if ENV != Environment.LOCAL else JSON, nullable=False)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<UserProfile(user_id={self.user_id}, last_updated={self.last_updated})>"
    
    def update_profile_from_facts(self, facts: List[MemoryFact]):
        """Update user profile from memory facts."""
        profile = {
            "medications": [],
            "symptoms": [],
            "conditions": [],
            "allergies": [],
            "lifestyle": [],
            "general": [],
            "last_updated": datetime.utcnow().isoformat()
        }
        
        for fact in facts:
            fact_type = fact.fact_type
            if fact_type in profile:
                profile[fact_type].append({
                    "text": fact.content,
                    "timestamp": fact.created_at.isoformat(),
                    "confidence": fact.confidence,
                    "source": fact.source
                })
        
        # Sort each category by timestamp
        for category in ["medications", "symptoms", "conditions", "allergies", "lifestyle", "general"]:
            profile[category].sort(key=lambda x: x["timestamp"], reverse=True)
        
        self.profile_data = profile
        self.last_updated = datetime.utcnow()


class SessionSummary(Base):
    """Session summary table for episodic memory (derived from MemoryFact)."""
    
    __tablename__ = "session_summaries"
    
    session_id = Column(String(100), primary_key=True)
    user_id = Column(String(100), index=True, nullable=False)
    summary_text = Column(Text, nullable=False)
    structured_facts = Column(JSONB if ENV != Environment.LOCAL else JSON)
    message_count = Column(Integer, default=0)
    topics = Column(String(500))  # Comma-separated topics
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<SessionSummary(session_id={self.session_id}, user={self.user_id})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "summary_text": self.summary_text,
            "structured_facts": self.structured_facts,
            "message_count": self.message_count,
            "topics": self.topics.split(",") if self.topics else [],
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }


class MemoryStats(Base):
    """Memory usage statistics for monitoring and optimization."""
    
    __tablename__ = "memory_stats"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), index=True, nullable=False)
    session_id = Column(String(100), index=True)
    stat_type = Column(String(50), nullable=False)  # 'short_term', 'episodic', 'long_term'
    metrics = Column(JSONB if ENV != Environment.LOCAL else JSON, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<MemoryStats(user={self.user_id}, type={self.stat_type}, time={self.timestamp})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "stat_type": self.stat_type,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


# Create tables if they don't exist
def create_tables(engine):
    """Create all database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ All memory tables created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create memory tables: {e}")
        return False


# Helper functions for common operations
def get_user_facts(user_id: str, fact_types: Optional[List[str]] = None, limit: int = 100) -> List[MemoryFact]:
    """Get all facts for a user, optionally filtered by type."""
    from memory.database_config import SessionLocal
    
    session = SessionLocal()
    try:
        query = session.query(MemoryFact).filter(MemoryFact.user_id == user_id)
        
        if fact_types:
            query = query.filter(MemoryFact.fact_type.in_(fact_types))
        
        return query.order_by(MemoryFact.created_at.desc()).limit(limit).all()
    
    finally:
        session.close()


def get_session_summary(session_id: str) -> Optional[SessionSummary]:
    """Get session summary by session ID."""
    from memory.database_config import SessionLocal
    
    session = SessionLocal()
    try:
        return session.query(SessionSummary).filter(SessionSummary.session_id == session_id).first()
    finally:
        session.close()


def update_user_profile(user_id: str):
    """Update user profile from existing facts."""
    from memory.database_config import SessionLocal
    
    session = SessionLocal()
    try:
        # Get all facts for user
        facts = session.query(MemoryFact).filter(MemoryFact.user_id == user_id).all()
        
        # Get or create user profile
        profile = session.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if not profile:
            profile = UserProfile(user_id=user_id, profile_data={})
            session.add(profile)
        
        # Update profile
        profile.update_profile_from_facts(facts)
        session.commit()
        
        return profile
    
    finally:
        session.close()


if __name__ == "__main__":
    print("🔧 Memory Models Test")
    print(f"Environment: {ENV.value}")
    
    if create_tables():
        print("🎉 Memory models setup complete!")
    else:
        print("💥 Memory models setup failed!")