"""
Memory Adapters for Persistent Storage

Implements Redis adapter for short-term memory and SQLAlchemy adapters
for PostgreSQL/SQLite with pgvector support for semantic search.
"""

import json
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import asdict
from redis import Redis
from sentence_transformers import SentenceTransformer
from memory.database_config import SessionLocal, engine, Base
from memory.models import MemoryFact, SessionSummary, UserProfile, MemoryStats
from memory.short_term import ShortTermMemory, Message
from memory.episodic import EpisodicMemory, ConversationSummary
from memory.long_term import LongTermMemory, MedicalFact
from config import TOKENIZER_MODEL


class RedisShortTermAdapter:
    """Redis adapter for short-term memory with TTL and LTRIM."""
    
    def __init__(self, redis_client: Redis, session_id: str, ttl: int = 7200, max_messages: int = 15):
        self.redis = redis_client
        self.session_id = session_id
        self.ttl = ttl
        self.max_messages = max_messages
        self.key = f"session:{session_id}:messages"
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add message to Redis with serialization."""
        if not content.strip():
            return
        
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        # Serialize and store
        message_json = json.dumps(message)
        self.redis.lpush(self.key, message_json)
        self.redis.ltrim(self.key, 0, self.max_messages - 1)  # Keep last N messages
        self.redis.expire(self.key, self.ttl)
    
    def get_context(self, max_tokens: int = 500) -> str:
        """Get context from Redis messages."""
        messages = self.redis.lrange(self.key, 0, -1)
        if not messages:
            return ""
        
        # Deserialize messages
        message_list = []
        current_tokens = 0
        
        for msg_json in reversed(messages):  # Most recent last
            msg = json.loads(msg_json)
            message_list.append(msg)
            
            # Simple token estimation
            content_tokens = len(msg['content'].split())
            current_tokens += content_tokens
            
            if current_tokens > max_tokens:
                break
        
        # Format context
        context_parts = []
        for msg in message_list:
            role = "User" if msg['role'] == 'user' else "Assistant"
            context_parts.append(f"{role}: {msg['content']}")
        
        return "\n".join(context_parts)
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages as list."""
        messages = self.redis.lrange(self.key, 0, -1)
        return [json.loads(msg) for msg in messages]
    
    def clear(self):
        """Clear all messages for session."""
        self.redis.delete(self.key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis memory statistics."""
        message_count = self.redis.llen(self.key)
        memory_usage = self.redis.memory_usage(self.key) if message_count > 0 else 0
        
        return {
            "session_id": self.session_id,
            "message_count": message_count,
            "memory_usage_bytes": memory_usage,
            "ttl_seconds": self.redis.ttl(self.key),
            "max_messages": self.max_messages
        }


class SQLAlchemyEpisodicAdapter:
    """SQLAlchemy adapter for episodic memory with pgvector support."""
    
    def __init__(self, user_id: str, session_id: str):
        self.user_id = user_id
        self.session_id = session_id
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def store_summary(self, summary: ConversationSummary):
        """Store conversation summary in database."""
        session = SessionLocal()
        try:
            # Create or update session summary
            session_summary = session.query(SessionSummary).filter(
                SessionSummary.session_id == self.session_id
            ).first()
            
            if not session_summary:
                session_summary = SessionSummary(
                    session_id=self.session_id,
                    user_id=self.user_id
                )
                session.add(session_summary)
            
            # Update summary data
            session_summary.summary_text = summary.summary_text
            session_summary.structured_facts = summary.structured_facts
            session_summary.message_count = len(summary.conversation_window)
            session_summary.topics = ",".join(summary.metadata.get('topics', []))
            
            # Store individual facts in memory_facts table
            for fact in summary.structured_facts:
                fact_id = hashlib.md5(f"{self.user_id}_{fact['text']}_{time.time()}".encode()).hexdigest()[:12]
                
                # Create embedding if using PostgreSQL
                embedding = None
                if engine.url.drivername == 'postgresql':
                    embedding = self.sentence_model.encode([fact['text']])[0].tolist()
                
                memory_fact = MemoryFact(
                    fact_id=fact_id,
                    user_id=self.user_id,
                    session_id=self.session_id,
                    fact_type=fact['type'],
                    content=fact['text'],
                    embedding=json.dumps(embedding) if embedding else None,
                    metadata=fact,
                    confidence=fact.get('confidence', 0.8),
                    source="episodic_summary"
                )
                
                session.add(memory_fact)
            
            session.commit()
            return True
        
        except Exception as e:
            session.rollback()
            print(f"❌ Failed to store summary: {e}")
            return False
        
        finally:
            session.close()
    
    def get_relevant_summaries(self, query: str, top_k: int = 3) -> List[ConversationSummary]:
        """Get relevant summaries using semantic search."""
        session = SessionLocal()
        try:
            # Get query embedding
            query_embedding = self.sentence_model.encode([query])[0].tolist()
            
            # Semantic search if using PostgreSQL
            if engine.url.drivername == 'postgresql':
                results = session.execute("""
                    SELECT session_id, summary_text, structured_facts, message_count, topics,
                           embedding <=> %s::vector as distance
                    FROM session_summaries 
                    WHERE user_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, self.user_id, query_embedding, top_k))
                
                summaries = []
                for row in results:
                    summary = ConversationSummary(
                        summary_id=row.session_id,
                        timestamp=time.time(),
                        summary_text=row.summary_text,
                        structured_facts=row.structured_facts,
                        conversation_window=[],  # Not stored in summary table
                        metadata={
                            'message_count': row.message_count,
                            'topics': row.topics.split(",") if row.topics else []
                        }
                    )
                    summaries.append(summary)
                
                return summaries
            
            else:
                # Fallback to keyword matching for SQLite
                results = session.query(SessionSummary).filter(
                    SessionSummary.user_id == self.user_id
                ).limit(top_k).all()
                
                return [
                    ConversationSummary(
                        summary_id=r.session_id,
                        timestamp=time.time(),
                        summary_text=r.summary_text,
                        structured_facts=r.structured_facts,
                        conversation_window=[],
                        metadata={
                            'message_count': r.message_count,
                            'topics': r.topics.split(",") if r.topics else []
                        }
                    )
                    for r in results
                ]
        
        except Exception as e:
            print(f"❌ Failed to retrieve summaries: {e}")
            return []
        
        finally:
            session.close()
    
    def get_context_from_summaries(self, query: str, max_tokens: int = 500) -> str:
        """Get context from relevant summaries."""
        summaries = self.get_relevant_summaries(query)
        
        if not summaries:
            return ""
        
        context_parts = []
        current_tokens = 0
        
        for summary in summaries:
            summary_text = f"Summary: {summary.summary_text}"
            summary_tokens = len(summary_text.split())
            
            if current_tokens + summary_tokens > max_tokens:
                break
            
            context_parts.append(summary_text)
            current_tokens += summary_tokens
        
        return "\n\n".join(context_parts)


class SQLAlchemyLongTermAdapter:
    """SQLAlchemy adapter for long-term memory with pgvector semantic search."""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def store_fact(self, fact: MedicalFact) -> bool:
        """Store medical fact in long-term memory."""
        session = SessionLocal()
        try:
            # Create embedding
            embedding = self.sentence_model.encode([fact.text])[0].tolist()
            
            # Create memory fact
            memory_fact = MemoryFact(
                fact_id=fact.fact_id,
                user_id=self.user_id,
                session_id=fact.metadata.get('session_id', ''),
                fact_type=fact.fact_type,
                content=fact.text,
                embedding=json.dumps(embedding),
                metadata={
                    'entities': fact.entities,
                    'source': fact.source,
                    'session_id': fact.metadata.get('session_id', ''),
                    'interaction_type': fact.metadata.get('interaction_type', 'Q&A')
                },
                confidence=fact.confidence,
                source=fact.source
            )
            
            session.add(memory_fact)
            session.commit()
            
            # Update user profile
            self._update_user_profile()
            
            return True
        
        except Exception as e:
            session.rollback()
            print(f"❌ Failed to store fact: {e}")
            return False
        
        finally:
            session.close()
    
    def retrieve_facts(self, query: str, top_k: int = 5, fact_types: List[str] = None) -> List[MedicalFact]:
        """Retrieve relevant facts using semantic search."""
        session = SessionLocal()
        try:
            # Get query embedding
            query_embedding = self.sentence_model.encode([query])[0].tolist()
            
            # Build query
            query_obj = session.query(MemoryFact).filter(
                MemoryFact.user_id == self.user_id
            )
            
            if fact_types:
                query_obj = query_obj.filter(MemoryFact.fact_type.in_(fact_types))
            
            # Semantic search if using PostgreSQL
            if engine.url.drivername == 'postgresql':
                results = query_obj.order_by(
                    MemoryFact.embedding.cosine_distance(query_embedding)
                ).limit(top_k * 2).all()
            else:
                # Fallback to created_at for SQLite
                results = query_obj.order_by(
                    MemoryFact.created_at.desc()
                ).limit(top_k * 2).all()
            
            # Convert to MedicalFact objects
            facts = []
            for result in results:
                # Parse embedding
                embedding_data = json.loads(result.embedding) if result.embedding else []
                
                # Parse metadata
                metadata = result.metadata or {}
                
                fact = MedicalFact(
                    fact_id=result.fact_id,
                    user_id=result.user_id,
                    timestamp=result.created_at.timestamp(),
                    text=result.content,
                    entities=metadata.get('entities', []),
                    fact_type=result.fact_type,
                    confidence=result.confidence,
                    source=result.source,
                    metadata={
                        'temporal_weight': 1.0,  # Simplified for now
                        'embedding_size': len(embedding_data)
                    }
                )
                facts.append(fact)
            
            # Sort by confidence and return top k
            facts.sort(key=lambda x: x.confidence, reverse=True)
            return facts[:top_k]
        
        except Exception as e:
            print(f"❌ Failed to retrieve facts: {e}")
            return []
        
        finally:
            session.close()
    
    def get_user_profile(self) -> Dict[str, Any]:
        """Get user medical profile."""
        session = SessionLocal()
        try:
            profile = session.query(UserProfile).filter(
                UserProfile.user_id == self.user_id
            ).first()
            
            if profile:
                return profile.profile_data
            else:
                return {"message": "No medical history found for this user."}
        
        except Exception as e:
            print(f"❌ Failed to get user profile: {e}")
            return {"message": "Error retrieving user profile."}
        
        finally:
            session.close()
    
    def _update_user_profile(self):
        """Update user profile from stored facts."""
        session = SessionLocal()
        try:
            # Get all facts for user
            facts = session.query(MemoryFact).filter(
                MemoryFact.user_id == self.user_id
            ).all()
            
            # Get or create user profile
            profile = session.query(UserProfile).filter(
                UserProfile.user_id == self.user_id
            ).first()
            
            if not profile:
                profile = UserProfile(user_id=self.user_id, profile_data={})
                session.add(profile)
            
            # Update profile
            profile_data = {
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
                if fact_type in profile_data:
                    profile_data[fact_type].append({
                        "text": fact.content,
                        "timestamp": fact.created_at.isoformat(),
                        "confidence": fact.confidence,
                        "source": fact.source
                    })
            
            # Sort each category by timestamp
            for category in ["medications", "symptoms", "conditions", "allergies", "lifestyle", "general"]:
                profile_data[category].sort(key=lambda x: x["timestamp"], reverse=True)
            
            profile.profile_data = profile_data
            session.commit()
        
        except Exception as e:
            session.rollback()
            print(f"❌ Failed to update user profile: {e}")
        
        finally:
            session.close()


class MemoryStatsAdapter:
    """Adapter for storing and retrieving memory statistics."""
    
    def __init__(self):
        pass
    
    def store_stats(self, user_id: str, session_id: str, stat_type: str, metrics: Dict[str, Any]):
        """Store memory statistics."""
        session = SessionLocal()
        try:
            stats = MemoryStats(
                user_id=user_id,
                session_id=session_id,
                stat_type=stat_type,
                metrics=metrics
            )
            
            session.add(stats)
            session.commit()
            return True
        
        except Exception as e:
            session.rollback()
            print(f"❌ Failed to store stats: {e}")
            return False
        
        finally:
            session.close()
    
    def get_stats(self, user_id: str, stat_type: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get memory statistics."""
        session = SessionLocal()
        try:
            query = session.query(MemoryStats).filter(MemoryStats.user_id == user_id)
            
            if stat_type:
                query = query.filter(MemoryStats.stat_type == stat_type)
            
            results = query.order_by(MemoryStats.timestamp.desc()).limit(limit).all()
            
            return [result.to_dict() for result in results]
        
        except Exception as e:
            print(f"❌ Failed to get stats: {e}")
            return []
        
        finally:
            session.close()


# Global adapter instances
def get_redis_adapter(session_id: str, redis_client: Redis = None) -> RedisShortTermAdapter:
    """Get Redis adapter for short-term memory."""
    if redis_client is None:
        # Create default Redis client
        import redis
        redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    return RedisShortTermAdapter(redis_client, session_id)


def get_episodic_adapter(user_id: str, session_id: str) -> SQLAlchemyEpisodicAdapter:
    """Get SQLAlchemy adapter for episodic memory."""
    return SQLAlchemyEpisodicAdapter(user_id, session_id)


def get_long_term_adapter(user_id: str) -> SQLAlchemyLongTermAdapter:
    """Get SQLAlchemy adapter for long-term memory."""
    return SQLAlchemyLongTermAdapter(user_id)


def get_stats_adapter() -> MemoryStatsAdapter:
    """Get adapter for memory statistics."""
    return MemoryStatsAdapter()


if __name__ == "__main__":
    print("🔧 Memory Adapters Test")
    
    # Test database connection
    from memory.database_config import test_connection
    if test_connection():
        print("✅ Database adapters ready!")
    else:
        print("❌ Database adapters failed!")