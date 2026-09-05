"""
Long-term Memory Module for RAG Chatbot

Implements semantic memory storage with medical entity extraction,
vector database integration, and temporal weighting for medical facts.
"""

import time
import json
import hashlib
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
try:
    from config import EMBEDDING_MODEL, PDF_DB_DIR
except ImportError:
    from src.config import EMBEDDING_MODEL, PDF_DB_DIR
try:
    from src.config import EMBEDDING_PROVIDER, OPENAI_EMBEDDING_MODEL
except ImportError:
    try:
        from config import EMBEDDING_PROVIDER, OPENAI_EMBEDDING_MODEL  # type: ignore
    except ImportError:
        EMBEDDING_PROVIDER = "local"
        OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass
class MedicalFact:
    """Represents a structured medical fact extracted from conversation."""
    fact_id: str
    user_id: str
    timestamp: float
    text: str
    entities: List[str]
    fact_type: str
    confidence: float
    source: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MedicalEntityExtractor:
    """Enhanced medical entity extractor for long-term memory."""
    
    def __init__(self):
        self.medical_patterns = {
            'medications': [
                r'\b(paracetamol|ibuprofen|aspirin|antibiotic|insulin|metformin)\b',
                r'\b(medication|drug|treatment|prescription|medicine)\b',
                r'\b(dose|dosage|pill|tablet|capsule|injection)\b'
            ],
            'symptoms': [
                r'\b(headache|fever|cough|pain|nausea|dizziness|fatigue)\b',
                r'\b(symptom|discomfort|ache|illness|condition)\b',
                r'\b(feeling|problem|issue|complaint)\b'
            ],
            'conditions': [
                r'\b(diabetes|hypertension|asthma|arthritis|depression|anxiety)\b',
                r'\b(condition|disease|disorder|illness|syndrome)\b',
                r'\b(diagnosis|medical issue|health problem)\b'
            ],
            'allergies': [
                r'\b(allergy|allergic|sensitivity|reaction)\b',
                r'\b(penicillin|peanut|shellfish|pollen)\b'
            ],
            'lifestyle': [
                r'\b(diet|exercise|sleep|stress|smoking|alcohol)\b',
                r'\b(lifestyle|habits|routine|wellness)\b'
            ]
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities from text."""
        import re
        
        entities = {}
        text_lower = text.lower()
        
        for category, patterns in self.medical_patterns.items():
            found = []
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                found.extend(matches)
            
            if found:
                entities[category] = list(set(found))  # Remove duplicates
        
        return entities
    
    def classify_fact_type(self, entities: Dict[str, List[str]]) -> str:
        """Classify the type of medical fact based on entities."""
        if 'medications' in entities:
            return 'medication'
        elif 'symptoms' in entities:
            return 'symptom'
        elif 'conditions' in entities:
            return 'condition'
        elif 'allergies' in entities:
            return 'allergy'
        elif 'lifestyle' in entities:
            return 'lifestyle'
        else:
            return 'general'


class TemporalWeighting:
    """Handles temporal weighting for medical facts."""
    
    def __init__(self, half_life_days: int = 90):  # 3 months half-life
        self.half_life_days = half_life_days
    
    def calculate_weight(self, timestamp: float) -> float:
        """Calculate temporal weight based on age of fact."""
        age_days = (time.time() - timestamp) / (24 * 3600)
        
        # Exponential decay
        decay_factor = 0.5 ** (age_days / self.half_life_days)
        
        # Minimum weight to prevent complete forgetting
        return max(decay_factor, 0.1)


class LongTermMemory:
    """Manages long-term semantic memory with vector storage."""
    
    def __init__(
        self,
        user_id: str = "default_user",
        embedding_model: str = EMBEDDING_MODEL,
        memory_db_dir: str = "embeddings/memory_db",
        half_life_days: int = 90
    ):
        self.user_id = user_id
        self.embedding_model = embedding_model
        self.memory_db_dir = memory_db_dir
        self.temporal_weighting = TemporalWeighting(half_life_days)
        
        # Initialize vector store — OpenAI nếu cấu hình, fallback HuggingFace
        if EMBEDDING_PROVIDER == "openai":
            try:
                from langchain_openai import OpenAIEmbeddings
                self.embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
            except Exception:
                self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = Chroma(
            persist_directory=memory_db_dir,
            embedding_function=self.embeddings
        )
        
        # Entity extractor
        self.entity_extractor = MedicalEntityExtractor()
        
        # Statistics
        self.total_facts = 0
        self.extracted_facts = 0
        self.retrieved_facts = 0
    
    def store_fact(self, text: str, source: str = "conversation", 
                   metadata: Dict[str, Any] = None) -> str:
        """Store a medical fact in long-term memory."""
        if not text.strip():
            return ""
        
        # Extract entities
        entities_dict = self.entity_extractor.extract_entities(text)
        fact_type = self.entity_extractor.classify_fact_type(entities_dict)
        
        # Flatten entities
        all_entities = []
        for entity_list in entities_dict.values():
            all_entities.extend(entity_list)
        
        # Create fact ID
        fact_id = hashlib.md5(f"{self.user_id}_{text}_{time.time()}".encode()).hexdigest()[:12]
        
        # Create fact
        fact = MedicalFact(
            fact_id=fact_id,
            user_id=self.user_id,
            timestamp=time.time(),
            text=text,
            entities=all_entities,
            fact_type=fact_type,
            confidence=0.8,  # Default confidence
            source=source,
            metadata=metadata or {}
        )
        
        # Store in vector database
        self._store_in_vector_db(fact)
        
        self.total_facts += 1
        self.extracted_facts += len(all_entities)
        
        return fact_id
    
    def _store_in_vector_db(self, fact: MedicalFact):
        """Store fact in vector database with metadata."""
        # Create embedding
        embedding = self.embeddings.embed_query(fact.text)
        
        # Prepare metadata
        metadata = {
            'fact_id': fact.fact_id,
            'user_id': fact.user_id,
            'fact_type': fact.fact_type,
            'entities': ','.join(fact.entities),
            'source': fact.source,
            'timestamp': fact.timestamp,
            'confidence': fact.confidence,
            'text_length': len(fact.text)
        }
        
        # Add to vector store
        self.vector_store.add_texts(
            texts=[fact.text],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[fact.fact_id]
        )
    
    def retrieve_facts(self, query: str, top_k: int = 5, 
                      fact_types: List[str] = None) -> List[MedicalFact]:
        """Retrieve relevant facts from long-term memory."""
        if not query.strip():
            return []
        
        # Build filter for fact types
        filter_dict = {"user_id": self.user_id}
        if fact_types:
            filter_dict["fact_type"] = {"$in": fact_types}
        
        # Retrieve from vector store
        docs = self.vector_store.similarity_search(
            query=query,
            k=top_k * 2,  # Get more results for filtering
            filter=filter_dict
        )
        
        # Convert to MedicalFact objects and apply temporal weighting
        facts = []
        for doc in docs:
            metadata = doc.metadata
            
            # Calculate temporal weight
            temporal_weight = self.temporal_weighting.calculate_weight(metadata['timestamp'])
            
            # Only include if temporal weight is above threshold
            if temporal_weight > 0.2:
                fact = MedicalFact(
                    fact_id=metadata['fact_id'],
                    user_id=metadata['user_id'],
                    timestamp=metadata['timestamp'],
                    text=doc.page_content,
                    entities=metadata['entities'].split(','),
                    fact_type=metadata['fact_type'],
                    confidence=metadata['confidence'],
                    source=metadata['source'],
                    metadata={'temporal_weight': temporal_weight}
                )
                facts.append(fact)
        
        # Sort by temporal weight and confidence
        facts.sort(key=lambda x: (
            x.metadata.get('temporal_weight', 0) * x.confidence
        ), reverse=True)
        
        self.retrieved_facts += len(facts)
        
        return facts[:top_k]
    
    def get_user_profile(self) -> Dict[str, Any]:
        """Generate a user medical profile from stored facts."""
        # Retrieve all facts for user
        all_docs = self.vector_store.get(
            filter={"user_id": self.user_id}
        )
        
        if not all_docs['ids']:
            return {"message": "No medical history found for this user."}
        
        # Group facts by type
        profile = {
            'medications': [],
            'symptoms': [],
            'conditions': [],
            'allergies': [],
            'lifestyle': [],
            'general': [],
            'last_updated': time.time()
        }
        
        for i, doc_id in enumerate(all_docs['ids']):
            doc = all_docs['documents'][i]
            metadata = all_docs['metadatas'][i]
            
            fact_type = metadata.get('fact_type', 'general')
            if fact_type in profile:
                profile[fact_type].append({
                    'text': doc,
                    'timestamp': metadata['timestamp'],
                    'confidence': metadata.get('confidence', 0.5),
                    'source': metadata.get('source', 'unknown')
                })
        
        # Sort each category by timestamp
        for category in ['medications', 'symptoms', 'conditions', 'allergies', 'lifestyle', 'general']:
            profile[category].sort(key=lambda x: x['timestamp'], reverse=True)
        
        return profile
    
    def update_fact_confidence(self, fact_id: str, new_confidence: float):
        """Update the confidence of a stored fact."""
        # This would require more complex vector store operations
        # For now, we'll note this as a future enhancement
        pass
    
    def forget_old_facts(self, max_age_days: int = 365):
        """Remove facts older than max_age_days."""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        
        # Get all facts for user
        all_docs = self.vector_store.get(
            filter={"user_id": self.user_id}
        )
        
        # Identify old facts
        old_ids = []
        for i, doc_id in enumerate(all_docs['ids']):
            metadata = all_docs['metadatas'][i]
            if metadata['timestamp'] < cutoff_time:
                old_ids.append(doc_id)
        
        # Remove old facts
        if old_ids:
            self.vector_store.delete(ids=old_ids)
    
    def clear_user_memory(self):
        """Clear all memory for the current user."""
        self.vector_store.delete(
            where={"user_id": self.user_id}
        )
        self.total_facts = 0
        self.extracted_facts = 0
        self.retrieved_facts = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about long-term memory."""
        all_docs = self.vector_store.get(
            where={"user_id": self.user_id}
        )
        
        fact_types = {}
        for metadata in all_docs['metadatas']:
            fact_type = metadata.get('fact_type', 'unknown')
            fact_types[fact_type] = fact_types.get(fact_type, 0) + 1
        
        return {
            'user_id': self.user_id,
            'total_facts': len(all_docs['ids']),
            'extracted_entities': self.extracted_facts,
            'retrieved_facts': self.retrieved_facts,
            'fact_types': fact_types,
            'database_size': len(all_docs['ids'])
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory state to dict."""
        return {
            'user_id': self.user_id,
            'stats': self.get_stats(),
            'temporal_weighting': {
                'half_life_days': self.temporal_weighting.half_life_days
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LongTermMemory':
        """Deserialize memory state from dict."""
        memory = cls(
            user_id=data['user_id'],
            half_life_days=data['temporal_weighting']['half_life_days']
        )
        return memory


# Global instance for easy access
long_term_memory = LongTermMemory()


def get_long_term_memory(user_id: str = "default_user") -> LongTermMemory:
    """Get the global long-term memory instance for a user."""
    global long_term_memory
    if long_term_memory.user_id != user_id:
        long_term_memory = LongTermMemory(user_id=user_id)
    return long_term_memory