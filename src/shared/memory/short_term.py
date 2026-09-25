"""
Short-term Memory Module for RAG Chatbot

Implements sliding window context management with medical entity retention
and context compression strategies.
"""

import time
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import deque
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
try:
    from shared.config import TOKENIZER_MODEL, MAX_TOKENS, SLIDING_WINDOW_TOKENS, SLIDING_OVERLAP
except ImportError:
    from src.shared.config import TOKENIZER_MODEL, MAX_TOKENS, SLIDING_WINDOW_TOKENS, SLIDING_OVERLAP


@dataclass
class Message:
    """Represents a single message in the conversation history."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MedicalEntityExtractor:
    """Extracts medical entities from text for importance scoring."""
    
    def __init__(self):
        # Common medical keywords and patterns
        self.medical_keywords = {
            'medications': [
                'paracetamol', 'ibuprofen', 'aspirin', 'antibiotic', 'insulin',
                'blood pressure', 'cholesterol', 'antidepressant', 'painkiller'
            ],
            'symptoms': [
                'headache', 'fever', 'cough', 'pain', 'nausea', 'dizziness',
                'fatigue', 'shortness of breath', 'chest pain', 'rash'
            ],
            'conditions': [
                'diabetes', 'hypertension', 'asthma', 'arthritis', 'depression',
                'anxiety', 'cancer', 'heart disease', 'stroke', 'infection'
            ],
            'medical_terms': [
                'dose', 'side effect', 'allergy', 'prescription', 'diagnosis',
                'treatment', 'symptom', 'condition', 'medication', 'drug'
            ]
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities from text."""
        text_lower = text.lower()
        entities = {}
        
        for category, keywords in self.medical_keywords.items():
            found = []
            for keyword in keywords:
                if keyword in text_lower:
                    found.append(keyword)
            if found:
                entities[category] = found
                
        return entities
    
    def calculate_importance_score(self, text: str) -> float:
        """Calculate importance score based on medical entity density."""
        entities = self.extract_entities(text)
        
        # Base score
        score = 0.5
        
        # Boost score based on entity types found
        if 'medications' in entities:
            score += 0.3 * len(entities['medications'])
        if 'symptoms' in entities:
            score += 0.2 * len(entities['symptoms'])
        if 'conditions' in entities:
            score += 0.4 * len(entities['conditions'])
        if 'medical_terms' in entities:
            score += 0.1 * len(entities['medical_terms'])
            
        return min(score, 1.0)  # Cap at 1.0


class ContextCompressor:
    """Compresses context using a small language model."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model = SentenceTransformer(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def compress_message(self, message: Message, target_tokens: int = 100) -> str:
        """Compress a message to target token count."""
        if not message.content:
            return ""
            
        # Simple compression: extract key sentences
        sentences = message.content.split('. ')
        if len(sentences) <= 2:
            return message.content
            
        # Use sentence embeddings to find most important sentences
        embeddings = self.model.encode(sentences)
        importance_scores = [abs(embedding).mean() for embedding in embeddings]
        
        # Sort sentences by importance
        sentence_scores = list(zip(sentences, importance_scores))
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Build compressed message
        compressed = []
        current_tokens = 0
        
        for sentence, _ in sentence_scores:
            sentence_tokens = len(self.tokenizer.tokenize(sentence))
            if current_tokens + sentence_tokens <= target_tokens:
                compressed.append(sentence)
                current_tokens += sentence_tokens
            else:
                break
                
        return '. '.join(compressed) + '.' if compressed else message.content


class ShortTermMemory:
    """Manages short-term memory with sliding window and medical entity retention."""
    
    def __init__(
        self,
        max_tokens: int = MAX_TOKENS,
        window_size: int = SLIDING_WINDOW_TOKENS,
        overlap: int = SLIDING_OVERLAP,
        compression_target: int = 100
    ):
        self.max_tokens = max_tokens
        self.window_size = window_size
        self.overlap = overlap
        self.compression_target = compression_target
        
        self.messages = deque()
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
        self.entity_extractor = MedicalEntityExtractor()
        self.compressor = ContextCompressor()
        
        # Statistics
        self.total_messages = 0
        self.compressed_messages = 0
        self.retained_important = 0
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add a new message to the conversation history."""
        if not content.strip():
            return
            
        # Calculate importance score
        importance_score = self.entity_extractor.calculate_importance_score(content)
        
        message = Message(
            role=role,
            content=content,
            timestamp=time.time(),
            metadata={
                'importance_score': importance_score,
                'original_tokens': len(self.tokenizer.tokenize(content)),
                **(metadata or {})
            }
        )
        
        self.messages.append(message)
        self.total_messages += 1
        
        # Manage memory size
        self._manage_memory_size()
    
    def _manage_memory_size(self):
        """Manage memory size using sliding window with medical entity retention."""
        current_tokens = self._count_tokens()
        
        if current_tokens <= self.max_tokens:
            return
        
        # Sort messages by importance (keep important medical messages)
        messages_list = list(self.messages)
        messages_list.sort(key=lambda m: (
            m.metadata.get('importance_score', 0.5),  # Primary: importance
            m.timestamp  # Secondary: recency
        ), reverse=True)
        
        # Keep important messages and recent messages
        kept_messages = []
        kept_tokens = 0
        
        for message in messages_list:
            message_tokens = len(self.tokenizer.tokenize(message.content))
            
            # Always keep very important medical messages
            if message.metadata.get('importance_score', 0) >= 0.8:
                kept_messages.append(message)
                kept_tokens += message_tokens
                self.retained_important += 1
            # Keep recent messages if we have space
            elif kept_tokens + message_tokens <= self.max_tokens:
                kept_messages.append(message)
                kept_tokens += message_tokens
            # Compress if still too large
            elif kept_tokens < self.max_tokens:
                compressed_content = self.compressor.compress_message(
                    message, 
                    self.compression_target
                )
                compressed_tokens = len(self.tokenizer.tokenize(compressed_content))
                
                if kept_tokens + compressed_tokens <= self.max_tokens:
                    compressed_message = Message(
                        role=message.role,
                        content=compressed_content,
                        timestamp=message.timestamp,
                        metadata={
                            **message.metadata,
                            'compressed': True,
                            'compression_ratio': compressed_tokens / message.metadata['original_tokens']
                        }
                    )
                    kept_messages.append(compressed_message)
                    kept_tokens += compressed_tokens
                    self.compressed_messages += 1
        
        # Re-sort by timestamp and update deque
        kept_messages.sort(key=lambda m: m.timestamp)
        self.messages = deque(kept_messages)
    
    def _count_tokens(self) -> int:
        """Count total tokens in current messages."""
        return sum(
            len(self.tokenizer.tokenize(m.content))
            for m in self.messages
        )
    
    def get_context_window(self, max_tokens: int = None) -> List[Dict[str, Any]]:
        """Get the current context window as a list of message dicts."""
        if max_tokens is None:
            max_tokens = self.max_tokens
            
        context = []
        current_tokens = 0
        
        # Add messages from most recent to oldest
        for message in reversed(self.messages):
            message_tokens = len(self.tokenizer.tokenize(message.content))
            
            if current_tokens + message_tokens > max_tokens:
                break
                
            context.insert(0, {
                'role': message.role,
                'content': message.content,
                'timestamp': message.timestamp,
                'metadata': message.metadata
            })
            current_tokens += message_tokens
        
        return context
    
    def get_context_string(self, max_tokens: int = None) -> str:
        """Get context as a formatted string for LLM input."""
        context = self.get_context_window(max_tokens)
        
        formatted_context = []
        for msg in context:
            role = "User" if msg['role'] == 'user' else "Assistant"
            formatted_context.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted_context)
    
    def clear(self):
        """Clear all messages from shared.memory."""
        self.messages.clear()
        self.total_messages = 0
        self.compressed_messages = 0
        self.retained_important = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage."""
        return {
            'total_messages': self.total_messages,
            'current_messages': len(self.messages),
            'total_tokens': self._count_tokens(),
            'max_tokens': self.max_tokens,
            'compressed_messages': self.compressed_messages,
            'retained_important': self.retained_important,
            'compression_ratio': (
                self.compressed_messages / max(self.total_messages, 1)
            )
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory state to dict."""
        return {
            'messages': [
                {
                    'role': m.role,
                    'content': m.content,
                    'timestamp': m.timestamp,
                    'metadata': m.metadata
                }
                for m in self.messages
            ],
            'stats': self.get_stats()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ShortTermMemory':
        """Deserialize memory state from dict."""
        memory = cls()
        for msg_data in data['messages']:
            message = Message(
                role=msg_data['role'],
                content=msg_data['content'],
                timestamp=msg_data['timestamp'],
                metadata=msg_data['metadata']
            )
            memory.messages.append(message)
        
        return memory


# Global instance for easy access
short_term_memory = ShortTermMemory()


def get_short_term_memory() -> ShortTermMemory:
    """Get the global short-term memory instance."""
    return short_term_memory