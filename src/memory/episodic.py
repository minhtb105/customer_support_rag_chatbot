"""
Episodic Memory Module for RAG Chatbot

Implements conversation summarization with medical entity extraction
and structured fact storage for long-term retrieval.
"""

import time
import json
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
try:
    from config import TOKENIZER_MODEL, MAX_TOKENS
except ImportError:
    from src.config import TOKENIZER_MODEL, MAX_TOKENS


@dataclass
class ConversationSummary:
    """Represents a structured summary of a conversation segment."""
    summary_id: str
    timestamp: float
    summary_text: str
    structured_facts: List[Dict[str, Any]]
    conversation_window: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MedicalFactExtractor:
    """Extracts structured medical facts from conversation text."""
    
    def __init__(self):
        # Medical entity patterns
        self.medical_patterns = {
            'medications': [
                r'\b(paracetamol|ibuprofen|aspirin|antibiotic|insulin)\b',
                r'\b(medication|drug|treatment|prescription)\b',
                r'\b(dose|dosage|pill|tablet|capsule)\b'
            ],
            'symptoms': [
                r'\b(headache|fever|cough|pain|nausea|dizziness)\b',
                r'\b(symptom|discomfort|ache|illness)\b',
                r'\b(feeling|condition|problem)\b'
            ],
            'conditions': [
                r'\b(diabetes|hypertension|asthma|arthritis|depression)\b',
                r'\b(condition|disease|disorder|illness)\b',
                r'\b(diagnosis|medical issue)\b'
            ],
            'temporal': [
                r'\b(yesterday|today|tomorrow|last week|recently)\b',
                r'\b(when|since|for|ago)\b',
                r'\b(always|sometimes|never|usually)\b'
            ]
        }
    
    def extract_facts(self, text: str) -> List[Dict[str, Any]]:
        """Extract structured facts from conversation text."""
        import re
        
        facts = []
        
        # Extract medication facts
        for pattern in self.medical_patterns['medications']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                facts.append({
                    'type': 'medication',
                    'entities': matches,
                    'text': text[:200] + '...' if len(text) > 200 else text,
                    'confidence': 0.8
                })
        
        # Extract symptom facts
        for pattern in self.medical_patterns['symptoms']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                facts.append({
                    'type': 'symptom',
                    'entities': matches,
                    'text': text[:200] + '...' if len(text) > 200 else text,
                    'confidence': 0.7
                })
        
        # Extract condition facts
        for pattern in self.medical_patterns['conditions']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                facts.append({
                    'type': 'condition',
                    'entities': matches,
                    'text': text[:200] + '...' if len(text) > 200 else text,
                    'confidence': 0.9
                })
        
        return facts


class ConversationSummarizer:
    """Summarizes conversation segments with medical focus."""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        try:
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                device=0  # Use GPU if available
            )
        except:
            # Fallback to simple extractive summarization
            self.summarizer = None
            self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
        self.fact_extractor = MedicalFactExtractor()
    
    def summarize_conversation(self, messages: List[Dict[str, Any]], 
                             max_length: int = 150, min_length: int = 50) -> str:
        """Generate a summary of the conversation segment."""
        if not messages:
            return "No conversation to summarize."
        
        # Format conversation text
        conversation_text = ""
        for msg in messages:
            role = "User" if msg['role'] == 'user' else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n"
        
        # Use model-based summarization if available
        if self.summarizer:
            try:
                summary = self.summarizer(
                    conversation_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )[0]['summary_text']
                return summary
            except:
                pass
        
        # Fallback to extractive summarization
        sentences = conversation_text.split('. ')
        if len(sentences) <= 3:
            return conversation_text
        
        # Use sentence embeddings to find most important sentences
        embeddings = self.sentence_model.encode(sentences)
        importance_scores = [abs(embedding).mean() for embedding in embeddings]
        
        # Sort sentences by importance
        sentence_scores = list(zip(sentences, importance_scores))
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Build summary
        summary_sentences = []
        current_length = 0
        
        for sentence, _ in sentence_scores:
            sentence_length = len(sentence.split())
            if current_length + sentence_length <= max_length:
                summary_sentences.append(sentence)
                current_length += sentence_length
            else:
                break
        
        return '. '.join(summary_sentences) + '.' if summary_sentences else conversation_text


class EpisodicMemory:
    """Manages episodic memory with dynamic summarization and structured facts."""
    
    def __init__(
        self,
        summary_threshold: int = 10,  # Summarize after 10 messages
        max_summaries: int = 50,      # Keep last 50 summaries
        summary_window: int = 5       # Summarize last 5 messages
    ):
        self.summary_threshold = summary_threshold
        self.max_summaries = max_summaries
        self.summary_window = summary_window
        
        self.summaries: List[ConversationSummary] = []
        self.conversation_buffer: List[Dict[str, Any]] = []
        self.message_count = 0
        
        self.summarizer = ConversationSummarizer()
        self.fact_extractor = MedicalFactExtractor()
        
        # Statistics
        self.total_summaries = 0
        self.extracted_facts = 0
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add a message to the conversation buffer."""
        if not content.strip():
            return
        
        message = {
            'role': role,
            'content': content,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        self.conversation_buffer.append(message)
        self.message_count += 1
        
        # Check if we need to create a summary
        if len(self.conversation_buffer) >= self.summary_threshold:
            self._create_summary()
    
    def _create_summary(self):
        """Create a summary of the current conversation buffer."""
        if not self.conversation_buffer:
            return
        
        # Get the last N messages for summarization
        window_messages = self.conversation_buffer[-self.summary_window:]
        
        # Generate summary
        summary_text = self.summarizer.summarize_conversation(
            window_messages,
            max_length=200,
            min_length=80
        )
        
        # Extract structured facts
        conversation_text = " ".join([msg['content'] for msg in window_messages])
        structured_facts = self.fact_extractor.extract_facts(conversation_text)
        
        # Create summary ID
        summary_id = f"summary_{int(time.time())}_{len(self.summaries)}"
        
        # Create metadata
        metadata = {
            'message_count': len(window_messages),
            'fact_count': len(structured_facts),
            'avg_importance': self._calculate_avg_importance(window_messages),
            'topics': self._extract_topics(window_messages)
        }
        
        # Create summary object
        summary = ConversationSummary(
            summary_id=summary_id,
            timestamp=time.time(),
            summary_text=summary_text,
            structured_facts=structured_facts,
            conversation_window=window_messages.copy(),
            metadata=metadata
        )
        
        # Add to summaries
        self.summaries.append(summary)
        self.total_summaries += 1
        self.extracted_facts += len(structured_facts)
        
        # Maintain size limit
        if len(self.summaries) > self.max_summaries:
            self.summaries.pop(0)
        
        # Clear processed messages (keep last few for context)
        self.conversation_buffer = self.conversation_buffer[-3:]
    
    def _calculate_avg_importance(self, messages: List[Dict[str, Any]]) -> float:
        """Calculate average importance of messages."""
        if not messages:
            return 0.0
        
        total_importance = 0.0
        for msg in messages:
            # Simple importance heuristic based on message length and medical terms
            content = msg['content'].lower()
            medical_terms = sum(1 for term in ['medication', 'symptom', 'condition', 'treatment'] 
                              if term in content)
            importance = 0.5 + (len(content.split()) / 100) + (medical_terms * 0.1)
            total_importance += min(importance, 1.0)
        
        return total_importance / len(messages)
    
    def _extract_topics(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Extract main topics from messages."""
        topics = set()
        all_text = " ".join([msg['content'].lower() for msg in messages])
        
        medical_topics = [
            'medication', 'symptom', 'condition', 'treatment', 'diagnosis',
            'prevention', 'lifestyle', 'nutrition', 'exercise', 'mental health'
        ]
        
        for topic in medical_topics:
            if topic in all_text:
                topics.add(topic)
        
        return list(topics)
    
    def get_relevant_summaries(self, query: str, top_k: int = 3) -> List[ConversationSummary]:
        """Get relevant summaries for a query."""
        if not self.summaries:
            return []
        
        # Simple keyword matching for now
        query_lower = query.lower()
        relevant = []
        
        for summary in self.summaries:
            # Check if summary contains query keywords
            summary_text = summary.summary_text.lower()
            query_words = set(query_lower.split())
            summary_words = set(summary_text.split())
            
            overlap = len(query_words & summary_words)
            if overlap > 0:
                relevant.append((overlap, summary))
        
        # Sort by relevance and return top k
        relevant.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in relevant[:top_k]]
    
    def get_context_from_summaries(self, query: str, max_tokens: int = 500) -> str:
        """Get context from relevant summaries."""
        relevant_summaries = self.get_relevant_summaries(query)
        
        if not relevant_summaries:
            return ""
        
        context_parts = []
        current_tokens = 0
        
        for summary in relevant_summaries:
            summary_text = f"Summary: {summary.summary_text}"
            summary_tokens = len(self.summarizer.tokenizer.tokenize(summary_text))
            
            if current_tokens + summary_tokens > max_tokens:
                break
            
            context_parts.append(summary_text)
            current_tokens += summary_tokens
        
        return "\n\n".join(context_parts)
    
    def get_all_facts(self) -> List[Dict[str, Any]]:
        """Get all extracted facts from all summaries."""
        all_facts = []
        for summary in self.summaries:
            all_facts.extend(summary.structured_facts)
        return all_facts
    
    def clear(self):
        """Clear all summaries and buffer."""
        self.summaries.clear()
        self.conversation_buffer.clear()
        self.message_count = 0
        self.total_summaries = 0
        self.extracted_facts = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about episodic memory."""
        return {
            'total_summaries': self.total_summaries,
            'current_summaries': len(self.summaries),
            'max_summaries': self.max_summaries,
            'extracted_facts': self.extracted_facts,
            'message_count': self.message_count,
            'buffer_size': len(self.conversation_buffer)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory state to dict."""
        return {
            'summaries': [s.to_dict() for s in self.summaries],
            'conversation_buffer': self.conversation_buffer,
            'message_count': self.message_count,
            'stats': self.get_stats()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EpisodicMemory':
        """Deserialize memory state from dict."""
        memory = cls()
        memory.summaries = [ConversationSummary(**s) for s in data['summaries']]
        memory.conversation_buffer = data['conversation_buffer']
        memory.message_count = data['message_count']
        
        return memory


# Global instance for easy access
episodic_memory = EpisodicMemory()


def get_episodic_memory() -> EpisodicMemory:
    """Get the global episodic memory instance."""
    return episodic_memory