"""
Memory Evaluation Module for RAG Chatbot

Provides comprehensive evaluation metrics for all memory layers:
- Short-term Memory: Hit rate, compression efficiency, token usage
- Episodic Memory: Summary quality, fact extraction accuracy
- Long-term Memory: Retrieval precision, temporal decay effectiveness
"""

import time
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from memory.short_term import ShortTermMemory
from memory.episodic import EpisodicMemory
from memory.long_term import LongTermMemory
from memory.long_term import MedicalFact


@dataclass
class EvaluationResult:
    """Represents evaluation results for a memory layer."""
    memory_type: str
    metrics: Dict[str, float]
    details: Dict[str, Any]
    timestamp: float


class MemoryEvaluator:
    """Comprehensive evaluator for all memory layers."""
    
    def __init__(self):
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.evaluation_history: List[EvaluationResult] = []
    
    def evaluate_short_term_memory(self, memory: ShortTermMemory) -> EvaluationResult:
        """Evaluate short-term memory performance."""
        stats = memory.get_stats()
        
        # Calculate metrics
        metrics = {}
        
        # Hit rate (if we had a way to track cache hits specifically)
        metrics['hit_rate'] = 0.0  # Placeholder - would need more tracking
        
        # Compression efficiency
        if stats['total_messages'] > 0:
            metrics['compression_ratio'] = stats['compression_ratio']
        else:
            metrics['compression_ratio'] = 0.0
        
        # Token efficiency
        if stats['max_tokens'] > 0:
            metrics['token_utilization'] = stats['total_tokens'] / stats['max_tokens']
        else:
            metrics['token_utilization'] = 0.0
        
        # Memory retention
        metrics['retention_rate'] = stats['retained_important'] / max(stats['total_messages'], 1)
        
        # Response time (simulate retrieval)
        start_time = time.perf_counter()
        context = memory.get_context_string(max_tokens=500)
        retrieval_time = time.perf_counter() - start_time
        metrics['retrieval_time'] = retrieval_time
        
        details = {
            'total_messages': stats['total_messages'],
            'current_messages': stats['current_messages'],
            'compressed_messages': stats['compressed_messages'],
            'important_messages_retained': stats['retained_important']
        }
        
        result = EvaluationResult(
            memory_type='short_term',
            metrics=metrics,
            details=details,
            timestamp=time.time()
        )
        
        self.evaluation_history.append(result)
        return result
    
    def evaluate_episodic_memory(self, memory: EpisodicMemory) -> EvaluationResult:
        """Evaluate episodic memory performance."""
        stats = memory.get_stats()
        
        metrics = {}
        
        # Summary quality (using sentence embeddings similarity)
        if stats['current_summaries'] > 0:
            summary_quality = self._calculate_summary_quality(memory)
            metrics['summary_quality'] = summary_quality
        else:
            metrics['summary_quality'] = 0.0
        
        # Fact extraction accuracy
        metrics['fact_extraction_rate'] = stats['extracted_facts'] / max(stats['total_summaries'], 1)
        
        # Retrieval precision
        retrieval_precision = self._calculate_retrieval_precision(memory)
        metrics['retrieval_precision'] = retrieval_precision
        
        # Temporal coherence
        temporal_coherence = self._calculate_temporal_coherence(memory)
        metrics['temporal_coherence'] = temporal_coherence
        
        # Response time
        start_time = time.perf_counter()
        context = memory.get_context_from_summaries("test query", max_tokens=300)
        retrieval_time = time.perf_counter() - start_time
        metrics['retrieval_time'] = retrieval_time
        
        details = {
            'total_summaries': stats['total_summaries'],
            'current_summaries': stats['current_summaries'],
            'extracted_facts': stats['extracted_facts'],
            'message_count': stats['message_count']
        }
        
        result = EvaluationResult(
            memory_type='episodic',
            metrics=metrics,
            details=details,
            timestamp=time.time()
        )
        
        self.evaluation_history.append(result)
        return result
    
    def evaluate_long_term_memory(self, memory: LongTermMemory) -> EvaluationResult:
        """Evaluate long-term memory performance."""
        stats = memory.get_stats()
        
        metrics = {}
        
        # Retrieval precision
        retrieval_precision = self._calculate_long_term_retrieval_precision(memory)
        metrics['retrieval_precision'] = retrieval_precision
        
        # Temporal decay effectiveness
        temporal_decay = self._calculate_temporal_decay_effectiveness(memory)
        metrics['temporal_decay'] = temporal_decay
        
        # Fact diversity
        fact_diversity = self._calculate_fact_diversity(stats['fact_types'])
        metrics['fact_diversity'] = fact_diversity
        
        # Database efficiency
        if stats['database_size'] > 0:
            metrics['storage_efficiency'] = stats['extracted_entities'] / stats['database_size']
        else:
            metrics['storage_efficiency'] = 0.0
        
        # Response time
        start_time = time.perf_counter()
        facts = memory.retrieve_facts("test query", top_k=5)
        retrieval_time = time.perf_counter() - start_time
        metrics['retrieval_time'] = retrieval_time
        
        details = {
            'total_facts': stats['total_facts'],
            'extracted_entities': stats['extracted_entities'],
            'retrieved_facts': stats['retrieved_facts'],
            'fact_types': stats['fact_types']
        }
        
        result = EvaluationResult(
            memory_type='long_term',
            metrics=metrics,
            details=details,
            timestamp=time.time()
        )
        
        self.evaluation_history.append(result)
        return result
    
    def _calculate_summary_quality(self, memory: EpisodicMemory) -> float:
        """Calculate summary quality using sentence embeddings."""
        summaries = memory.summaries
        if not summaries:
            return 0.0
        
        # Calculate similarity between original conversation and summary
        total_similarity = 0.0
        count = 0
        
        for summary in summaries[-5:]:  # Check last 5 summaries
            if summary.conversation_window:
                # Get original text
                original_text = " ".join([msg['content'] for msg in summary.conversation_window])
                
                # Calculate embeddings
                original_emb = self.sentence_model.encode([original_text])[0]
                summary_emb = self.sentence_model.encode([summary.summary_text])[0]
                
                # Calculate cosine similarity
                similarity = np.dot(original_emb, summary_emb) / (
                    np.linalg.norm(original_emb) * np.linalg.norm(summary_emb)
                )
                
                total_similarity += similarity
                count += 1
        
        return total_similarity / max(count, 1)
    
    def _calculate_retrieval_precision(self, memory: EpisodicMemory) -> float:
        """Calculate retrieval precision for episodic memory."""
        # Simulate retrieval with test queries
        test_queries = [
            "medication", "symptom", "condition", "treatment", "diagnosis"
        ]
        
        total_precision = 0.0
        count = 0
        
        for query in test_queries:
            relevant_summaries = memory.get_relevant_summaries(query, top_k=3)
            
            # Check if summaries contain relevant content
            relevant_count = 0
            for summary in relevant_summaries:
                summary_text = summary.summary_text.lower()
                if any(term in summary_text for term in query.split()):
                    relevant_count += 1
            
            precision = relevant_count / max(len(relevant_summaries), 1)
            total_precision += precision
            count += 1
        
        return total_precision / max(count, 1)
    
    def _calculate_temporal_coherence(self, memory: EpisodicMemory) -> float:
        """Calculate temporal coherence of summaries."""
        summaries = memory.summaries
        if len(summaries) < 2:
            return 1.0
        
        # Check if consecutive summaries are temporally coherent
        coherent_pairs = 0
        total_pairs = len(summaries) - 1
        
        for i in range(total_pairs):
            current = summaries[i]
            next_summary = summaries[i + 1]
            
            # Check if timestamps are in order
            if next_summary.timestamp >= current.timestamp:
                coherent_pairs += 1
        
        return coherent_pairs / total_pairs
    
    def _calculate_long_term_retrieval_precision(self, memory: LongTermMemory) -> float:
        """Calculate retrieval precision for long-term memory."""
        # Simulate retrieval with test queries
        test_queries = [
            "medication", "symptom", "condition", "allergy", "lifestyle"
        ]
        
        total_precision = 0.0
        count = 0
        
        for query in test_queries:
            facts = memory.retrieve_facts(query, top_k=3)
            
            # Check if facts are relevant to query
            relevant_count = 0
            for fact in facts:
                fact_text = fact.text.lower()
                if any(term in fact_text for term in query.split()):
                    relevant_count += 1
            
            precision = relevant_count / max(len(facts), 1)
            total_precision += precision
            count += 1
        
        return total_precision / max(count, 1)
    
    def _calculate_temporal_decay_effectiveness(self, memory: LongTermMemory) -> float:
        """Calculate effectiveness of temporal decay."""
        # Get all facts and check their temporal weights
        all_docs = memory.vector_store.get(
            filter={"user_id": memory.user_id}
        )
        
        if not all_docs['ids']:
            return 1.0
        
        # Calculate average temporal weight
        total_weight = 0.0
        count = 0
        
        for metadata in all_docs['metadatas']:
            timestamp = metadata['timestamp']
            weight = memory.temporal_weighting.calculate_weight(timestamp)
            total_weight += weight
            count += 1
        
        return total_weight / max(count, 1)
    
    def _calculate_fact_diversity(self, fact_types: Dict[str, int]) -> float:
        """Calculate diversity of stored facts."""
        if not fact_types:
            return 0.0
        
        total_facts = sum(fact_types.values())
        num_types = len(fact_types)
        
        # Calculate entropy-based diversity
        entropy = 0.0
        for count in fact_types.values():
            p = count / total_facts
            entropy -= p * np.log2(p)
        
        # Normalize by max possible entropy
        max_entropy = np.log2(num_types) if num_types > 1 else 1.0
        diversity = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return diversity
    
    def evaluate_system_overall(self, short_term: ShortTermMemory, 
                               episodic: EpisodicMemory, 
                               long_term: LongTermMemory) -> Dict[str, Any]:
        """Evaluate the overall memory system performance."""
        
        # Evaluate each memory layer
        st_result = self.evaluate_short_term_memory(short_term)
        e_result = self.evaluate_episodic_memory(episodic)
        lt_result = self.evaluate_long_term_memory(long_term)
        
        # Calculate overall metrics
        overall_metrics = {
            'system_response_time': (
                st_result.metrics['retrieval_time'] +
                e_result.metrics['retrieval_time'] +
                lt_result.metrics['retrieval_time']
            ),
            'memory_integration_score': self._calculate_integration_score(st_result, e_result, lt_result),
            'storage_efficiency': (
                st_result.metrics['compression_ratio'] * 0.3 +
                e_result.metrics['fact_extraction_rate'] * 0.3 +
                lt_result.metrics['storage_efficiency'] * 0.4
            ),
            'retrieval_effectiveness': (
                e_result.metrics['retrieval_precision'] * 0.4 +
                lt_result.metrics['retrieval_precision'] * 0.6
            )
        }
        
        return {
            'timestamp': time.time(),
            'overall_metrics': overall_metrics,
            'memory_layers': {
                'short_term': st_result.metrics,
                'episodic': e_result.metrics,
                'long_term': lt_result.metrics
            },
            'recommendations': self._generate_recommendations(overall_metrics)
        }
    
    def _calculate_integration_score(self, st_result: EvaluationResult, 
                                   e_result: EvaluationResult, 
                                   lt_result: EvaluationResult) -> float:
        """Calculate integration score between memory layers."""
        # Integration score based on complementary strengths
        integration_factors = [
            st_result.metrics['retention_rate'],  # Short-term retention
            e_result.metrics['summary_quality'],  # Episodic quality
            lt_result.metrics['retrieval_precision']  # Long-term precision
        ]
        
        return np.mean(integration_factors)
    
    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations based on evaluation metrics."""
        recommendations = []
        
        if metrics['system_response_time'] > 1.0:
            recommendations.append("Consider optimizing memory retrieval speed")
        
        if metrics['storage_efficiency'] < 0.5:
            recommendations.append("Improve memory compression and storage efficiency")
        
        if metrics['retrieval_effectiveness'] < 0.7:
            recommendations.append("Enhance retrieval algorithms for better precision")
        
        if metrics['memory_integration_score'] < 0.6:
            recommendations.append("Improve integration between memory layers")
        
        if not recommendations:
            recommendations.append("Memory system is performing well!")
        
        return recommendations
    
    def get_evaluation_report(self) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report."""
        if not self.evaluation_history:
            return {"message": "No evaluations available yet."}
        
        # Group by memory type
        by_type = defaultdict(list)
        for result in self.evaluation_history:
            by_type[result.memory_type].append(result)
        
        # Calculate averages
        averages = {}
        for memory_type, results in by_type.items():
            avg_metrics = {}
            for metric in results[0].metrics.keys():
                values = [r.metrics[metric] for r in results]
                avg_metrics[metric] = np.mean(values)
            averages[memory_type] = avg_metrics
        
        return {
            'evaluation_count': len(self.evaluation_history),
            'memory_types_evaluated': list(by_type.keys()),
            'average_metrics': averages,
            'latest_evaluation': self.evaluation_history[-1] if self.evaluation_history else None,
            'timestamp': time.time()
        }


# Global evaluator instance
memory_evaluator = MemoryEvaluator()


def get_memory_evaluator() -> MemoryEvaluator:
    """Get the global memory evaluator instance."""
    return memory_evaluator