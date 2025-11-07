import time, hashlib, json, numpy as np, faiss, re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, List
from sentence_transformers import SentenceTransformer
from models.llm_io import LLMOutput


@dataclass
class CacheEntry:
    key: str
    output: LLMOutput
    timestamp: float
    embedding: Optional[np.ndarray] = None
    context_hash: Optional[str] = None


class CAGHybridCache:
    """
    Cache-Augmented Generation with hybrid exact + semantic (FAISS) lookup.
    Workflow:
        1. Try exact cache (O(1))
        2. If not found → semantic FAISS search
    """
    def __init__(
        self,
        max_size: int = 1024,
        ttl_seconds: int = 60 * 60 * 24,
        semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        semantic_threshold: float = 0.82,
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.semantic_threshold = semantic_threshold

        # Exact cache (LRU)
        self._exact_store: "OrderedDict[str, CacheEntry]" = OrderedDict()

        # Semantic cache
        self._embedder = SentenceTransformer(semantic_model_name)
        self._dim = self._embedder.get_sentence_embedding_dimension()
        self._faiss_index = faiss.IndexFlatIP(self._dim)
        self._semantic_entries: List[CacheEntry] = []
        self._semantic_matrix: List[np.ndarray] = []

        # Stats
        self.hits = 0
        self.misses = 0

    # ---------- Internal Helpers ----------

    def _normalize_key(self, query: str) -> str:
        query = query.lower().strip()
        query = re.sub(r"[^a-z0-9\s]", "", query)  # remove punctuation
        tokens = query.split()
        
        return " ".join(tokens)

    def _compose_key(self, query: str, context_ids: Optional[List[str]] = None) -> str:
        context_ids = context_ids or []
        ctx_part = "_".join(sorted(map(str, context_ids)))
        raw = f"{self._normalize_key(query)}::{ctx_part}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _normalize_emb(self, emb: np.ndarray) -> np.ndarray:
        return emb / (np.linalg.norm(emb) + 1e-9)

    def _is_expired(self, entry: CacheEntry) -> bool:
        return (time.time() - entry.timestamp) > self.ttl_seconds

    def _evict_if_needed(self):
        while len(self._exact_store) > self.max_size:
            self._exact_store.popitem(last=False)

        while len(self._semantic_entries) > self.max_size:
            self._semantic_entries.pop(0)
            self._semantic_matrix.pop(0)
            self._rebuild_faiss()

    def _rebuild_faiss(self):
        """Rebuild FAISS index from stored embeddings."""
        self._faiss_index.reset()
        if self._semantic_matrix:
            mat = np.vstack(self._semantic_matrix).astype("float32")
            self._faiss_index.add(mat)

    # ---------- Public API ----------

    def get(self, query: str, context_ids: Optional[List[str]] = None) -> Optional[LLMOutput]:
        key = self._compose_key(query, context_ids)

        # 1) Try exact cache first
        entry = self._exact_store.get(key)
        if entry and not self._is_expired(entry):
            self._exact_store.move_to_end(key)
            self.hits += 1
            
            return entry.output
        elif entry:
            self._exact_store.pop(key, None)  # expired
            
        # Fallback: semantic search
        if not self._semantic_entries:
            self.misses += 1
            return None

        q_emb = self._normalize_emb(
            self._embedder.encode([query], convert_to_numpy=True).astype("float32")
        )
        D, I = self._faiss_index.search(q_emb, 1)
        best_score = float(D[0][0])
        best_idx = int(I[0][0])

        if best_score >= self.semantic_threshold:
            candidate = self._semantic_entries[best_idx]
            if not self._is_expired(candidate):
                self.hits += 1
                return candidate.output

        self.misses += 1
        return None

    def put(self, query: str, output: LLMOutput):
        context_ids = [ctx.source_id for ctx in output.contexts]
        key = self._compose_key(query, context_ids)

        # Exact cache insert
        exact_entry = CacheEntry(
            key=key,
            output=output,
            timestamp=time.time(),
        )
        self._exact_store[key] = exact_entry
        self._exact_store.move_to_end(key)

        # Semantic cache insert
        emb = self._normalize_emb(
            self._embedder.encode([query], convert_to_numpy=True).astype("float32")
        )
        sem_entry = CacheEntry(
            key=key,
            output=output,
            timestamp=time.time(),
            embedding=emb,
            context_hash=hashlib.md5(
                json.dumps(context_ids, sort_keys=True).encode()
            ).hexdigest(),
        )
        self._semantic_entries.append(sem_entry)
        self._semantic_matrix.append(emb)
        self._rebuild_faiss()
        self._evict_if_needed()

    def stats(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "exact_size": len(self._exact_store),
            "semantic_size": len(self._semantic_entries),
        }

    def clear(self):
        self._exact_store.clear()
        self._semantic_entries.clear()
        self._semantic_matrix.clear()
        self._faiss_index.reset()
        self.hits = 0
        self.misses = 0
        