import time, hashlib, numpy as np, re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, List

# Semantic cache removed — exact-only POC (ponytail: re-add via .[local] + FAISS when local embeddings land).
# Signature keeps semantic_* params as deprecated no-ops so callers don't change.
_SEMANTIC_AVAILABLE = False
try:
    from models.llm_io import LLMOutput
except ImportError:
    from src.models.llm_io import LLMOutput


@dataclass
class CacheEntry:
    key: str
    output: LLMOutput
    timestamp: float
    embedding: Optional[np.ndarray] = None
    context_hash: Optional[str] = None


class CAGHybridCache:
    """
    Cache-Augmented Generation — exact-only POC (semantic FAISS removed).
    Workflow: exact cache (O(1)) with poison bypass + TTL + LRU.
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

        # Semantic disabled (see module note); params kept for compat only.

        # Stats
        self.hits = 0
        self.misses = 0

    # ---------- Internal Helpers ----------

    def _normalize_key(self, query: str) -> str:
        query = query.lower().strip()
        query = re.sub(r"[^a-z0-9\s]", "", query)  # remove punctuation
        tokens = query.split()
        return " ".join(tokens)

    def _is_poisoned(self, text: str) -> bool:
        """LLM03 poison filter — detect instruction injection in query/context."""
        low = (text or "").lower()
        poison_markers = [
            "ignore previous",
            "ignore all previous",
            "reveal system prompt",
            "system: you are now",
            "### instruction",
            "### system:",
            "dan mode",
        ]
        return any(m in low for m in poison_markers)

    def _compose_key(self, query: str, context_ids: Optional[List[str]] = None) -> str:
        context_ids = context_ids or []
        ctx_part = "_".join(sorted(map(str, context_ids)))
        raw = f"{self._normalize_key(query)}::{ctx_part}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _is_expired(self, entry: CacheEntry) -> bool:
        return (time.time() - entry.timestamp) > self.ttl_seconds

    def _evict_if_needed(self):
        while len(self._exact_store) > self.max_size:
            self._exact_store.popitem(last=False)

    def _rebuild_faiss(self):
        """No-op compat shim (semantic removed)."""
        return

    # ---------- Public API ----------

    def get(self, query: str, context_ids: Optional[List[str]] = None) -> Optional[LLMOutput]:
        # Poisoned queries bypass cache (avoid serving poisoned hits)
        if self._is_poisoned(query):
            self.misses += 1
            return None
        key = self._compose_key(query, context_ids)

        # 1) Try exact cache first
        entry = self._exact_store.get(key)
        if entry and not self._is_expired(entry):
            self._exact_store.move_to_end(key)
            self.hits += 1
            return entry.output
        elif entry:
            self._exact_store.pop(key, None)  # expired
        # Semantic removed: exact-only
        self.misses += 1
        return None

    def put(self, query: str, output: LLMOutput):
        # Do not cache poisoned outputs
        if self._is_poisoned(query):
            return
        # Also check poison in contexts
        for ctx in output.contexts:
            if self._is_poisoned(getattr(ctx, "content", "")):
                return
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
        self._evict_if_needed()

    def stats(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "exact_size": len(self._exact_store),
            "semantic_size": 0,  # compat key, semantic removed
        }

    def clear(self):
        self._exact_store.clear()
        self.hits = 0
        self.misses = 0
        