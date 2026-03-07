"""
Retrieval Utility Functions

Common utilities shared across retrieval components.
"""

from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document


def should_merge(query: str) -> bool:
    """Determine if query should trigger section merging."""
    q = query.lower()
    return any(k in q for k in [
        "summary", "summarize", "overview",
        "key facts", "describe", "explain",
        "list", "what are", "how does"
    ])


def trim_text(text: str, max_chars: int = 4000) -> str:
    """Trim text to maximum character limit."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[Content truncated]"


def make_section_key(section: Any) -> Optional[Tuple]:
    """Create a hashable key for section grouping."""
    if not section:
        return None
    if isinstance(section, list):
        return tuple(section)
    return section


def deserialize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Deserialize metadata from string format back to proper types."""
    if "section_path" in meta and isinstance(meta["section_path"], str):
        meta["section_path"] = meta["section_path"].split("|||")
        
    if "page_numbers" in meta and isinstance(meta["page_numbers"], str):
        meta["page_numbers"] = [
            int(x) for x in meta["page_numbers"].split("|||") if x
        ]
        
    return meta


def normalize_docs(docs: List[Document]) -> List[Document]:
    """Normalize document metadata for consistent processing."""
    for d in docs:
        d.metadata = deserialize_metadata(d.metadata)
    return docs


def merge_documents_by_section(
    docs: List[Document], 
    query: str
) -> List[Document]:
    """Merge documents by section if query suggests it."""
    if not should_merge(query):
        return docs
    
    # Group by source_id and section_path
    grouped = {}
    for doc in docs:
        sid = doc.metadata.get("source_id")
        section = make_section_key(doc.metadata.get("section_path"))
        key = (sid, section)
        grouped.setdefault(key, []).append(doc)
    
    # Merge documents in each group
    merged = []
    for key, parts in grouped.items():
        parts = sorted(parts, key=lambda d: d.metadata.get("chunk_index", 0))
        merged_text = "\n\n---\n\n".join(p.page_content for p in parts)
        merged_text = trim_text(merged_text)
        
        # Create merged document with combined metadata
        merged_doc = Document(
            page_content=merged_text,
            metadata={
                **parts[0].metadata,
                "chunk_indices": [d.metadata.get("chunk_index") for d in parts if d.metadata.get("chunk_index") is not None],
                "page_numbers": sorted({
                    p for d in parts
                    for p in (d.metadata.get("page_numbers") or [])
                })
            }
        )
        merged.append(merged_doc)
    
    return merged


def calculate_document_score(doc: Document, query: str) -> float:
    """Calculate relevance score for a document."""
    # Simple heuristic: higher score for documents with query terms
    query_terms = set(query.lower().split())
    doc_text = doc.page_content.lower()
    doc_terms = set(doc_text.split())
    
    overlap = len(query_terms & doc_terms)
    return overlap / max(len(query_terms), 1)