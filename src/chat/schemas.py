"""Chat service schemas — RAG Q&A request/response."""
from __future__ import annotations

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="Câu hỏi y tế của người dùng", examples=["Dấu hiệu đái tháo đường type 2 là gì?"])
    top_k: int = Field(5, ge=1, le=20, description="Số chunk truy vấn")
    tone: Optional[Literal["strict", "friendly", "balanced"]] = Field(None, description="Tone trả lời; auto nếu None")
    user_id: str = Field("default_user", description="ID người dùng cho memory")
    include_audit: bool = Field(True, description="Có trả về audit trail (citations, faithfulness) không")


class CitationItem(BaseModel):
    source_id: str
    dataset: Optional[str] = None
    section_path: Optional[List[str]] = None
    page_numbers: Optional[List[int]] = None
    score: Optional[float] = None
    content_snippet: str = Field(..., description="Đoạn trích 300 ký tự đầu")


class AuditTrail(BaseModel):
    cited_sources: List[int]
    citations: List[CitationItem]
    prompt_version: Optional[str] = None
    reranker_model: Optional[str] = None
    latency_ms: Optional[float] = None
    faithfulness_hint: Optional[str] = Field(None, description="Gợi ý faithfulness (nếu bật LLM-as-judge)")


class QueryResponse(BaseModel):
    answer: str
    cited_sources: List[int]
    contexts: List[Dict[str, Any]]
    audit: Optional[AuditTrail] = None
    langsmith: Optional[Dict[str, Any]] = None
    cache_hit: bool = False
    timings: Optional[Dict[str, float]] = None
    # HILT extensions
    status: Optional[str] = Field(None, description="answered | pending_review")
    review_id: Optional[str] = None
    evaluation: Optional[Dict[str, Any]] = None
    is_low_confidence: Optional[bool] = None
    effective_user_id: Optional[str] = None

    model_config = {"extra": "allow"}
