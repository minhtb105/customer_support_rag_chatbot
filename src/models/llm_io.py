from pydantic import BaseModel, Field
from typing import List, Optional


class ContextItem(BaseModel):
    source_id: str
    content: str

    # citation / highlight metadata
    section_path: Optional[List[str]] = None
    page_numbers: Optional[List[int]] = None
    chunk_indices: List[int] = Field(default_factory=list)

    score: Optional[float] = None
    dataset: Optional[str] = None
    # --- enriched for tracing ---
    file_name: Optional[str] = None  # e.g., WHO_Classification_Diabetes_2019.pdf
    chunking_strategy: Optional[str] = None
    embedding_model: Optional[str] = None
    chunk_hash: Optional[str] = None
    updated_at: Optional[str] = None  # ISO from metadata_store.files.updated_at
    first_page: Optional[int] = None  # trang đầu tiên nếu chunk span nhiều trang

    model_config = {"extra": "allow"}


class LLMInput(BaseModel):
    query: str = Field(..., description="User's medical question.")
    contexts: List[ContextItem] = Field(
        ..., description="Retrieved passages relevant to the query.")


class LLMOutput(BaseModel):
    answer: str = Field(...,
                        description="Generated assistant's answer to the query.")
    cited_sources: List[int] = Field(default_factory=list)
    contexts: List[ContextItem]
