from pydantic import BaseModel, Field
from typing import List, Optional


class ContextItem(BaseModel):
    source_id: str
    content: str

    # citation / highlight metadata
    section_path: Optional[str] = None
    page_numbers: Optional[List[int]] = None
    chunk_indices: List[int] = []

    score: Optional[float] = None
    dataset: Optional[str] = None


class LLMInput(BaseModel):
    query: str = Field(..., description="User's medical question.")
    contexts: List[ContextItem] = Field(
        ..., description="Retrieved passages relevant to the query.")


class LLMOutput(BaseModel):
    answer: str = Field(...,
                        description="Generated assistant's answer to the query.")
    cited_sources: List[int] = Field(default_factory=list)
    contexts: List[ContextItem]
