from pydantic import BaseModel, Field
from typing import List, Dict


class ContextItem(BaseModel):
    source_id: str
    content: str
    score: float | None = None
    dataset: str | None = None
    
class LLMInput(BaseModel):
    query: str = Field(..., description="User's medical question.")
    contexts: List[ContextItem] = Field(..., description="Retrieved passages relevant to the query.")
    
class LLMOutput(BaseModel):
    answer: str = Field(..., description="Generated assistant's answer to the query.")
    cited_sources: List[int] = Field(default_factory=list)
    contexts: List[ContextItem]
    