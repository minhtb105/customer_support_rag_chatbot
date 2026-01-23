from pydantic import BaseModel, Field
from typing import Optional


class ChunkMetadata(BaseModel):
    section_path: Optional[str] = None
    page_numbers: Optional[str] = None
    chunk_index: int


class Chunk(BaseModel):
    source_id: str
    chunk_id: str
    text: str
    metadata: ChunkMetadata
