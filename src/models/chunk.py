from pydantic import BaseModel
from typing import List, Optional


class ChunkMetadata(BaseModel):
    section_path: Optional[List[str]] = None
    page_numbers: Optional[List[int]] = None
    chunk_index: int


class Chunk(BaseModel):
    source_id: str
    chunk_id: str
    text: str
    metadata: ChunkMetadata
