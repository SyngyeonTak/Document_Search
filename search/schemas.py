from dataclasses import dataclass, field
from typing import Optional, Any

@dataclass
class SearchRequest:
    query: str
    top_k: int = 10
    filters:dict[str, Any] = field(default_factory=dict)

@dataclass
class SearchResult:
    unique_id: str
    doc_id: str
    chunk_id: int
    score: float
    title: str
    source: str
    url: str
    section_title: str
    text: str
    char_length: int = 0
    word_count: int = 0
    retrieval_method: str = ""
    raw_score: Optional[float] = None
    raw: Optional[dict] = None
    