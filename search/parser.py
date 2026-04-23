from search.schemas import SearchResult

def make_unique_id(src: dict) -> str:
    doc_id = str(src.get("doc_id", ""))
    chunk_id = str(src.get("chunk_id", ""))
    return f"{doc_id}_{chunk_id}"

def parse_hits(hits: list[dict], method: str) -> list[SearchResult]:
    results = []

    for hit in hits:
        src = hit.get("_source", {})
        result = SearchResult(
            unique_id=make_unique_id(src),
            doc_id=str(src.get("doc_id", "")),
            chunk_id=int(src.get("chunk_id", 0)),
            score=float(hit.get("_score", 0.0) or 0.0),
            raw_score=float(hit.get("_score", 0.0) or 0.0),
            title=src.get("title", ""),
            source=src.get("source", ""),
            url=src.get("url", ""),
            section_title=src.get("section_title", ""),
            text=src.get("text", ""),
            char_length=int(src.get("char_length", 0) or 0),
            word_count=int(src.get("word_count", 0) or 0),
            retrieval_method=method,
            raw=hit,
        )
        
        results.append(result)
    
    return results