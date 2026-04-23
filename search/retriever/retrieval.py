import streamlit as st
from ..config import INDEX_NAME

def search_docs(_client, query: str, size: int = 10, source_filter: str = ""):
    must_clauses = []
    filter_clauses = []

    if query.strip():
        must_clauses.append({
            "multi_match" : {
                "query": query,
                "fields": ["title^2", "section_title^1.5", "text"],
                "type": "best_fields"
            }
        })
    else:
        must_clauses.append({"match_all": {}})

    if source_filter and source_filter != "ALL":
        filter_clauses.append({"term": {"source": source_filter}})

    body = {
        "size": size,
        "_source": [
            "title", "source", "url", "section_title",
            "text", "chunk_id", "char_length", "word_count"
        ],
        "query": {
            "bool": {
                "must": must_clauses,
                "filter": filter_clauses
            }
        },
        "highlight": {
            "fields": {
                "text": {},
                "title": {},
                "section_title": {}
            }
        }
    }

    resp = _client.search(index=INDEX_NAME, body=body)
    hits = resp["hits"]["hits"]

    rows = []
    for hit in hits:
        src = hit["_source"]
        rows.append({
            "score": hit["_score"],
            "title": src.get("title", ""),
            "source": src.get("source", ""),
            "url": src.get("url", ""),
            "section_title": src.get("section_title", ""),
            "chunk_id": src.get("chunk_id", ""),
            "text": src.get("text", ""),
            "highlight": hit.get("highlight", {}).get("text", []),
        })

    return rows

def get_source_terms(_client):
    body = {
        "size": 0,
        "aggs": {
            "source": {
                "terms": {
                    "field": "source",
                    "size": 50
                }
            }
        }
    }
    resp = _client.search(index=INDEX_NAME, body=body)
    buckets = resp["aggregations"]["source"]["buckets"]
    return ["ALL"] + [b["key"] for b in buckets]