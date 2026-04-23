from __future__ import annotations
from typing import Any

def _is_valid_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if value == "ALL":
        return False
    return True


def build_filter_clauses(filters: dict[str, Any] | None) -> list[dict]:
    """
    OpenSearch bool.filter 에 들어갈 clause 리스트를 생성합니다.

    예:
        filters = {
            "source": "https://github.com/huggingface/transformers",
            "doc_id": "abc123",
        }

    반환:
        [
            {"term": {"source.keyword": "..."}},
            {"term": {"doc_id.keyword": "abc123"}}
        ]
    """
    if not filters:
        return []

    clauses: list[dict] = []

    source = filters.get("source")
    if _is_valid_value(source):
        clauses.append({"term": {"source.keyword": source}})

    doc_id = filters.get("doc_id")
    if _is_valid_value(doc_id):
        clauses.append({"term": {"doc_id.keyword": doc_id}})

    title = filters.get("title")
    if _is_valid_value(title):
        clauses.append({"term": {"title.keyword": title}})

    url = filters.get("url")
    if _is_valid_value(url):
        clauses.append({"term": {"url.keyword": url}})

    return clauses
