from search.schemas import SearchRequest
from search.parser import parse_hits
from search.filters import build_filter_clauses

class BM25Retriever:
    name = "bm25"

    def __init__(self, client, index_name: str):
        self.client = client
        self.index_name = index_name

    def search(self, request: SearchRequest):
        filter_clauses = build_filter_clauses(request.filters)

        body = {
            "size": request.top_k,
            "_source": [
                "doc_id", "chunk_id", "title", "source", "url",
                "section_title", "text", "char_length", "word_count"
            ],
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": request.query,
                                "fields": ["title^2.5", "section_title^1.8", "text"],
                                "type": "best_fields"
                            }
                        }
                    ],
                    "filter": filter_clauses
                }
            },
            "highlight": {
                "fields": {
                    "title": {},
                    "section_title": {},
                    "text": {}
                }
            }
        }

        resp = self.client.search(index=self.index_name, body=body)
        return parse_hits(resp["hits"]["hits"], method=self.name)

    
