# retrievers/vector.py
from search.schemas import SearchRequest
from search.parser import parse_hits
from search.filters import build_filter_clauses

class VectorRetriever:
    name = "vector"

    def __init__(self, client, index_name: str, embedder, vector_field: str = "embedding"):
        self.client = client
        self.index_name = index_name
        self.embedder = embedder
        self.vector_field = vector_field

    def search(self, request: SearchRequest):
        query_vector = self.embedder.embed_query(request.query)
        filter_clauses = build_filter_clauses(request.filters)

        knn_query = {
            "vector": query_vector,
            "k": request.top_k,
        }

        if filter_clauses:
            knn_query["filter"] = {
                "bool": {
                    "filter": filter_clauses
                }
            }

        body = {
            "size": request.top_k,
            "_source": [
                "doc_id", "chunk_id", "title", "source", "url",
                "section_title", "text", "char_length", "word_count"
            ],
            "query": {
                "knn": {
                    self.vector_field: knn_query
                }
            }
        }

        resp = self.client.search(index=self.index_name, body=body)
        return parse_hits(resp["hits"]["hits"], method=self.name)