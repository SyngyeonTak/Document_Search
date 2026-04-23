from search.retriever.bm25 import BM25Retriever
from search.retriever.vector import VectorRetriever
from search.retriever.hybrid import HybridRetriever
from search.embedder import get_embedder
from search.schemas import SearchRequest

class SearchService:
    def __init__(self, bm25, vector, hybrid):
        self.retrievers = {
            "bm25": bm25,
            "vector": vector,
            "hybrid": hybrid,
        }

    def search(self, mode: str, query: str, top_k: int = 10, filters=None):
        retriever = self.retrievers[mode]
        request = SearchRequest(
            query=query,
            top_k=top_k,
            filters=filters or {},
        )
        return retriever.search(request)


def get_search_service(client, index_name="hf_markdown_docs"):
    bm25 = BM25Retriever(client, index_name)
    vector = VectorRetriever(client, index_name, get_embedder(), )
    hybrid = HybridRetriever(bm25, vector)

    return SearchService(bm25, vector, hybrid)