from copy import deepcopy

def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)

class HybridRetriever:
    name = "hybrid_rrf"

    def __init__(self, bm25_retriever, vector_retriever, alpha: float = 0.5):
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.alpha = alpha

        def search(self, request):
            bm25_results = self.bm25_retriever.search(
                type(request)(
                    query=request.query,
                    top_k=request.top_k * 3,
                    filters=request.filters,
                )
            )
            vector_results = self.vector_retriever.search(
                type(request)(
                    query=request.query,
                    top_k=request.top_k * 3,
                    filters=request.filters,
                )
            )

            score_map = {}
            result_map = {}

            for rank, item in enumerate(bm25_results, start=1):
                uid= item.unique_id
                result_map[uid] = deepcopy(item)
                score_map[uid] = score_map.get(uid, 0.0) + self.alpha * rrf_score(rank)

            for rank, item in enumerate(vector_results, start=1):
                uid = item.unique_id
                if uid not in result_map:
                    result_map[uid] = deepcopy(item)
                score_map[uid] = score_map.get(uid, 0.0) + (1 - self.alpha) * rrf_score(rank)

            ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:request.top_k]

            fused = []
            for uid, score in ranked:
                item = result_map[uid]
                item.score = score
                item.raw_score = score
                item.retrieval_method = self.name
                fused.append(item)

            return fused                
