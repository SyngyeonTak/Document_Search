import requests
from .config import OPENSEARCH_MODEL_ID
from opensearch_client import get_client

class OpenSearchEmbedder:
    def __init__(self, model_id: str):
        self.client = get_client()
        self.model_id = model_id

    def embed_query(self, text: str) -> list[float]:
        body = {
            "text_docs": [text],
            "return_number": True,
            "target_response": ["sentence_embedding"]
        }

        res = self.client.transport.perform_request(
            method="POST",
            url=f"/_plugins/_ml/_predict/text_embedding/{self.model_id}",
            body=body
        )

        return res["inference_results"][0]["output"][0]["data"]


_embedder_instance = None


def get_embedder():
    global _embedder_instance

    if _embedder_instance is None:
        model_id = OPENSEARCH_MODEL_ID

        if not model_id:
            raise ValueError("OPENSEARCH_MODEL_ID is not set in .env")

        _embedder_instance = OpenSearchEmbedder(model_id)

    return _embedder_instance