import os
from dotenv import load_dotenv

load_dotenv()

INDEX_NAME = os.getenv("OPENSEARCH_INDEX", "hf_markdown_docs")
OPENSEARCH_MODEL_ID = os.getenv("OPENSEARCH_MODEL_ID")
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST")
OPENSEARCH_PORT = os.getenv("OPENSEARCH_PORT")
