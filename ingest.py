import os
import re
import hashlib
from typing import List, Dict, Iterable

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from opensearchpy import OpenSearch, helpers
from opensearch_client import get_client

load_dotenv()

client = get_client()
print(client.info())
INDEX_NAME = os.getenv("OPENSEARCH_INDEX", "hf_markdown_docs")


def create_index(client: OpenSearch, index_name: str) -> None:
    if client.indices.exists(index=index_name):
        print(f"[INFO] Index already exists: {index_name}")
        return

    mapping = {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        },
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "chunk_id": {"type": "integer"},
                "title": {
                    "type": "text",
                    "fields": {
                        "keyword": {"type": "keyword", "ignore_above": 256}
                    }
                },
                "source": {"type": "keyword"},
                "url": {"type": "keyword"},
                "text": {"type": "text"},
                "section_title": {"type": "text"},
                "char_length": {"type": "integer"},
                "word_count": {"type": "integer"},
            }
        }
    }

    client.indices.create(index=index_name, body=mapping)
    print(f"[INFO] Created index: {index_name}")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_markdown_sections(text: str) -> List[str]:
    """
    Markdown heading(#, ##, ###...) 기준으로 section 분리.
    heading이 없으면 paragraph 단위 fallback.
    """
    text = normalize_whitespace(text)

    heading_pattern = re.compile(r"(?m)^(#{1,6}\s.+)$")
    matches = list(heading_pattern.finditer(text))

    if not matches:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paragraphs

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)

    return sections


def chunk_section(
    section_text: str,
    max_chars: int = 1200,
    overlap_chars: int = 150
) -> List[str]:
    """
    section 하나가 길면 paragraph 기준으로 다시 자르고,
    너무 긴 paragraph는 sliding window로 자름.
    """
    section_text = normalize_whitespace(section_text)

    if len(section_text) <= max_chars:
        return [section_text]

    paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars: # 2는 문단 사이 줄바꿈 길이 \n\n
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)

            if len(para) <= max_chars:
                current = para
            else:
                # 아주 긴 paragraph는 sliding window
                start = 0
                while start < len(para):
                    end = min(start + max_chars, len(para))
                    chunk = para[start:end].strip()
                    if chunk:
                        chunks.append(chunk)
                    if end == len(para):
                        break
                    start = max(0, end - overlap_chars)
                current = ""

    if current:
        chunks.append(current)

    return chunks

def extract_section_title(section_text: str) -> str:
    first_line = section_text.split("\n", 1)[0].strip()
    if first_line.startswith("#"):
        return re.sub(r"^#{1,6}\s*", "", first_line).strip()
    return ""

def build_chunks_from_document(
    raw_doc: Dict,
    max_chars: int = 1200,
    overlap_chars: int = 150
) -> List[Dict]:
    """
    데이터셋 필드명은 버전에 따라 약간 다를 수 있으므로 최대한 유연하게 처리.
    """
    text = raw_doc.get("markdown", "") or ""
    title = raw_doc.get("title", "") or raw_doc.get("file_path", "") or "untitled"
    url = raw_doc.get("url", "") or raw_doc.get("github_url", "") or ""
    source = url

    text = normalize_whitespace(text)
    if not text:
        return []

    doc_key = hashlib.md5((title + url + text[:300]).encode("utf-8")).hexdigest()

    sections = split_markdown_sections(text)
    output = []
    chunk_idx = 0

    for section in sections:
        section_title = extract_section_title(section)
        section_chunks = chunk_section(
            section_text=section,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )

        for chunk_text in section_chunks:
            output.append({
                "doc_id": doc_key,
                "chunk_id": chunk_idx,
                "title": title,
                "source": source,
                "url": url,
                "section_title": section_title,
                "text": chunk_text,
                "char_length": len(chunk_text),
                "word_count": len(chunk_text.split()),
            })
            chunk_idx += 1

    return output


def generate_actions(index_name: str, rows: Iterable[Dict]) -> Iterable[Dict]:
    for row in rows:
        unique_id = f"{row['doc_id']}_{row['chunk_id']}"
        yield {
            "_index": index_name,
            "_id": unique_id,
            "_source": row
        }


def main():
    client = get_client()
    create_index(client, INDEX_NAME)

    print("[INFO] Loading dataset from Hugging Face...")
    dataset = load_dataset("philschmid/markdown-documentation-transformers")

    train_split = dataset["train"]
    print(f"[INFO] Total rows: {len(train_split)}")

    all_chunks = []

    for row in tqdm(train_split, desc="Chunking documents"):
        chunks = build_chunks_from_document(
            row,
            max_chars=1200,
            overlap_chars=150
        )
        all_chunks.extend(chunks)

    print(f"[INFO] Total chunks: {len(all_chunks)}")

    print("[INFO] Bulk indexing...")
    helpers.bulk(
        client,
        generate_actions(INDEX_NAME, all_chunks),
        chunk_size=500,
        request_timeout=120
    )

    client.indices.refresh(index=INDEX_NAME)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()