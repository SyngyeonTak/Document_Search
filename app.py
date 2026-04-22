import os
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
from opensearch_client import get_client

load_dotenv()

INDEX_NAME = os.getenv("OPENSEARCH_INDEX", "hf_markdown_docs")

@st.cache_data
def search_docs(_client, query: str, size: int = 10, source_filter: str = ""):
    must_clauses = []
    filter_clauses = []

    if query.strip():
        must_clauses.append({
            "multi_match": {
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


def get_source_terms(client):
    body = {
        "size": 0,
        "aggs": {
            "sources": {
                "terms": {
                    "field": "source",
                    "size": 50
                }
            }
        }
    }
    resp = client.search(index=INDEX_NAME, body=body)
    buckets = resp["aggregations"]["sources"]["buckets"]
    return ["ALL"] + [b["key"] for b in buckets]


def main():
    st.set_page_config(page_title="OpenSearch Markdown Docs Search", layout="wide")
    st.title("OpenSearch Markdown Documentation Search")
    st.caption("Hugging Face markdown docs + markdown-aware chunking + BM25 search")

    client = get_client()

    with st.sidebar:
        st.header("검색 옵션")
        try:
            source_options = get_source_terms(client)
        except Exception:
            source_options = ["ALL"]
        source_filter = st.selectbox("Source", source_options, index=0)
        size = st.slider("Top K", 3, 30, 10)

    query = st.text_input("질문 또는 검색어", value="How to load a model")

    if st.button("검색"):
        with st.spinner("검색 중..."):
            results = search_docs(client, query=query, size=size, source_filter=source_filter)

        st.subheader(f"검색 결과: {len(results)}건")

        for i, row in enumerate(results, start=1):
            with st.container(border=True):
                st.markdown(f"### {i}. {row['title']}")
                st.write(f"**score**: {row['score']:.3f}")
                st.write(f"**source**: {row['source']}")
                if row["section_title"]:
                    st.write(f"**section**: {row['section_title']}")
                if row["url"]:
                    st.markdown(f"**url**: {row['url']}")

                if row["highlight"]:
                    st.markdown("**highlight**")
                    for h in row["highlight"]:
                        st.markdown(h, unsafe_allow_html=True)

                st.markdown("**chunk text**")
                st.code(row["text"][:1500], language="markdown")

    st.divider()
    st.subheader("샘플 질의")
    sample_queries = [
        "How to load a pretrained model",
        "How to tokenize text",
        "Trainer usage example",
        "Pipeline for text generation",
        "dataset loading"
    ]
    st.write(pd.DataFrame({"sample_queries": sample_queries}))


if __name__ == "__main__":
    main()