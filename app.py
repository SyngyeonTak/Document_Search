import pandas as pd
import streamlit as st

from opensearch_client import get_client
from search.retriever.retrieval import get_source_terms
from search.retriever.service import get_search_service


@st.cache_resource
def get_cached_client():
    return get_client()


@st.cache_resource
def get_cached_search_service():
    client = get_cached_client()
    return get_search_service(client)


@st.cache_data
def cached_get_source_terms():
    client = get_cached_client()
    return get_source_terms(client)


@st.cache_data
def cached_run_search(mode: str, query: str, size: int, source_filter: str):
    service = get_cached_search_service()

    filters = {}
    if source_filter and source_filter != "ALL":
        filters["source"] = source_filter

    return service.search(
        mode=mode,
        query=query,
        top_k=size,
        filters=filters,
    )


def render_results(results, title: str):
    st.subheader(f"{title} ({len(results)}건)")

    for i, row in enumerate(results, start=1):
        with st.container(border=True):
            st.markdown(f"### {i}. {row.title or ''}")
            st.write(f"**score**: {(row.score or 0.0):.3f}")
            st.write(f"**source**: {row.source or ''}")

            if row.section_title:
                st.write(f"**section**: {row.section_title}")

            if row.url:
                st.markdown(f"**url**: {row.url}")

            if row.retrieval_method:
                st.write(f"**method**: {row.retrieval_method}")

            if getattr(row, "highlight", None):
                st.markdown("**highlight**")
                for h in (row.highlight or []):
                    st.markdown(h, unsafe_allow_html=True)

            st.markdown("**chunk text**")
            st.code((row.text or "")[:1500], language="markdown")


def main():
    st.set_page_config(page_title="OpenSearch Retrieval Playground", layout="wide")
    st.title("OpenSearch Retrieval Playground")
    st.caption("BM25 / Vector / Hybrid 검색 비교")

    with st.sidebar:
        st.header("검색 옵션")

        run_mode = st.radio(
            "실행 방식",
            options=["single", "compare"],
            format_func=lambda x: "단일 검색" if x == "single" else "3개 방식 비교",
        )

        search_mode = "bm25"
        if run_mode == "single":
            search_mode = st.selectbox(
                "검색 방식",
                options=["bm25", "vector", "hybrid"],
                index=0,
            )

        try:
            source_options = cached_get_source_terms()
        except Exception:
            source_options = ["ALL"]

        source_filter = st.selectbox("Source", source_options, index=0)
        size = st.slider("Top K", 3, 30, 10)

    query = st.text_input("질문 또는 검색어", value="How to load a model")

    if st.button("검색"):
        with st.spinner("검색 중..."):
            if run_mode == "single":
                results = cached_run_search(
                    mode=search_mode,
                    query=query,
                    size=size,
                    source_filter=source_filter,
                )
                render_results(results, f"{search_mode.upper()} 결과")

            else:
                bm25_results = cached_run_search(
                    mode="bm25",
                    query=query,
                    size=size,
                    source_filter=source_filter,
                )
                vector_results = cached_run_search(
                    mode="vector",
                    query=query,
                    size=size,
                    source_filter=source_filter,
                )
                hybrid_results = cached_run_search(
                    mode="hybrid",
                    query=query,
                    size=size,
                    source_filter=source_filter,
                )

                tab1, tab2, tab3 = st.tabs(["BM25", "Vector", "Hybrid"])

                with tab1:
                    render_results(bm25_results, "BM25 결과")

                with tab2:
                    render_results(vector_results, "Vector 결과")

                with tab3:
                    render_results(hybrid_results, "Hybrid 결과")

    st.divider()
    st.subheader("샘플 질의")
    sample_queries = [
        "How to load a pretrained model",
        "How to tokenize text",
        "Trainer usage example",
        "Pipeline for text generation",
        "dataset loading",
    ]
    st.write(pd.DataFrame({"sample_queries": sample_queries}))


if __name__ == "__main__":
    main()