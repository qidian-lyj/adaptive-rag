"""演示界面：streamlit run app/demo_app.py

四种策略可视化对比：输入问题 -> 查看回答、引用来源、检索到的段落、路由决策。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st

from adaptive_rag.config import PROCESSED_DIR
from adaptive_rag.indexing import BM25Index, VectorIndex
from adaptive_rag.pipeline import AdaptiveRAG, NaiveRAG
from adaptive_rag.rerank import RerankRetriever
from adaptive_rag.retrieval import Retriever
from adaptive_rag.router import RuleBasedRouter

st.set_page_config(page_title="Adaptive-RAG Demo", page_icon="🔍", layout="wide")

STRATEGIES = {
    "Naive RAG（向量检索）": "naive",
    "Hybrid RAG（BM25+向量 RRF）": "hybrid",
    "Rerank（粗排+精排）": "rerank",
    "Adaptive RAG（自适应路由）": "adaptive",
}


@st.cache_resource(show_spinner="加载索引中...")
def load_components():
    corpus = {
        d["doc_id"]: d
        for d in (json.loads(l) for l in (PROCESSED_DIR / "corpus.jsonl").read_text(encoding="utf-8").splitlines())
    }
    base = Retriever(corpus, VectorIndex.load(), BM25Index.load(), mode="vector")
    return corpus, base


def build_pipeline(strategy: str):
    _, base = load_components()
    client, model = None, None
    try:
        from adaptive_rag.generation import get_client

        client, model = get_client()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    if strategy == "rerank":
        return NaiveRAG(RerankRetriever(base), client, model)
    if strategy == "adaptive":
        return AdaptiveRAG(base, RuleBasedRouter(), client, model)
    if strategy == "hybrid":
        from adaptive_rag.pipeline import HybridRAG

        base.mode = "hybrid"
        return HybridRAG(base, client, model)
    return NaiveRAG(base, client, model)


st.title("🔍 Adaptive-RAG：面向多跳问答的自适应检索增强问答系统")
st.caption("HotpotQA 语料库 · 9,769 篇维基段落 · Naive / Hybrid / Rerank / Adaptive 四种策略可切换对比")

strategy_label = st.radio("策略", list(STRATEGIES.keys()), horizontal=True)
question = st.text_input(
    "输入问题（试试多跳问题）",
    value="What is the birthdate of the actor who played the lead in the film directed by Christopher Nolan about dreams?",
)

if st.button("提问", type="primary"):
    if not question.strip():
        st.warning("请输入问题")
        st.stop()
    pipeline = build_pipeline(STRATEGIES[strategy_label])
    with st.spinner("检索 + 生成中..."):
        out = pipeline.ask(question.strip())

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("回答")
        st.markdown(out["answer"].replace("[", " [").replace("]", "] "))

    with col2:
        st.subheader("路由决策")
        badge = "🟢 单跳直接检索" if out.get("route") == "simple" else "🟠 " + str(out.get("route"))
        st.info(badge)
        if out.get("sub_questions"):
            st.write("**分解的子问题：**")
            for sq in out["sub_questions"]:
                st.markdown(f"- {sq}")

    st.subheader("引用的检索段落")
    for i, r in enumerate(out["retrieved"], start=1):
        with st.expander(f"[{i}] {r.title}（score={r.score:.3f}）"):
            st.write(r.text)
