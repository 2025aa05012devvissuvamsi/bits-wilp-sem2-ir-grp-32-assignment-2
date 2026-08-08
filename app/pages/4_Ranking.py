"""Ranking visualization (assignment requirement D: use PageRank/HITS and show why ranking matters)."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.streamlit_helpers import get_doc_index, get_search_engine, index_data_exists

st.set_page_config(page_title="Ranking", layout="wide")
st.title("Ranking Visualization")

if not index_data_exists():
    st.warning("Index not built yet. Go to the **Index Management** page first.")
    st.stop()


def make_network_figure(graph, node_scores, doc_index):
    pos = nx.spring_layout(graph, seed=42, k=0.5)
    edge_x, edge_y = [], []
    for u, v in graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.5, color="#bbb"), hoverinfo="none")

    doc_index_idx = doc_index.set_index("doc_id")
    topics = sorted(doc_index["seed_source"].unique())
    palette = px.colors.qualitative.Set2
    topic_color = {t: palette[i % len(palette)] for i, t in enumerate(topics)}

    node_x, node_y, text, color, size = [], [], [], [], []
    for n in graph.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        title = doc_index_idx.loc[n, "title"] if n in doc_index_idx.index else n
        topic = doc_index_idx.loc[n, "seed_source"] if n in doc_index_idx.index else "unknown"
        score = node_scores.get(n, 0.0)
        text.append(f"{title}<br>Topic: {topic}<br>Score: {score:.3f}")
        color.append(topic_color.get(topic, "#888"))
        size.append(8 + 30 * score)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers", text=text, hoverinfo="text",
        marker=dict(color=color, size=size, line=dict(width=1, color="#333")),
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), height=600, margin=dict(l=0, r=0, t=20, b=0))
    return fig


from indexing.link_graph import load_graph

engine = get_search_engine()
doc_index = get_doc_index()
graph = load_graph()

algorithm = st.radio("Ranking algorithm", ["pagerank", "hits"], horizontal=True)
scores = engine.pagerank if algorithm == "pagerank" else engine.hits_authority

st.subheader("Link graph (node size/color intensity = authority score)")
st.plotly_chart(make_network_figure(graph, scores, doc_index), use_container_width=True)

st.subheader(f"Top 10 documents by {algorithm.upper()}")
doc_index_idx = doc_index.set_index("doc_id")
top_scores = sorted(scores.items(), key=lambda x: -x[1])[:10]
top_df = pd.DataFrame([
    {"title": doc_index_idx.loc[doc_id, "title"] if doc_id in doc_index_idx.index else doc_id,
     "seed_source": doc_index_idx.loc[doc_id, "seed_source"] if doc_id in doc_index_idx.index else "",
     "score": score}
    for doc_id, score in top_scores
])
fig = px.bar(top_df, x="title", y="score", color="seed_source")
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("See ranking matter: content-only vs. hybrid for a query")
query = st.text_input("Try a query", value="artificial intelligence")
if query:
    content_only, hybrid = engine.compare_rankings(query, top_k=8, link_algorithm=algorithm)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Content-only ranking**")
        fig = px.bar(content_only, x="title", y="content_score")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Hybrid ranking (content + link authority)**")
        fig = px.bar(hybrid, x="title", y="final_score")
        st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Compare the two orderings: documents with strong link authority but weaker topical "
        "similarity can move up (or down) once link-based ranking is blended in."
    )
