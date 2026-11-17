"""Step 3: Streamlit search app over the ChromaDB index built by build_index.py."""

import chromadb
import streamlit as st
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from build_index import COLLECTION, DB_DIR, MODEL_NAME, build_index, load_documents

TOP_K = 5
EXAMPLES = [
    "What are common warning signs of stroke?",
    "What causes memory problems in older adults?",
    "How can high blood pressure affect the brain?",
    "What are the risks of general anesthesia?",
]


@st.cache_resource(show_spinner="Loading embedding model and index...")
def load_resources():
    """Runs once per server process, not on every interaction."""
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=DB_DIR, settings=Settings(anonymized_telemetry=False))
    names = [c if isinstance(c, str) else c.name for c in client.list_collections()]
    if COLLECTION not in names:
        # Space storage is not persistent: rebuild the (small) index if it is missing.
        build_index(load_documents(), db_dir=DB_DIR, model=model)
    return model, client.get_collection(COLLECTION)


def search(query: str, k: int = TOP_K) -> list[dict]:
    model, collection = load_resources()
    # Same model for queries as for the stored chunks, so vectors live in the same space.
    q = model.encode([query]).tolist()
    res = collection.query(query_embeddings=q, n_results=k)
    return [
        {"text": doc, "similarity": 1 - dist, **meta}
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


st.set_page_config(page_title="Brain & Health Search", page_icon="🧠")
st.title("Brain & Health Search")
st.write("Explore trustworthy medical information using semantic search.")
st.caption(
    "Educational information from NIH sources (MedQuAD: NINDS, NHLBI, MedlinePlus, GHR) on "
    "neurology, cardiology, and anesthesia. This tool retrieves passages from "
    "existing documents; it does not provide individualized medical advice. "
    "In an emergency, call 911."
)

if "query" not in st.session_state:
    st.session_state.query = ""

st.write("**Try an example:**")
cols = st.columns(2)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 2].button(ex, use_container_width=True):
        st.session_state.query = ex

with st.form("search"):
    query = st.text_input("Ask a health question", key="query",
                          placeholder="e.g., What are early signs of Parkinson's disease?")
    submitted = st.form_submit_button("Search", type="primary")

if query.strip():
    results = search(query.strip())
    st.subheader(f"Top {len(results)} passages")
    for r in results:
        with st.container(border=True):
            st.markdown(f"#### [{r['title']}]({r['url']})")
            st.write(r["text"])
            st.caption(f"{r['specialty']} · {r['source']} · chunk {r['chunk_position']} · "
                       f"cosine similarity: {r['similarity']:.3f} · {r['url']}")
