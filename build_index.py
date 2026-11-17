"""Step 2: chunk the documents, embed each chunk, and store everything in ChromaDB.

Follows the AIPI 510 starter notebook: SentenceTransformer('all-MiniLM-L6-v2')
encodes the text, and collection.add() stores documents + embeddings + ids.
Added on top: word-window chunking with overlap, and per-chunk metadata.

Usage:
    python build_index.py          # build chromadb/ and print chunking diagnostics
"""

import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ---- Configuration: change these to experiment with chunking ----
CHUNK_SIZE = 120     # words per chunk
CHUNK_OVERLAP = 30   # words shared by consecutive chunks
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DB_DIR = "chromadb"
COLLECTION = "medlineplus"
DATA_FILE = Path("data/medlineplus_topics.jsonl")
HF_DATASET_REPO = "samuelteshome/medlineplus-health-topics"


def load_documents() -> list[dict]:
    """Load the dataset from the Hugging Face Hub, falling back to the local copy."""
    path = DATA_FILE
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(HF_DATASET_REPO, DATA_FILE.name, repo_type="dataset")
        print(f"Loaded dataset from Hugging Face: {HF_DATASET_REPO}")
    except Exception as e:  # offline or repo not reachable
        print(f"Hub download failed ({type(e).__name__}); using local {DATA_FILE}")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_words(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into windows of `size` words; each window starts `size - overlap` words later.

    The loop stops once a window reaches the end of the text, so the final words
    are always kept and no tiny tail chunk duplicates the previous one.
    """
    assert 0 <= overlap < size
    words = text.split()
    chunks = []
    step = size - overlap
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def make_chunks(docs: list[dict]) -> list[dict]:
    """Chunk every document and attach traceable metadata; skip empty and duplicate chunks."""
    records, seen = [], set()
    for doc in docs:
        position = 0
        for text in chunk_words(doc["text"]):
            if not text or text in seen:
                continue
            seen.add(text)
            records.append({
                "id": f"{doc['doc_id']}-chunk{position}",
                "text": text,
                "metadata": {
                    "title": doc["title"],
                    "url": doc["url"],
                    "doc_id": doc["doc_id"],
                    "chunk_position": position,
                },
            })
            position += 1
    return records


def print_diagnostics(docs: list[dict], chunks: list[dict], example_title: str = "Stroke"):
    words = [len(c["text"].split()) for c in chunks]
    print("\n=== Chunking diagnostics ===")
    print(f"Chunk size / overlap:   {CHUNK_SIZE} / {CHUNK_OVERLAP} words")
    print(f"Documents processed:    {len(docs)}")
    print(f"Total chunks generated: {len(chunks)}")
    print(f"Average words / chunk:  {sum(words) / len(words):.1f} (min {min(words)}, max {max(words)})")
    print(f"\nExample chunks from '{example_title}':")
    for c in chunks:
        if c["metadata"]["title"] == example_title:
            print(f"\n[{c['id']}] ({len(c['text'].split())} words)\n{c['text']}")


def build_index(docs: list[dict], db_dir: str = DB_DIR, model=None):
    model = model or SentenceTransformer(MODEL_NAME)
    chunks = make_chunks(docs)
    embeddings = model.encode([c["text"] for c in chunks], show_progress_bar=False)

    client = chromadb.PersistentClient(path=db_dir, settings=Settings(anonymized_telemetry=False))
    try:
        client.delete_collection(COLLECTION)  # rebuild from scratch, no stale chunks
    except Exception:
        pass
    # Cosine distance, so similarity = 1 - distance is a meaningful score in the app.
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings.tolist(),
        metadatas=[c["metadata"] for c in chunks],
    )
    return collection, chunks


if __name__ == "__main__":
    docs = load_documents()
    collection, chunks = build_index(docs)
    print_diagnostics(docs, chunks)
    print(f"\nStored {collection.count()} chunks in {DB_DIR}/ (collection '{COLLECTION}')")
