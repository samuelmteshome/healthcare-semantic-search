"""Smoke test: chunking, indexing, and retrieval on the real dataset (local copy)."""

import json

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from build_index import COLLECTION, DATA_FILE, MODEL_NAME, build_index, chunk_words


def test_chunking_keeps_tail_and_overlap():
    words = [f"w{i}" for i in range(250)]
    chunks = chunk_words(" ".join(words), size=120, overlap=30)
    assert [len(c.split()) for c in chunks] == [120, 120, 70]
    assert chunks[0].split()[-30:] == chunks[1].split()[:30]  # 30-word overlap
    assert chunks[-1].split()[-1] == "w249"                  # last word kept
    assert chunk_words("short text") == ["short text"]
    assert chunk_words("") == []


def test_index_and_search(tmp_path):
    docs = [json.loads(line) for line in open(DATA_FILE)]
    model = SentenceTransformer(MODEL_NAME)
    collection, chunks = build_index(docs, db_dir=str(tmp_path), model=model)
    assert collection.count() == len(chunks) > len(docs)
    assert len({c["id"] for c in chunks}) == len(chunks)

    # Reload from disk, as the app does, and query with the same model.
    client = chromadb.PersistentClient(path=str(tmp_path), settings=Settings(anonymized_telemetry=False))
    col = client.get_collection(COLLECTION)
    res = col.query(query_embeddings=model.encode(["warning signs of stroke"]).tolist(), n_results=3)
    meta = res["metadatas"][0][0]
    assert "Stroke" in meta["title"]
    assert meta["url"].startswith("https://medlineplus.gov/")
    assert 0 < 1 - res["distances"][0][0] <= 1
