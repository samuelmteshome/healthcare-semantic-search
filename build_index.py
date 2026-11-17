"""Load the dataset, chunk the documents, embed each chunk, and store everything in ChromaDB.

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
COLLECTION = "brain_health"
HF_DATASET = "lavita/MedQuAD"                  # NIH question-answer pairs, CC BY 4.0
DATA_FILE = Path("data/medquad_subset.jsonl")  # local snapshot of the selected documents

# Topics kept from MedQuAD: (question_focus, document_source) -> specialty.
# NINDS = National Institute of Neurological Disorders and Stroke,
# NHLBI = National Heart, Lung, and Blood Institute, MPlusHealthTopics = MedlinePlus,
# GHR = Genetics Home Reference.
NEURO = ["Stroke", "Transient Ischemic Attack", "Atrial Fibrillation and Stroke", "Dementia",
         "Alzheimer's Disease", "Dementia With Lewy Bodies", "Frontotemporal Dementia",
         "Multi-Infarct Dementia", "Parkinson's Disease", "Deep Brain Stimulation for Parkinson's Disease",
         "Epilepsy", "Migraine", "Headache", "Traumatic Brain Injury", "Multiple Sclerosis",
         "Myasthenia Gravis", "Guillain-Barr Syndrome", "Peripheral Neuropathy", "Huntington's Disease",
         "Coma", "Cerebral Hypoxia", "Syncope"]
CARDIO = ["Heart Attack", "Heart Failure", "Atrial Fibrillation", "Arrhythmia", "Coronary Heart Disease",
          "Angina", "High Blood Pressure", "Hypotension", "Sudden Cardiac Arrest", "Cardiomyopathy",
          "Heart Valve Disease", "Cardiogenic Shock", "Pulmonary Embolism", "Deep Vein Thrombosis",
          "Carotid Artery Disease", "Electrocardiogram"]
TOPICS = (
    {(t, "NINDS"): "Neurology" for t in NEURO}
    | {("Delirium", "MPlusHealthTopics"): "Neurology"}
    | {(t, "NHLBI"): "Cardiology" for t in CARDIO}
    | {(t, "MPlusHealthTopics"): "Anesthesia & Perioperative" for t in
       ["Anesthesia", "Surgery", "After Surgery", "Heart Surgery", "Coronary Artery Bypass Surgery",
        "Critical Care", "Pain Relievers", "Blood Transfusion and Donation"]}
    | {(t, "GHR"): "Anesthesia & Perioperative" for t in
       ["malignant hyperthermia", "pseudocholinesterase deficiency"]}
    | {(t, "NHLBI"): "Anesthesia & Perioperative" for t in
       ["Respiratory Failure", "ARDS", "Oxygen Therapy", "Sleep Apnea"]}
)

# MedQuAD (pre-2019) links to old NINDS pages that now return 404. These are the current pages
# for the same topics (checked Nov 2026); the original link is kept as `source_url`.
_NINDS = "https://www.ninds.nih.gov/health-information/disorders/"
CURRENT_URLS = {t: _NINDS + slug for t, slug in {
    "Stroke": "stroke", "Transient Ischemic Attack": "transient-ischemic-attack-tia",
    "Atrial Fibrillation and Stroke": "stroke", "Dementia": "dementia",
    "Alzheimer's Disease": "alzheimers-disease", "Dementia With Lewy Bodies": "lewy-body-dementia",
    "Frontotemporal Dementia": "frontotemporal-degeneration", "Multi-Infarct Dementia": "vascular-dementia",
    "Parkinson's Disease": "parkinsons-disease",
    "Deep Brain Stimulation for Parkinson's Disease": "deep-brain-stimulation-parkinsons-disease",
    "Epilepsy": "epilepsy-and-seizures", "Migraine": "migraine", "Headache": "headache",
    "Traumatic Brain Injury": "traumatic-brain-injury-tbi", "Multiple Sclerosis": "multiple-sclerosis-ms",
    "Myasthenia Gravis": "myasthenia-gravis", "Guillain-Barr Syndrome": "guillain-barre-syndrome",
    "Peripheral Neuropathy": "peripheral-neuropathy", "Huntington's Disease": "huntingtons-disease",
    "Coma": "glossary-neurological-terms#coma",
    "Cerebral Hypoxia": "glossary-neurological-terms#cerebral%20hypoxia",
    "Syncope": "syncope-fainting",
}.items()}
assert set(NEURO) <= set(CURRENT_URLS)


def load_documents() -> list[dict]:
    """Download MedQuAD from the Hugging Face Hub and merge each topic's Q&A pairs into one document.

    Falls back to the local snapshot in data/ if the Hub is unreachable.
    """
    try:
        from datasets import load_dataset
        df = load_dataset(HF_DATASET, split="train").to_pandas()
    except Exception as e:
        print(f"Hub download failed ({type(e).__name__}); using local {DATA_FILE}")
        with DATA_FILE.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    df = df[df.answer.fillna("").str.strip() != ""]
    keys = list(zip(df.question_focus, df.document_source))
    df = df.assign(specialty=[TOPICS.get(k) for k in keys]).dropna(subset=["specialty"])
    missing = set(TOPICS) - set(zip(df.question_focus, df.document_source))
    assert not missing, f"Topics not found in {HF_DATASET}: {missing}"

    docs = []
    for (focus, source), g in df.groupby(["question_focus", "document_source"], sort=False):
        g = g.sort_values("question_id")
        docs.append({
            "doc_id": f"{source}-{g.document_id.iloc[0]}",
            "title": focus[0].upper() + focus[1:],
            "url": CURRENT_URLS.get(focus, g.document_url.iloc[0]) if source == "NINDS"
                   else g.document_url.iloc[0].replace("http://", "https://"),
            "source_url": g.document_url.iloc[0],
            "source": source,
            "specialty": g.specialty.iloc[0],
            # Keep each question with its answer so chunks read naturally.
            "text": " ".join(f"{q} {' '.join(a.split())}" for q, a in zip(g.question, g.answer)),
        })
    DATA_FILE.parent.mkdir(exist_ok=True)
    with DATA_FILE.open("w") as f:
        f.writelines(json.dumps(d, ensure_ascii=False) + "\n" for d in docs)
    print(f"Loaded {HF_DATASET} from Hugging Face; saved {len(docs)} documents to {DATA_FILE}")
    return docs


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
                    "source": doc["source"],
                    "specialty": doc["specialty"],
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
    for area in sorted({d["specialty"] for d in docs}):
        print(f"  {area}: {sum(d['specialty'] == area for d in docs)}")
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
