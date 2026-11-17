# Brain & Health Search

*Explore trustworthy medical information using semantic search.*

A semantic search engine over 53 NIH patient-education documents in three areas:
**neurology, cardiology, and anesthesia/perioperative care**. Built for the AIPI 510
(Data Sourcing for Analytics) semantic search activity. You type a health question in plain
language. The app returns the most relevant passages from the original NIH documents, each
with its title and a link to the source page. It does not generate answers.

- **Live app:** https://brain-health-search.streamlit.app
- **Code:** https://github.com/samuelmteshome/healthcare-semantic-search
- **Dataset:** [`lavita/MedQuAD`](https://huggingface.co/datasets/lavita/MedQuAD) on the Hugging Face Hub

> Educational information only. This tool does not provide individualized medical advice.

## Assignment requirements

| Requirement | How this project meets it |
|---|---|
| 1. Pick a dataset | 53 documents selected from the public Hugging Face dataset `lavita/MedQuAD` (NIH question-answer pairs) |
| 2. Chunk + embed the data | `build_index.py`: 120-word chunks with 30-word overlap, embedded with `all-MiniLM-L6-v2` |
| 3. Store in a Chroma vector DB | `build_index.py` writes a persistent Chroma collection (`chromadb/`) with ids, text, embeddings and metadata |
| 4. Build a search app | `app.py`: Streamlit app that embeds the query and retrieves the top 5 chunks from Chroma |
| 5. Publish the app | Live at https://brain-health-search.streamlit.app (Streamlit Community Cloud). See Deployment for why not Hugging Face Spaces |

## Dataset

[MedQuAD](https://github.com/abachaa/MedQuAD) (Medical Question Answering Dataset) contains
question-answer pairs built from 12 trusted NIH websites. Each record has a question
("What are the symptoms of Stroke?"), the answer text from the NIH page, the topic
(`question_focus`), the source website and the page URL. This project loads the Hugging Face
version, [`lavita/MedQuAD`](https://huggingface.co/datasets/lavita/MedQuAD) (47,441 records),
with the `datasets` library.

`build_index.py` keeps 53 topics and merges all question-answer pairs of a topic into one
document:

| Area | Docs | Source | Topics (examples) |
|---|---|---|---|
| Neurology | 23 | NINDS, MedlinePlus | stroke, TIA, dementia, Alzheimer's, Lewy body dementia, Parkinson's, epilepsy, migraine, TBI, MS, myasthenia gravis, Guillain-Barré, coma, cerebral hypoxia, syncope, delirium |
| Cardiology | 16 | NHLBI | heart attack, heart failure, atrial fibrillation, arrhythmia, coronary heart disease, angina, high/low blood pressure, sudden cardiac arrest, cardiomyopathy, valve disease, cardiogenic shock, PE, DVT, ECG |
| Anesthesia & perioperative | 14 | MedlinePlus, Genetics Home Reference, NHLBI | anesthesia, surgery, after surgery, heart surgery, CABG, critical care, pain relievers, blood transfusion, malignant hyperthermia, pseudocholinesterase deficiency, respiratory failure, ARDS, oxygen therapy, sleep apnea |

The full list is `TOPICS` in `build_index.py`. Answer text is used as published; only extra
whitespace is collapsed. The selected documents are also saved to `data/medquad_subset.jsonl`,
which `build_index.py` uses if the Hugging Face Hub is unreachable. No patient-level data is used.

**Links.** MedQuAD predates its 2019 paper, and its NINDS page addresses now return 404.
`build_index.py` maps each NINDS topic to its current NINDS page (all 53 result links were
checked and return HTTP 200, November 2026). The original MedQuAD link is kept in each
document as `source_url`.

**License and citation.** MedQuAD is released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Ben Abacha A, Demner-Fushman D.
A question-entailment approach to question answering. *BMC Bioinformatics.* 2019.
[doi:10.1186/s12859-019-3119-4](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3119-4).
The `lavita/MedQuAD` version omits answers from subsets that MedlinePlus restricts by copyright.

## Semantic search vs. keyword search

Keyword search matches the exact words in your query. A search for "prolonged paralysis after
succinylcholine" only finds pages that contain those words.

Semantic search compares meaning. An embedding model turns every passage and every query into
a vector (here, 384 numbers) so that texts with similar meaning sit close together. The app
finds the passages whose vectors are closest to the query's vector. "Prolonged paralysis after
succinylcholine" returns the pseudocholinesterase deficiency passage, and "muscle rigidity and
high fever after anesthetic gas" returns malignant hyperthermia, even when the wording differs.

## Architecture

```
lavita/MedQuAD (Hugging Face) ──▶ 53 topic documents (neuro / cardio / anesthesia)
                                         │
                     build_index.py      ▼
                 120-word chunks, 30-word overlap  (+ title, url, specialty, ids, position)
                                         │
                 all-MiniLM-L6-v2        ▼
                 one 384-d embedding per chunk
                                         │
                                         ▼
                 ChromaDB collection "brain_health" (cosine distance)
                                         ▲
user query ──▶ same embedding model ─────┘  top-5 nearest chunks ──▶ Streamlit results
```

This follows the course starter notebook: `SentenceTransformer` encodes the text,
`collection.add()` stores documents, embeddings and ids, and `collection.query()` with the
encoded query retrieves the nearest chunks. The project adds chunking, metadata, a
persistent index and a Streamlit front end.

## Chunking

**Why chunk?** An embedding squeezes a whole text into one vector. The NHLBI heart failure
document is over 2,000 words and covers causes, symptoms, diagnosis and treatment, so a single
vector would be a blur of all of them. Splitting the document into short passages gives each
passage a focused vector. A question about symptoms can then match the passage about
symptoms, and the user reads that passage instead of the whole document.

**Strategy.** Each document is split into windows of `CHUNK_SIZE = 120` words. Each new window
starts `CHUNK_SIZE - CHUNK_OVERLAP = 90` words after the previous one, so consecutive chunks
share `CHUNK_OVERLAP = 30` words. Both constants sit at the top of `build_index.py`.

**Why overlap?** A fixed cut can split a sentence or list in half (for example, a list of
stroke symptoms). With overlap, text near a boundary appears in both neighbouring chunks, so
at least one chunk holds the full idea.

**Small vs. large chunks.** Small chunks give precise matches but can lose context (a list
without its heading). Large chunks keep context but mix topics, which dilutes the embedding
and shows the reader more irrelevant text. `all-MiniLM-L6-v2` reads at most 256 word pieces,
so chunks much longer than about 200 words would be truncated. 120 words sits safely inside
that limit.

**Edge cases.** The loop stops at the first window that reaches the end of the document, so
the last words are always kept and no tiny tail chunk repeats the previous one. Short
documents become a single chunk. Empty and duplicate chunks are skipped.

**Metadata.** Every chunk stores `title`, `url`, `doc_id` (parent document), `source`,
`specialty` and `chunk_position`, plus a unique id such as `NINDS-0000261-chunk0` (first chunk
of the NINDS stroke document). The app shows the title, link, specialty and chunk position
under each passage, so every result traces back to its source page.

**Diagnostics** (`python build_index.py`):

```
=== Chunking diagnostics ===
Chunk size / overlap:   120 / 30 words
Documents processed:    53
  Anesthesia & Perioperative: 14
  Cardiology: 16
  Neurology: 23
Total chunks generated: 912
Average words / chunk:  116.9 (min 31, max 120)

Example chunks from 'Stroke':
[NINDS-0000261-chunk0] (120 words)
What is (are) Stroke ? A stroke occurs when the blood supply to part of the brain is suddenly interrupted ...
```

The script prints every chunk of the stroke document, so you can see the 30-word overlap
between neighbours.

## Embedding model

[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
the model used in the starter notebook. It is free, small (about 90 MB), runs on CPU and
produces 384-dimensional vectors. The same model embeds the chunks and the queries, so both
live in the same vector space. The Chroma collection uses cosine distance, and the app shows
**cosine similarity = 1 − cosine distance** (higher means more similar; it is a ranking score,
not a probability or a measure of medical accuracy).

## Run locally

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python build_index.py          # load dataset, chunk, embed, write chromadb/
streamlit run app.py           # http://localhost:8501
python -m pytest -q tests      # smoke test: chunking, indexing, retrieval
```

## Deployment

**Streamlit Community Cloud (live app).** Free hosting for public GitHub repos.

1. Sign in at https://share.streamlit.io with GitHub.
2. Create app → repository `samuelmteshome/healthcare-semantic-search`, branch `main`, main file `app.py`.
3. Advanced settings → Python 3.12 → Deploy.

The host installs `requirements.txt` (CPU-only PyTorch, so no GPU libraries). The index
is not committed: hosted storage is not persistent, so on first start `app.py` downloads the
dataset and model and rebuilds the index (about 30 seconds). The model and index then load once
per server process (`st.cache_resource`), not on every search.

**Hugging Face Spaces.** The assignment suggests Spaces, but as of November 2026 Hugging Face
only hosts Gradio and Docker Spaces on free CPU for PRO accounts (`402 Payment Required` for
free accounts), and the Streamlit SDK option is gone. The included `Dockerfile` (port 7860,
index built at image build time) works for a Docker Space on a PRO account:

```bash
hf repo create <user>/healthcare-semantic-search --repo-type space --space-sdk docker
```

A Docker Space also needs this YAML header at the top of the Space's README:
`sdk: docker` and `app_port: 7860`.

## Example searches

- What are common warning signs of stroke?
- What causes memory problems in older adults?
- How can high blood pressure affect the brain?
- What are the risks of general anesthesia?
- Prolonged paralysis after succinylcholine (returns pseudocholinesterase deficiency)
- Chest pain when climbing stairs (returns heart failure, coronary heart disease, heart attack)

## Limitations

- Small corpus (53 topics), so many questions have no good passage; the app still returns the 5 nearest.
- MedQuAD answers were collected before 2019 and may not reflect current guidelines.
- MedQuAD questions are templates ("What is (are) Stroke ?"); they are kept because they label each answer.
- Long NHLBI documents produce many chunks, so cardiology passages can crowd the top 5. Several results can come from the same document.
- "How can high blood pressure affect the brain?" returns blood-pressure passages rather than a focused passage on the brain.
- `all-MiniLM-L6-v2` is a general-purpose English model, not trained on medical text.
- Retrieval only: the app returns source text and never interprets it for a specific patient.

## Credit

Data: MedQuAD (Ben Abacha and Demner-Fushman, 2019, CC BY 4.0), built from NIH websites
(NINDS, NHLBI, MedlinePlus, Genetics Home Reference).
Code by Samuel Teshome for Duke AIPI 510.
