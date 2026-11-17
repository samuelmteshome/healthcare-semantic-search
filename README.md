---
title: Brain & Health Search
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Semantic search over NIH MedlinePlus health topics
---

# Brain & Health Search

*Explore trustworthy medical information using semantic search.*

A semantic search engine over 45 NIH MedlinePlus health topic summaries, built for the
AIPI 510 (Data Sourcing for Analytics) semantic search activity. You type a health question in
plain language. The app returns the most relevant passages from the original MedlinePlus
documents, each with its title and a link to the source page. It does not generate answers.

- **Live app:** https://huggingface.co/spaces/samuelteshome/healthcare-semantic-search
- **Dataset:** https://huggingface.co/datasets/samuelteshome/medlineplus-health-topics
- **Code:** https://github.com/samuelmteshome/healthcare-semantic-search

> Educational information only. This tool does not provide individualized medical advice.

## Assignment requirements

| Requirement | How this project meets it |
|---|---|
| 1. Pick a dataset | 45 MedlinePlus health topics (neurology, cardiovascular, diabetes and chronic disease, general health), published as a Hugging Face dataset |
| 2. Chunk + embed the data | `build_index.py`: 120-word chunks with 30-word overlap, embedded with `all-MiniLM-L6-v2` |
| 3. Store in a Chroma vector DB | `build_index.py` writes a persistent Chroma collection (`chromadb/`) with ids, text, embeddings and metadata |
| 4. Build a search app | `app.py`: Streamlit app that embeds the query and retrieves the top 5 chunks from Chroma |
| 5. Publish on Hugging Face Spaces | Docker Space (see Deployment) |

## Dataset

[MedlinePlus](https://medlineplus.gov/) is the National Library of Medicine's patient
education site. NLM publishes all health topics as a daily [XML file](https://medlineplus.gov/xml.html).
`prepare_data.py` downloads that file, keeps 45 English topics chosen by title, strips the
HTML from each topic's summary, and writes `data/medlineplus_topics.jsonl` (one topic per line:
`doc_id`, `title`, `url`, `text`). The same file is published on the Hugging Face Hub as
[`samuelteshome/medlineplus-health-topics`](https://huggingface.co/datasets/samuelteshome/medlineplus-health-topics).
`build_index.py` loads it from the Hub and falls back to the local copy if the Hub is unreachable.

Topics include Stroke (and ischemic, hemorrhagic, rehabilitation), Dementia, Alzheimer's
Disease, Lewy Body Dementia, Parkinson's Disease, Delirium, Memory, Epilepsy, Concussion,
Multiple Sclerosis, High Blood Pressure, Heart Failure, Atrial Fibrillation, Cholesterol,
Diabetes (type 1, type 2, prediabetes, complications, A1C), Chronic Kidney Disease, COPD,
Asthma, Depression, Healthy Aging, Healthy Sleep, Exercise, Nutrition, Falls and others.
The full list is `TOPICS` in `prepare_data.py`. The text is not edited beyond HTML removal.
No patient-level data is used.

## Semantic search vs. keyword search

Keyword search matches the exact words in your query. A search for "tremor and slow movement"
only finds pages that contain those words.

Semantic search compares meaning. An embedding model turns every passage and every query into
a vector (here, 384 numbers) so that texts with similar meaning sit close together. The app
finds the passages whose vectors are closest to the query's vector. "Tremor and slow movement"
returns the Parkinson's disease passage that describes "trembling of hands" and "slowness of
movement", even though the wording differs.

## Architecture

```
MedlinePlus XML ──prepare_data.py──▶ 45 documents (HF dataset)
                                         │
                     build_index.py      ▼
                 120-word chunks, 30-word overlap  (+ title, url, ids, position)
                                         │
                 all-MiniLM-L6-v2        ▼
                 one 384-d embedding per chunk
                                         │
                                         ▼
                 ChromaDB collection "medlineplus" (cosine distance)
                                         ▲
user query ──▶ same embedding model ─────┘  top-5 nearest chunks ──▶ Streamlit results
```

This follows the course starter notebook: `SentenceTransformer` encodes the text,
`collection.add()` stores documents, embeddings and ids, and `collection.query()` with the
encoded query retrieves the nearest chunks. The project adds chunking, metadata, a
persistent index and a Streamlit front end.

## Chunking

**Why chunk?** An embedding squeezes a whole text into one vector. A 1,000-word topic page
about stroke covers risk factors, symptoms, diagnosis and treatment, so its single vector is
a blur of all of them. Splitting the page into short passages gives each passage a focused
vector. A question about warning signs can then match the passage about warning signs,
and the user reads that passage instead of the whole page.

**Strategy.** Each document is split into windows of `CHUNK_SIZE = 120` words. Each new window
starts `CHUNK_SIZE - CHUNK_OVERLAP = 90` words after the previous one, so consecutive chunks
share `CHUNK_OVERLAP = 30` words. Both constants sit at the top of `build_index.py`.

**Why overlap?** A fixed cut can split a sentence or list in half (for example, the F.A.S.T.
stroke signs). With overlap, text near a boundary appears in both neighbouring chunks, so at
least one chunk holds the full idea.

**Small vs. large chunks.** Small chunks give precise matches but can lose context (a list
without its heading). Large chunks keep context but mix topics, which dilutes the embedding
and shows the reader more irrelevant text. `all-MiniLM-L6-v2` reads at most 256 word pieces,
so chunks much longer than about 200 words would be truncated. 120 words sits safely inside
that limit.

**Edge cases.** The loop stops at the first window that reaches the end of the document, so
the last words are always kept and no tiny tail chunk repeats the previous one. Short
documents become a single chunk. Empty and duplicate chunks are skipped.

**Metadata.** Every chunk stores `title`, `url`, `doc_id` (parent document) and
`chunk_position`, plus a unique id such as `mplus-8-chunk4` (Stroke, 5th chunk). The app uses
this to show the source title and link under each passage, so every result traces back to
its original MedlinePlus page.

**Diagnostics** (`python build_index.py`):

```
=== Chunking diagnostics ===
Chunk size / overlap:   120 / 30 words
Documents processed:    45
Total chunks generated: 302
Average words / chunk:  113.8 (min 32, max 120)

Example chunks from 'Stroke':
[mplus-8-chunk0] (120 words)
What is a stroke? A stroke happens when there is a loss of blood flow to part of the brain. ...
```

The script prints every chunk of the Stroke page, so you can see the 30-word overlap between
neighbours.

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

To rebuild the dataset from the latest MedlinePlus XML: `python prepare_data.py`
(add `--push` to upload it to the Hub; requires `hf auth login`).

## Deployment (Hugging Face Spaces)

The Space uses the Docker SDK (see the YAML header of this README). The `Dockerfile`
installs CPU-only PyTorch and the requirements, then runs `build_index.py` during the image
build. The image therefore ships with the dataset, the model and a ready Chroma index, and
the app starts without your computer. Space storage is not persistent, so the app also
rebuilds the index on startup if it is ever missing. The model and index load once per
server process (`st.cache_resource`), not on every search.

To deploy your own copy:

```bash
hf repo create <user>/healthcare-semantic-search --repo-type space --space-sdk docker
git remote add space https://huggingface.co/spaces/<user>/healthcare-semantic-search
git push space main
```

## Example searches

- What are common warning signs of stroke?
- What causes memory problems in older adults?
- How can high blood pressure affect the brain?
- What are common symptoms of diabetes?
- Sudden confusion in a hospitalized older patient (returns Delirium)

## Limitations

- Small corpus (45 topics), so many questions have no good passage; the app still returns the 5 nearest.
- MedlinePlus summaries are short overviews. Some links between topics are thin; for example, "How can high blood pressure affect the brain?" returns blood-pressure passages rather than a passage on the brain.
- Several of the top 5 results can come from the same document.
- `all-MiniLM-L6-v2` is a general-purpose English model, not trained on medical text.
- The dataset is a snapshot of the MedlinePlus XML (November 2026); MedlinePlus pages may change.
- Retrieval only: the app returns source text and never interprets it for a specific patient.

## License and credit

Health topic text: MedlinePlus, U.S. National Library of Medicine (public domain).
Code by Samuel Teshome for Duke AIPI 510.
