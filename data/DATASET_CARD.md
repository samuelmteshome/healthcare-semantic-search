---
license: other
license_name: public-domain-us-government
language:
- en
pretty_name: MedlinePlus Health Topics (Brain & Health subset)
task_categories:
- text-retrieval
tags:
- medical
- health
- semantic-search
size_categories:
- n<1K
---

# MedlinePlus Health Topics: Brain & Health subset

45 patient-education summaries from the U.S. National Library of Medicine's
[MedlinePlus health topics XML](https://medlineplus.gov/xml.html), selected for a semantic
search class project (Duke AIPI 510). Topics cover neurological conditions (stroke, dementia,
Alzheimer's, Parkinson's, delirium), cardiovascular disease, diabetes and other chronic
diseases, and general health education.

Each line of `medlineplus_topics.jsonl` is one topic:

| field | description |
|---|---|
| `doc_id` | `mplus-<MedlinePlus topic id>` |
| `title` | MedlinePlus topic title |
| `url` | original MedlinePlus page |
| `text` | the topic's full summary with HTML removed |

The text is unchanged apart from HTML removal. Built by `prepare_data.py` in
[github.com/samuelmteshome/healthcare-semantic-search](https://github.com/samuelmteshome/healthcare-semantic-search).

MedlinePlus health topic summaries are produced by NLM and are in the public domain.
Please credit MedlinePlus as the source. This content is educational and is not medical advice.
