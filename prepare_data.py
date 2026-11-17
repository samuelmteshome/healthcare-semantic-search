"""Step 1: build the dataset from the official MedlinePlus health topics XML.

Downloads the XML (or reads a local copy), keeps a small list of topics,
strips HTML from each summary, and writes data/medlineplus_topics.jsonl.
With --push, it also uploads that file to a Hugging Face dataset repo.

Usage:
    python prepare_data.py [--xml path/to/mplus_topics.xml] [--push]
"""

import argparse
import html
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

XML_INDEX = "https://medlineplus.gov/xml.html"
OUT_FILE = Path("data/medlineplus_topics.jsonl")
HF_DATASET_REPO = "samuelteshome/medlineplus-health-topics"

# 45 topics: neurology first, then cardiovascular, diabetes/chronic disease, general health.
TOPICS = [
    # Neurological
    "Stroke", "Ischemic Stroke", "Hemorrhagic Stroke", "Stroke Rehabilitation",
    "Dementia", "Alzheimer's Disease", "Alzheimer's Caregivers", "Lewy Body Dementia",
    "Parkinson's Disease", "Delirium", "Memory", "Headache", "Seizures", "Epilepsy",
    "Concussion", "Multiple Sclerosis",
    # Cardiovascular
    "High Blood Pressure", "How to Prevent High Blood Pressure", "Low Blood Pressure",
    "Heart Diseases", "Heart Attack", "Heart Failure", "Atrial Fibrillation",
    "Cholesterol", "How to Prevent Heart Disease",
    # Diabetes and other chronic disease
    "Diabetes", "Diabetes Type 1", "Diabetes Type 2", "Prediabetes",
    "Diabetes Complications", "A1C", "Obesity", "Chronic Kidney Disease",
    "Asthma", "COPD", "Depression",
    # General health education
    "Healthy Aging", "Healthy Sleep", "Sleep Apnea", "Benefits of Exercise",
    "Exercise for Older Adults", "Nutrition", "Nutrition for Older Adults",
    "Quitting Smoking", "Falls",
]


def latest_xml_url() -> str:
    page = urllib.request.urlopen(XML_INDEX).read().decode()
    # File names carry the date (mplus_topics_YYYY-MM-DD.xml), so the max is the newest.
    return "https://medlineplus.gov/xml/" + max(re.findall(r"mplus_topics_\d{4}-\d{2}-\d{2}\.xml", page))


def clean_html(raw: str) -> str:
    """Remove tags and entities, collapse whitespace."""
    # Block tags (paragraphs, list items, headings) become spaces; inline tags (<a>, <b>) vanish.
    text = re.sub(r"</?(p|li|ul|ol|br|h\d|div)\b[^>]*>", " ", raw or "")
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def load_topics(xml_bytes: bytes) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    by_title = {
        t.get("title"): t
        for t in root.findall("health-topic")
        if t.get("language") == "English"
    }
    missing = [t for t in TOPICS if t not in by_title]
    if missing:
        raise SystemExit(f"Topics not found in XML: {missing}")
    docs = []
    for title in TOPICS:
        t = by_title[title]
        docs.append({
            "doc_id": f"mplus-{t.get('id')}",
            "title": title,
            "url": t.get("url"),
            "text": clean_html(t.findtext("full-summary")),
        })
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", help="local MedlinePlus XML file (default: download the latest)")
    ap.add_argument("--push", action="store_true", help="upload the JSONL to the Hugging Face Hub")
    args = ap.parse_args()

    if args.xml:
        xml_bytes = Path(args.xml).read_bytes()
    else:
        url = latest_xml_url()
        print("Downloading", url)
        xml_bytes = urllib.request.urlopen(url).read()

    docs = load_topics(xml_bytes)
    OUT_FILE.parent.mkdir(exist_ok=True)
    with OUT_FILE.open("w") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Wrote {len(docs)} documents to {OUT_FILE}")

    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(HF_DATASET_REPO, repo_type="dataset", exist_ok=True)
        api.upload_file(path_or_fileobj=str(OUT_FILE), path_in_repo=OUT_FILE.name,
                        repo_id=HF_DATASET_REPO, repo_type="dataset")
        api.upload_file(path_or_fileobj="data/DATASET_CARD.md", path_in_repo="README.md",
                        repo_id=HF_DATASET_REPO, repo_type="dataset")
        print(f"Pushed to https://huggingface.co/datasets/{HF_DATASET_REPO}")


if __name__ == "__main__":
    main()
