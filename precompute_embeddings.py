"""
Precomputes embeddings for the 25 corpus documents and default ticker query embeddings,
saving everything into corpus/corpus_embeddings.npz.
"""

import os
import json
import numpy as np

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print("Loading sentence-transformers...", flush=True)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded!", flush=True)

with open("corpus/documents.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

passages = [
    f"Company: {d['company']} | Section: {d['section']} | {d['content']}"
    for d in docs
]
doc_ids = [d["doc_id"] for d in docs]

print(f"Encoding {len(passages)} passages...", flush=True)
embeddings = model.encode(passages, convert_to_numpy=True, normalize_embeddings=True)

# Also pre-encode queries for all 12 tickers + general queries
tickers = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ITC", "SBIN",
    "ICICIBANK", "LT", "TATAMOTORS", "SUNPHARMA", "ADANIENT", "BHARTIARTL"
]
query_embs = {}
for t in tickers:
    q = f"Financial performance, order book, debt, margins, and operational risks for {t}"
    query_embs[t] = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]

np.savez_compressed(
    "corpus/corpus_embeddings.npz",
    embeddings=embeddings,
    doc_ids=np.array(doc_ids),
    query_tickers=np.array(list(query_embs.keys())),
    query_embeddings=np.array(list(query_embs.values())),
)

print("Saved corpus/corpus_embeddings.npz successfully!", flush=True)
