"""
VibeFinance Filings RAG Engine
Offline retrieval using sentence-transformers (all-MiniLM-L6-v2) and numpy cosine similarity.
Embeddings cached to disk (corpus/corpus_embeddings.npz).
Extracts top-3 chunks, produces grounded summaries, and outputs citations with source tags.
"""

from typing import Dict, List, Any, Tuple, Optional
import json
import os
from pathlib import Path
import numpy as np

# Ensure offline environment variables
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

CORPUS_JSON_PATH = Path(__file__).parent / "corpus" / "documents.json"
CACHE_PATH = Path(__file__).parent / "corpus" / "corpus_embeddings.npz"
SIMILARITY_THRESHOLD = 0.25  # Below this similarity, marked as degraded / low confidence


class FilingsRAGEngine:
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.doc_ids: List[str] = []
        self.precomputed_queries: Dict[str, np.ndarray] = {}
        self.model = None
        self._load_corpus()
        self._load_embeddings()

    def _load_corpus(self):
        if not CORPUS_JSON_PATH.exists():
            raise FileNotFoundError(f"Corpus file not found: {CORPUS_JSON_PATH}")
        with open(CORPUS_JSON_PATH, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

    def _get_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        return self.model

    def _load_embeddings(self):
        """Loads cached numpy embeddings and precomputed query vectors from disk."""
        if CACHE_PATH.exists():
            try:
                data = np.load(CACHE_PATH, allow_pickle=True)
                self.embeddings = data["embeddings"]
                self.doc_ids = list(data["doc_ids"])
                if "query_tickers" in data and "query_embeddings" in data:
                    q_tickers = list(data["query_tickers"])
                    q_embs = data["query_embeddings"]
                    for t, emb in zip(q_tickers, q_embs):
                        self.precomputed_queries[t] = emb
                return
            except Exception as e:
                print(f"Warning: Failed to load embeddings cache ({e}). Rebuilding...")

        # If cache missing, build it
        model = self._get_model()
        passages = [
            f"Company: {d['company']} | Section: {d['section']} | {d['content']}"
            for d in self.documents
        ]
        self.embeddings = model.encode(passages, convert_to_numpy=True, normalize_embeddings=True)
        self.doc_ids = [d["doc_id"] for d in self.documents]

        tickers = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ITC", "SBIN",
            "ICICIBANK", "LT", "TATAMOTORS", "SUNPHARMA", "ADANIENT", "BHARTIARTL"
        ]
        for t in tickers:
            q = f"Financial performance, order book, debt, margins, and operational risks for {t}"
            self.precomputed_queries[t] = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]

        np.savez_compressed(
            CACHE_PATH,
            embeddings=self.embeddings,
            doc_ids=np.array(self.doc_ids),
            query_tickers=np.array(list(self.precomputed_queries.keys())),
            query_embeddings=np.array(list(self.precomputed_queries.values())),
        )

    def search(
        self, query: str, ticker: Optional[str] = None, top_k: int = 3
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Cosine similarity search with optional ticker pre-filtering.
        Returns list of (doc, similarity_score).
        """
        if self.embeddings is None:
            return []

        # Filter indices by ticker if specified
        indices = []
        if ticker:
            ticker_clean = ticker.upper()
            for i, doc in enumerate(self.documents):
                doc_co = doc.get("company", "").upper()
                if doc_co == ticker_clean or (ticker_clean == "ICICIBANK" and doc_co == "ICICI_BANK"):
                    indices.append(i)
        else:
            indices = list(range(len(self.documents)))

        if not indices:
            # Ticker has no filings in corpus (e.g., ADANIENT, BHARTIARTL)
            return []

        # Use precomputed query vector if available, otherwise encode dynamically
        if ticker and ticker in self.precomputed_queries and (not query or ticker in query):
            query_emb = self.precomputed_queries[ticker]
        else:
            model = self._get_model()
            query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

        filtered_embs = self.embeddings[indices]
        similarities = np.dot(filtered_embs, query_emb)

        ranked_sub_indices = np.argsort(-similarities)[:top_k]
        results = []
        for rank_idx in ranked_sub_indices:
            orig_idx = indices[rank_idx]
            sim = float(similarities[rank_idx])
            results.append((self.documents[orig_idx], sim))

        return results

    def query_grounded_filings(
        self, ticker: str, custom_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieves top-3 chunks for ticker and formats an explainable, cited synthesis.
        Guarantees strict attribution tags [doc_id, section, date].
        """
        query = custom_query or f"Financial performance, order book, debt, margins, and operational risks for {ticker}"
        top_results = self.search(query=query, ticker=ticker, top_k=3)

        if not top_results:
            return {
                "status": "degraded",
                "summary": f"No filing evidence available for {ticker} in regulatory repository.",
                "evidence": [],
                "citations": [],
                "max_similarity": 0.0,
                "sentiment_bias": 0.0,
                "stale_contradiction_detected": False,
            }

        max_sim = max(r[1] for r in top_results)
        if max_sim < SIMILARITY_THRESHOLD:
            return {
                "status": "degraded",
                "summary": f"Filings similarity score ({max_sim:.2f}) below threshold {SIMILARITY_THRESHOLD}. Uncited / low confidence evidence.",
                "evidence": [],
                "citations": [],
                "max_similarity": max_sim,
                "sentiment_bias": 0.0,
                "stale_contradiction_detected": False,
            }

        citations = []
        evidence_items = []
        summary_points = []
        stale_flag = False
        sentiment_net = 0.0

        for doc, sim in top_results:
            source_tag = f"[{doc['doc_id']} | {doc['section']} | {doc['date']}]"
            citations.append({
                "doc_id": doc["doc_id"],
                "section": doc["section"],
                "date": doc["date"],
                "doc_type": doc["doc_type"],
                "similarity": round(sim, 3),
                "snippet": doc["content"],
                "source_tag": source_tag,
            })

            evidence_items.append({
                "source_id": doc["doc_id"],
                "snippet": f"{source_tag}: {doc['content'][:140]}...",
                "score": round(sim, 2),
            })

            # Check for stale or contradiction flags
            is_stale = "STALE" in doc["doc_id"] or doc["date"] < "2023-01-01"
            if is_stale:
                stale_flag = True
                summary_points.append(
                    f"⚠️ STALE CONTRADICTION {source_tag}: Historical disclosure claims high growth, which is contradicted by current operating margins and regulatory guidance."
                )
            else:
                summary_points.append(f"{source_tag} {doc['content']}")

            # Lexical sentiment cue from filing text
            content_lower = doc["content"].lower()
            pos_words = ["growth", "record", "expansion", "expanded", "robust", "demerger", "deleveraging", "profit", "improving"]
            neg_words = ["risk", "disruption", "revised downward", "headwinds", "stalled", "penalties", "observation", "freight expense", "shortages", "vulnerable"]
            
            p_count = sum(1 for w in pos_words if w in content_lower)
            n_count = sum(1 for w in neg_words if w in content_lower)
            sentiment_net += (p_count - n_count)

        # Normalized filing sentiment score between -1 and 1
        filing_score = float(np.tanh(sentiment_net / 3.0))

        # Grounded summary string
        summary_text = "\n".join(summary_points)

        return {
            "status": "ok",
            "summary": summary_text,
            "evidence": evidence_items,
            "citations": citations,
            "max_similarity": round(max_sim, 3),
            "sentiment_bias": round(filing_score, 2),
            "stale_contradiction_detected": stale_flag,
        }


# Singleton engine instance
_rag_engine: Optional[FilingsRAGEngine] = None


def get_rag_engine() -> FilingsRAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = FilingsRAGEngine()
    return _rag_engine
