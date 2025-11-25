import os
import json
import numpy as np

class MizanDocumentRouter:
    """
    Lightweight document-level router with caching.
    Determines which document(s) are relevant for a given question
    BEFORE chunk-level retrieval.
    """

    def __init__(self, embed_fn, cache_dir=None):
        self.embed = embed_fn
        self.doc_embeddings = {}   # filename -> numpy array
        self.doc_texts = {}        # filename -> summary text

        self.cache_dir = cache_dir
        self.cache_file = None
        if cache_dir:
            self.cache_file = os.path.join(cache_dir, "router_docs.json")

            # Try loading router cache
            self._load_cache()

    # ------------------------------------------------------------
    # SAVE to JSON
    # ------------------------------------------------------------
    def _save_cache(self):
        if not self.cache_file:
            return

        data = {}
        for fname, emb in self.doc_embeddings.items():
            data[fname] = {
                "embedding": emb.tolist(),
                "summary": self.doc_texts.get(fname, "")
            }

        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print("💾 Saved router embeddings cache.")

    # ------------------------------------------------------------
    # LOAD from JSON
    # ------------------------------------------------------------
    def _load_cache(self):
        if not self.cache_file or not os.path.exists(self.cache_file):
            return

        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for fname, item in data.items():
                emb = np.array(item["embedding"], dtype=np.float32)
                self.doc_embeddings[fname] = emb
                self.doc_texts[fname] = item.get("summary", "")

            print("📦 Loaded document router cache.")

        except Exception as e:
            print("⚠️ Failed to load router cache:", e)

    # ------------------------------------------------------------
    # INDEX DOCUMENTS
    # ------------------------------------------------------------
    def index_documents(self, docs_dir, summary_chars=5000):
        """
        Pre-compute embeddings for each document.
        Skip if already cached.
        """
        for fname in os.listdir(docs_dir):
            if not fname.endswith(".txt"):
                continue

            # Already in cache? Skip reprocessing.
            if fname in self.doc_embeddings:
                print(f"⚡ Using cached router embedding: {fname}")
                continue

            path = os.path.join(docs_dir, fname)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    full_text = f.read()

                summary = full_text[:summary_chars]

                emb = np.array(self.embed(summary), dtype=np.float32)
                self.doc_embeddings[fname] = emb
                self.doc_texts[fname] = summary

                print(f"📘 Document indexed for routing: {fname}")

            except Exception as e:
                print(f"❌ Failed to load {fname}: {e}")

        # Save updated router cache
        self._save_cache()

    # ------------------------------------------------------------
    # ROUTE QUERY → BEST FILE
    # ------------------------------------------------------------
    def route(self, question, top_k=1):
        q_emb = np.array(self.embed(question), dtype=np.float32)

        scores = []
        for fname, emb in self.doc_embeddings.items():
            sim = float(
                np.dot(q_emb, emb) /
                (np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-8)
            )
            scores.append((sim, fname))

        scores.sort(reverse=True)

        if not scores:
            return None

        # Return top 1 filename
        if top_k == 1:
            return scores[0][1]

        return [fname for _, fname in scores[:top_k]]
