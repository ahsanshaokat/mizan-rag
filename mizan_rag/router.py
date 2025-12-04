import os
import json
import numpy as np
import torch
import torch.nn.functional as F


class MizanDocumentRouter:
    """
    Document-level router with proper normalization and caching.

    FIXES ADDED:
    ------------
    ✔ Normalize embeddings (critical!)
    ✔ Normalize query embedding
    ✔ Safer cosine similarity
    ✔ Smaller summary (improves separation)
    ✔ Proper tensor → numpy handling
    ✔ Stable sorting
    """

    def __init__(self, embed_fn, cache_dir=None):
        self.embed = embed_fn
        self.doc_embeddings = {}   # filename -> numpy array (normalized)
        self.doc_texts = {}        # filename -> summary text

        self.cache_dir = cache_dir
        self.cache_file = (
            os.path.join(cache_dir, "router_docs.json") if cache_dir else None
        )

        # Try loading cached embeddings
        if self.cache_file:
            self._load_cache()

    # ------------------------------------------------------------
    # SAVE CACHE
    # ------------------------------------------------------------
    def _save_cache(self):
        if not self.cache_file:
            return

        payload = {}
        for fname, emb in self.doc_embeddings.items():
            payload[fname] = {
                "embedding": emb.tolist(),
                "summary": self.doc_texts.get(fname, "")
            }

        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print("💾 Saved router embeddings cache.")

    # ------------------------------------------------------------
    # LOAD CACHE
    # ------------------------------------------------------------
    def _load_cache(self):
        if not self.cache_file or not os.path.exists(self.cache_file):
            return

        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for fname, info in data.items():
                emb = np.array(info["embedding"], dtype=np.float32)

                # FIX: normalize cached embeddings
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm

                self.doc_embeddings[fname] = emb
                self.doc_texts[fname] = info.get("summary", "")

            print("📦 Loaded document router cache.")

        except Exception as e:
            print("⚠️ Failed to load router cache:", e)

    # ------------------------------------------------------------
    # INDEX DOCUMENT SUMMARIES
    # ------------------------------------------------------------
    def index_documents(self, docs_dir, summary_chars=2000):
        """
        Create one document-level embedding per file.
        """

        for fname in os.listdir(docs_dir):
            if not fname.endswith(".txt"):
                continue

            # Already cached? Skip
            if fname in self.doc_embeddings:
                print(f"⚡ Using cached router embedding: {fname}")
                continue

            path = os.path.join(docs_dir, fname)

            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    full_text = f.read()

                summary = full_text[:summary_chars]

                # ---- Embed summary using embedder ----
                emb = self.embed(summary)

                # Convert to numpy
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu().numpy()

                emb = np.array(emb, dtype=np.float32)

                # ---- CRITICAL: Normalize ----
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm

                self.doc_embeddings[fname] = emb
                self.doc_texts[fname] = summary

                print(f"📘 Document indexed for routing: {fname}")

            except Exception as e:
                print(f"❌ Failed to load {fname}: {e}")

        self._save_cache()

    # ------------------------------------------------------------
    # ROUTE QUERY → DOCUMENT NAME
    # ------------------------------------------------------------
    def route(self, question, top_k=1):
        """
        Return the best document(s) for the question.
        """

        q_emb = self.embed(question)

        # Convert to numpy
        if isinstance(q_emb, torch.Tensor):
            q_emb = q_emb.detach().cpu().numpy()

        q_emb = np.array(q_emb, dtype=np.float32)

        # ---- Normalize query embedding ----
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm

        scores = []

        # ---- Compute cosine similarity ----
        for fname, emb in self.doc_embeddings.items():
            sim = float(np.dot(q_emb, emb))
            scores.append((sim, fname))

        scores.sort(reverse=True)

        if not scores:
            return None

        if top_k == 1:
            return scores[0][1]

        return [fname for _, fname in scores[:top_k]]
