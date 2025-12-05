import os
import json
import numpy as np
import torch
import textwrap


class MizanDocumentRouter:
    """
    ENTERPRISE ROUTER V3
    --------------------
    Features:
    ✔ Multi-chunk document routing (avg + max pooling)
    ✔ Hybrid metric: 0.7*Mizan + 0.3*Cosine
    ✔ Query expansion for better routing
    ✔ Normalized embeddings everywhere
    ✔ Cache persists multi-chunk embeddings
    ✔ Fast + extremely accurate topic routing
    """

    # ============================================================
    # INIT
    # ============================================================
    def __init__(self, embed_fn, cache_dir=None, chunks_per_doc=5, chunk_chars=1500):
        self.embed = embed_fn

        # multi-chunk config
        self.chunks_per_doc = chunks_per_doc   # number of chunk embeddings per doc
        self.chunk_chars = chunk_chars         # length per summary chunk

        # storage
        self.doc_embeddings = {}  # fname -> [emb1, emb2, ...]
        self.doc_chunks = {}      # fname -> [text1, text2, ...]

        # cache
        self.cache_dir = cache_dir
        self.cache_file = (
            os.path.join(cache_dir, "router_docs_v3.json") if cache_dir else None
        )

        if self.cache_file:
            self._load_cache()

    # ============================================================
    # SAVE CACHE
    # ============================================================
    def _save_cache(self):
        if not self.cache_file:
            return

        payload = {}
        for fname, emb_list in self.doc_embeddings.items():
            payload[fname] = {
                "embeddings": [emb.tolist() for emb in emb_list],
                "chunks": self.doc_chunks.get(fname, [])
            }

        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print("💾 Saved router cache (V3).")

    # ============================================================
    # LOAD CACHE
    # ============================================================
    def _load_cache(self):
        if not os.path.exists(self.cache_file):
            return

        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for fname, info in data.items():
                # load & normalize
                emb_list = []
                for e in info["embeddings"]:
                    e = np.array(e, dtype=np.float32)
                    n = np.linalg.norm(e)
                    if n > 0:
                        e = e / n
                    emb_list.append(e)

                self.doc_embeddings[fname] = emb_list
                self.doc_chunks[fname] = info.get("chunks", [])

            print("📦 Loaded enterprise router cache (V3).")

        except Exception as e:
            print("⚠️ Router cache load failed:", e)

    # ============================================================
    # SPLIT DOCUMENT → MULTI-CHUNK SUMMARY
    # ============================================================
    def _make_summary_chunks(self, text):
        chunks = []
        length = len(text)

        step = max(1, length // self.chunks_per_doc)

        for i in range(self.chunks_per_doc):
            start = i * step
            end = min(start + self.chunk_chars, length)
            chunks.append(text[start:end])

        return chunks

    # ============================================================
    # MIZAN METRIC
    # ============================================================
    def _mizan(self, a, b, alpha=0.2):
        c = float(np.dot(a, b))
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)

        if na > 0 and nb > 0:
            ratio = min(na, nb) / max(na, nb)
        else:
            ratio = 1.0

        pos = max(0, c) * ratio
        neg = min(0, c)

        return pos + alpha * neg

    # ============================================================
    # HYBRID SIMILARITY
    # ============================================================
    def _hybrid(self, a, b):
        cos = float(np.dot(a, b))
        miz = self._mizan(a, b)
        return 0.7 * miz + 0.3 * cos

    # ============================================================
    # INDEX DOCUMENTS (MULTI-CHUNK MODE)
    # ============================================================
    def index_documents(self, docs_dir):
        for fname in os.listdir(docs_dir):
            if not fname.endswith(".txt"):
                continue

            if fname in self.doc_embeddings:
                print(f"⚡ Router cache hit: {fname}")
                continue

            path = os.path.join(docs_dir, fname)

            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    full_text = f.read()

                # ---------- create summary slices ----------
                chunks = self._make_summary_chunks(full_text)
                self.doc_chunks[fname] = chunks

                emb_list = []

                for ch in chunks:
                    emb = self.embed(ch)

                    if isinstance(emb, torch.Tensor):
                        emb = emb.detach().cpu().numpy()

                    emb = np.array(emb, dtype=np.float32)
                    n = np.linalg.norm(emb)
                    if n > 0:
                        emb = emb / n

                    emb_list.append(emb)

                self.doc_embeddings[fname] = emb_list
                print(f"📘 Document indexed (multi-chunk): {fname}")

            except Exception as e:
                print(f"❌ Error loading {fname}: {e}")

        self._save_cache()

    # ============================================================
    # QUERY EXPANSION (simple but effective)
    # ============================================================
    def _expand_query(self, q):
        add = [
            "main idea",
            "topic of text",
            "subject of document",
            "theme",
        ]
        return q + " " + " ".join(add)

    # ============================================================
    # ROUTE QUERY → BEST DOCUMENT(S)
    # ============================================================
    def route(self, question, top_k=1):
        if not self.doc_embeddings:
            print("⚠️ No router docs available.")
            return None

        # expand query a bit for better document matching
        q = self._expand_query(question)

        q_emb = self.embed(q)
        if isinstance(q_emb, torch.Tensor):
            q_emb = q_emb.detach().cpu().numpy()
        q_emb = np.array(q_emb, dtype=np.float32)
        nq = np.linalg.norm(q_emb)
        if nq > 0:
            q_emb = q_emb / nq

        # --------------- SCORE EACH DOCUMENT ----------------
        scores = []

        for fname, emb_list in self.doc_embeddings.items():
            doc_scores = [self._hybrid(q_emb, emb) for emb in emb_list]

            # Strategy:
            # max-chunk score (strong match) + mean-chunk (contextual match)
            final_score = max(doc_scores) * 0.6 + np.mean(doc_scores) * 0.4

            scores.append((final_score, fname))

        scores.sort(reverse=True)

        if top_k == 1:
            return scores[0][1]

        return [fname for _, fname in scores[:top_k]]
