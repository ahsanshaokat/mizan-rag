"""
Corrected MizanRetriever — FINAL FORM
-------------------------------------

Fix:
✔ Return embedding tensor in search() so Ranker does not recompute embeddings.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
from mizan_vector.metrics import mizan_similarity
from mizan_rag.utils.cache import MizanCache


class MizanRetriever:
    def __init__(self, embed_fn, cache: Optional[MizanCache] = None):
        self.embed_fn = embed_fn
        self.cache = cache

        # index entries: (doc_id, chunk_text, emb_tensor_normalized)
        self.index: List[Tuple[str, str, torch.Tensor]] = []
        self.built = False

        # ---------------- LOAD INDEX FROM CACHE ----------------
        if self.cache:
            cached = self.cache.load_index()
            if cached:
                self.index = []
                for doc_id, text, emb_list in cached:
                    emb = torch.tensor(emb_list, dtype=torch.float32)
                    emb = F.normalize(emb, p=2, dim=0)
                    self.index.append((doc_id, text, emb))

                self.built = True
                print("📦 Loaded retriever index from cache.")


    # =====================================================================
    def add_documents(self, chunks: List[Tuple[str, str]]):
        if self.built and self.index:
            print("⚠️ Index already exists — skipping add_documents().")
            return

        print("🔧 Embedding and indexing chunks...")

        self.index = []
        for doc_id, text in chunks:

            # Try cache
            emb_list = self.cache.get_embedding(text) if self.cache else None

            if emb_list is not None:
                emb = torch.tensor(emb_list, dtype=torch.float32)
            else:
                emb = self.embed_fn(text)
                if self.cache:
                    self.cache.store_embedding(text, emb.cpu().tolist())

            # Normalize always
            emb = F.normalize(emb, p=2, dim=0)

            self.index.append((doc_id, text, emb))

        self.built = True

        # Save normalized index
        if self.cache:
            dumpable = [(d, t, e.cpu().tolist()) for d, t, e in self.index]
            self.cache.save_index(dumpable)

        print(f"✅ Indexed {len(self.index)} chunks.\n")


    # =====================================================================
    def search(self, query: str, top_k: int = 5, metric: str = "mizan", restrict_document=None):

        if not self.built:
            raise RuntimeError("Retriever index is empty — call add_documents().")

        # Embed + normalize query
        q_emb = self.embed_fn(query)
        if not isinstance(q_emb, torch.Tensor):
            q_emb = torch.tensor(q_emb, dtype=torch.float32)
        q_emb = F.normalize(q_emb, p=2, dim=0)

        scored = []

        # ---------------- LOOP THROUGH INDEX ----------------
        for doc_id, text, emb in self.index:

            if restrict_document and doc_id != restrict_document:
                continue

            # Compute similarity
            if metric == "cosine":
                score = float(F.cosine_similarity(q_emb, emb, dim=0).item())
            else:
                score = float(mizan_similarity(q_emb, emb))

            # RETURN embedding too  ← FIXED
            scored.append((score, doc_id, text, emb))

        # ---------------- SORT (descending) ----------------
        scored.sort(key=lambda x: x[0], reverse=True)

        return scored[:top_k]
