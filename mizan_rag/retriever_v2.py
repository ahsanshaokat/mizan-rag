"""
Mizan RAG - Retriever (Caching Version)
=======================================

First-pass dense retriever.

Features
--------
✔ Cosine + Mizan similarity
✔ Embedding caching (Tensor → list → Tensor)
✔ Full index caching
✔ Prevents double indexing
✔ Converts cached embeddings (lists) → tensors
✔ Production-ready, stable

Author: Ahsan Shaokat
"""

from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
from mizan_vector.metrics import mizan_similarity
from mizan_rag.utils.cache import MizanCache


class MizanRetriever:
    """
    Fast first-pass retriever.

    Stores tuples:
        (doc_id, chunk_text, embedding_tensor)

    Embeddings may come from:
        - freshly generated tensors
        - cached lists (converted back to tensors)
    """

    def __init__(self, embed_fn, cache: Optional[MizanCache] = None):
        self.embed_fn = embed_fn
        self.cache = cache

        # always tensors
        self.index: List[Tuple[str, str, torch.Tensor]] = []
        self.built = False

        # --------------------------------------------
        # Load cached index (convert list → tensor)
        # --------------------------------------------
        if self.cache:
            cached_index = self.cache.load_index()
            if cached_index:
                self.index = []
                for doc_id, text, emb_list in cached_index:
                    emb_tensor = torch.tensor(emb_list, dtype=torch.float32)
                    self.index.append((doc_id, text, emb_tensor))

                self.built = True
                print("📦 Loaded retriever index from cache.")


    # =====================================================================
    #                    ADD DOCUMENT CHUNKS
    # =====================================================================
    def add_documents(self, chunks: List[Tuple[str, str]]):
        if self.built and self.index:
            print("⚠️ Index already exists — skipping add_documents().")
            return

        print("🔧 Embedding and indexing chunks...")

        new_index = []

        for doc_id, text in chunks:

            # --------------------------------------------------------
            # 1. Try to fetch embedding from cache (list)
            # --------------------------------------------------------
            emb_list = self.cache.get_embedding(text) if self.cache else None

            if emb_list is not None:
                emb = torch.tensor(emb_list, dtype=torch.float32)

            else:
                # --------------------------------------------------------
                # 2. Compute new embedding
                # --------------------------------------------------------
                emb = self.embed_fn(text)

                # Save to cache as list
                if self.cache:
                    self.cache.store_embedding(text, emb.cpu().tolist())

            new_index.append((doc_id, text, emb))

        self.index = new_index
        self.built = True

        # --------------------------------------------------------
        # Save index back into cache (tensor → list)
        # --------------------------------------------------------
        if self.cache:
            dumpable = [(doc, txt, e.cpu().tolist()) for doc, txt, e in self.index]
            self.cache.save_index(dumpable)

        print(f"✅ Indexed {len(self.index)} chunks.\n")


    # =====================================================================
    #                          SEARCH
    # =====================================================================
    def search(self, query: str, top_k: int = 5, metric: str = "mizan", restrict_document=None):
        """
        Vector search with optional document restriction.

        Args:
            query : str
                User query.
            top_k : int
                Number of results to return.
            metric : "cosine" | "mizan"
            restrict_document : str | None
                If provided, only search inside this specific document ID.
        """

        if not self.built:
            raise RuntimeError("Retriever index is empty — call add_documents().")

        if metric not in ("cosine", "mizan"):
            raise ValueError("metric must be 'cosine' or 'mizan'")

        # -------------------- Embed Query --------------------
        q_emb = self.embed_fn(query)

        # Force tensor
        if not isinstance(q_emb, torch.Tensor):
            q_emb = torch.tensor(q_emb, dtype=torch.float32)

        scored = []

        # -------------------- Loop Over Chunks --------------------
        for doc_id, text, emb in self.index:

            # Restrict search to only one document
            if restrict_document and doc_id != restrict_document:
                continue

            # Ensure tensor
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb, dtype=q_emb.dtype, device=q_emb.device)

            # Compute similarity
            if metric == "cosine":
                score = F.cosine_similarity(q_emb, emb, dim=0).item()
            else:
                score = mizan_similarity(q_emb, emb)

            scored.append((score, doc_id, text))

        # -------------------- Sort & Return --------------------
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]




    # =====================================================================
    #                     BATCH SEARCH
    # =====================================================================
    def batch_search(self, queries: List[str], top_k: int = 5, metric: str = "mizan"):
        return {
            q: self.search(q, top_k=top_k, metric=metric)
            for q in queries
        }
