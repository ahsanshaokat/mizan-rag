"""
Mizan RAG - Ranker (Corrected Final Version)

This version FIXES:
    ✔ Uses document embeddings from retriever (NO recompute)
    ✔ Compatible with retriever returning: (score, doc_id, text, emb)
    ✔ Stable cosine + mizan + hybrid scoring
    ✔ Safe list→tensor conversion
    ✔ GPU/CPU-safe dtype/device alignment
"""

from typing import Callable, List, Tuple
import torch
import torch.nn.functional as F
from mizan_vector.metrics import mizan_similarity


class MizanRanker:
    """
    Second-pass reranker for Mizan RAG.

    Expected input from retriever:
        (orig_score, doc_id, chunk_text, chunk_embedding_tensor)
    """

    def __init__(
        self,
        embed_fn: Callable[[str], torch.Tensor],
        metric: str = "mizan",
        weight_cosine: float = 0.5,
        weight_mizan: float = 0.5,
    ):
        if metric not in ("cosine", "mizan", "hybrid"):
            raise ValueError("metric must be 'cosine', 'mizan', or 'hybrid'")

        self.embed_fn = embed_fn
        self.metric = metric
        self.w_cos = weight_cosine
        self.w_miz = weight_mizan

    # --------------------------------------------------------------
    # Safe dtype + device alignment
    # --------------------------------------------------------------
    def _ensure_tensor(self, emb, ref: torch.Tensor):
        if isinstance(emb, torch.Tensor):
            return emb.to(dtype=ref.dtype, device=ref.device)

        return torch.tensor(emb, dtype=ref.dtype, device=ref.device)

    # --------------------------------------------------------------
    # Scoring functions
    # --------------------------------------------------------------
    def _cos_score(self, q_emb, d_emb) -> float:
        return F.cosine_similarity(q_emb, d_emb, dim=0).item()

    def _miz_score(self, q_emb, d_emb) -> float:
        return mizan_similarity(q_emb, d_emb)

    def _hybrid_score(self, q_emb, d_emb) -> float:
        return (
            self.w_cos * self._cos_score(q_emb, d_emb)
            + self.w_miz * self._miz_score(q_emb, d_emb)
        )

    # --------------------------------------------------------------
    # MAIN RERANK FUNCTION
    # --------------------------------------------------------------
    def rerank(
        self,
        query: str,
        retrieved: List[Tuple[float, str, str, torch.Tensor]],
        top_k: int = 5,
    ) -> List[Tuple[float, str, str]]:
        """
        Re-rank retrieved chunks.

        INPUT format:
            (orig_score, doc_id, chunk_text, chunk_emb)

        OUTPUT format:
            (new_score, doc_id, chunk_text)
        """
        if not retrieved:
            return []

        # Compute the query embedding ONCE
        q_emb = self.embed_fn(query)
        if not isinstance(q_emb, torch.Tensor):
            q_emb = torch.tensor(q_emb, dtype=torch.float32)

        reranked = []

        for orig_score, doc_id, text, d_emb in retrieved:

            # Align types/devices
            d_emb = self._ensure_tensor(d_emb, q_emb)

            # Score with chosen metric
            if self.metric == "cosine":
                score = self._cos_score(q_emb, d_emb)

            elif self.metric == "mizan":
                score = self._miz_score(q_emb, d_emb)

            else:  # hybrid
                score = self._hybrid_score(q_emb, d_emb)

            reranked.append((score, doc_id, text))

        reranked.sort(key=lambda x: x[0], reverse=True)

        return reranked[:top_k]

    # --------------------------------------------------------------
    # BATCH RERANK
    # --------------------------------------------------------------
    def batch_rerank(
        self,
        queries: List[str],
        retrieved_batches: List[List[Tuple[float, str, str, torch.Tensor]]],
        top_k: int = 5,
    ):
        return {
            q: self.rerank(q, retrieved, top_k=top_k)
            for q, retrieved in zip(queries, retrieved_batches)
        }
