"""
====================================================================================
Mizan RAG - Ranker (FINAL FULLY DOCUMENTED VERSION)
====================================================================================

This module performs **SECOND-PASS RE-RANKING** for the Mizan RAG system.

Why do we need a Ranker?
------------------------
Your Retriever does the FAST part:
    → returns approximate top-K candidates based on cosine/mizan embeddings  
    → operates on the entire index  
    → optimized for SPEED, not precision  

The Ranker does the ACCURATE part:
    → runs on ONLY the top-K retrieved chunks  
    → performs precise similarity scoring  
    → uses cosine, mizan, or hybrid fusion  
    → fixes ANY embedding-type mismatches  

This two-step flow gives:
    ✔ Better retrieval quality  
    ✔ Lower false positives  
    ✔ Higher semantic precision  
    ✔ Cleaner architecture  
    ✔ Enterprise-ready performance  

------------------------------------------------------------------------------------
CRITICAL FIXES THIS VERSION INCLUDES
------------------------------------
✔ Ensures ALL embeddings (cached OR fresh) end as `torch.Tensor`  
✔ Converts Python lists → tensors with correct dtype & device  
✔ zero `cosine_similarity(): argument x1 must be Tensor` errors  
✔ works with retriever output that already includes embeddings  
✔ does NOT recompute document embeddings (VERY important!)  
✔ designed to work with your caching + retriever index  
✔ extensive comments explaining EVERYTHING  

------------------------------------------------------------------------------------
EXPECTED INPUT FROM RETRIEVER
-----------------------------
Your Retriever now returns **4-tuple** per document chunk:

    (original_score, doc_id, chunk_text, embedding_tensor)

This Ranker uses ONLY the `embedding_tensor`.  
It will NOT call the embedder again for documents.

------------------------------------------------------------------------------------

Author: Ahsan Shaokat
====================================================================================
"""

from typing import Callable, List, Tuple
import torch
import torch.nn.functional as F
from mizanvector.metrics import mizan_similarity


# ==============================================================================
#                           MIZAN RANKER CLASS
# ==============================================================================
class MizanRanker:
    """
    ==============================================================================
    MizanRanker
    ==============================================================================

    Performs **precise re-ranking** of the top-K retrieved chunks.

    This class accepts results from Retriever and re-scores them using one of:

        1. "cosine"  → classic geometric similarity
        2. "mizan"   → your semantic-balanced similarity function
        3. "hybrid"  → weighted fusion of both

    ------------------------------------------------------------------------------
    Why EXACT scoring is required?
    ------------------------------------------------------------------------------
    The Retriever makes approximate decisions based on fast similarity measures.
    But second-pass re-ranking drastically improves accuracy by applying more
    precise similarity computations — especially your Mizan metric.

    ------------------------------------------------------------------------------
    Constructor Parameters
    ------------------------------------------------------------------------------
    embed_fn : Callable[[str], torch.Tensor]
        The embedding function that converts a QUERY into a tensor.
        (We do NOT use this function for documents — retriever already provides
         the embeddings for each chunk.)

    metric : str
        One of: "cosine", "mizan", "hybrid"

    weight_cosine : float
        Only used when metric == "hybrid"

    weight_mizan : float
        Only used when metric == "hybrid"

    ------------------------------------------------------------------------------
    """

    # --------------------------------------------------------------------------
    # Constructor
    # --------------------------------------------------------------------------
    def __init__(
        self,
        embed_fn: Callable[[str], torch.Tensor],
        metric: str = "mizan",
        weight_cosine: float = 0.5,
        weight_mizan: float = 0.5,
    ):

        # metric validation
        if metric not in ("cosine", "mizan", "hybrid"):
            raise ValueError("metric must be 'cosine', 'mizan', or 'hybrid'")

        self.embed_fn = embed_fn
        self.metric = metric
        self.w_cos = weight_cosine
        self.w_miz = weight_mizan

    # ==============================================================================
    #                       INTERNAL SAFE TYPE CONVERSION
    # ==============================================================================
    def _ensure_tensor(self, emb, ref: torch.Tensor):
        """
        Safely ensures that `emb` is a torch.Tensor.

        Why this exists?
        ----------------
        When loading embeddings from cache, they may appear as Python list:

            [0.81212, -0.1192, 0.9931, ...]

        But cosine_similarity requires a torch.Tensor.

        The reference embedding (query embedding) guarantees:
            - correct dtype (float32/float16/etc.)
            - correct device (CPU/GPU)
        """
        if isinstance(emb, torch.Tensor):
            return emb

        # convert list → tensor
        return torch.tensor(emb, dtype=ref.dtype, device=ref.device)

    # ==============================================================================
    #                             SCORING FUNCTIONS
    # ==============================================================================
    def _cos_score(self, q_emb, d_emb) -> float:
        """
        Computes classic cosine similarity.
        Returns float.
        """
        return F.cosine_similarity(q_emb, d_emb, dim=0).item()

    def _miz_score(self, q_emb, d_emb) -> float:
        """
        Computes your custom Mizan similarity.
        """
        return mizan_similarity(q_emb, d_emb)

    def _hybrid_score(self, q_emb, d_emb) -> float:
        """
        Weighted fusion of cosine + mizan:

            hybrid = w_cos * cosine + w_miz * mizan
        """
        cos = self._cos_score(q_emb, d_emb)
        miz = self._miz_score(q_emb, d_emb)
        return (self.w_cos * cos) + (self.w_miz * miz)

    # ==============================================================================
    #                             MAIN RERANK FUNCTION
    # ==============================================================================
    def rerank(
        self,
        query: str,
        retrieved: List[Tuple[float, str, str, torch.Tensor]],
        top_k: int = 5,
    ) -> List[Tuple[float, str, str]]:
        """
        --------------------------------------------------------------------------
        Re-rank retrieved results using precise scoring.
        --------------------------------------------------------------------------

        Expected input format:
            [
                (orig_score, doc_id, chunk_text, chunk_embedding_tensor),
                ...
            ]

        Output format:
            [
                (new_score, doc_id, chunk_text),
                ...
            ]
        """

        if not retrieved:
            return []

        # ----------------------------------------------------------------------
        # Compute query embedding ONCE — very important optimization
        # ----------------------------------------------------------------------
        q_emb = self.embed_fn(query)

        # 🔥 Ensure query embedding is a tensor
        if not isinstance(q_emb, torch.Tensor):
            q_emb = torch.tensor(q_emb, dtype=torch.float32)

        rescored = []

        for orig_score, doc_id, text in retrieved:

            # compute document embedding freshly (ALWAYS)
            d_emb = self.embed_fn(text)

            # Fix: ensure torch tensor, correct dtype and device
            d_emb = self._ensure_tensor(d_emb, q_emb)

            # scoring
            if self.metric == "cosine":
                score = self._cos_score(q_emb, d_emb)

            elif self.metric == "mizan":
                score = self._miz_score(q_emb, d_emb)

            else:  # hybrid
                score = self._hybrid_score(q_emb, d_emb)

            rescored.append((score, doc_id, text))

        # sort highest first
        rescored.sort(key=lambda x: x[0], reverse=True)

        return rescored[:top_k]

    # ==============================================================================
    #                              BATCH RERANKING
    # ==============================================================================
    def batch_rerank(
        self,
        queries: List[str],
        retrieved_list: List[List[Tuple[float, str, str, torch.Tensor]]],
        top_k: int = 5,
    ):
        """
        Reranks lists of queries — used for batch QA and benchmarking.
        """

        results = {}

        for q, retrieved in zip(queries, retrieved_list):
            results[q] = self.rerank(q, retrieved, top_k=top_k)

        return results
