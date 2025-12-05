"""
Mizan RAG Pipeline
==================

Enterprise-grade Retrieval-Augmented Generation pipeline.

Components:
    - MizanChunker          → text chunking
    - MizanRetriever        → fast similarity search (cosine / mizan)
    - MizanRanker           → slow accurate re-ranking (cosine / mizan / hybrid)
    - MizanSummarizer       → multi-provider LLM engine
    - MizanTextEncoderWrapper (mizan-embedder)
    - MizanCache (optional)
    - Logging hooks (optional)

Supported LLM providers:
    OpenAI     | GPT-4.x, GPT-4o, GPT-o
    Groq       | Llama-3.1 SpecDec models
    Grok       | xAI
    OpenRouter | 100+ models
    HuggingFace| Chat + text-generation
    Local HF   | CPU/GPU inference

Pipeline Flow:
    1. Chunk       → MizanChunker
    2. Embed       → MizanTextEncoderWrapper (mizan-embedder)
    3. Retrieve    → MizanRetriever
    4. Rerank      → MizanRanker
    5. Summarize   → MizanSummarizer

Author: Ahsan Shaokat
Commercial Status: READY
"""

import os
from typing import Dict, List, Tuple, Optional, Callable

from mizan_embedder import MizanTextEncoderWrapper
from mizan_rag.chunker import MizanChunker
from mizan_rag.router import MizanDocumentRouter
from mizan_rag.retriever import MizanRetriever
from mizan_rag.ranker import MizanRanker
from mizan_rag.summarizer import MizanSummarizer
from mizan_rag.utils.cache import MizanCache



# =====================================================================
#                        RAG PIPELINE CLASS
# =====================================================================

class MizanRAGPipeline:
    """
    Full Mizan RAG Orchestration Layer.

    ALL PARAMETERS (with sane defaults)
    -----------------------------------
    embed_model : str
        HF model or path to custom trained Mizan embedding model.

    chunk_size : int
        Word-based chunk size (default: 250 words).

    overlap : int
        Word overlap between consecutive chunks (default: 80).

    ranker_metric : str
        "cosine"  — geometric cosine similarity
        "mizan"   — balanced mizan similarity (recommended)
        "hybrid"  — weighted fusion

    ranker_weights : (float, float)
        Only used when ranker_metric == “hybrid”.

    llm_provider : Optional[str]
        "openai" | "groq" | "grok" | "openrouter" | "hf" | "local"
        If None → auto-detected.

    summarizer_model : str
        LLM model name.

    api_key : Optional[str]
        API key for LLM providers (OpenAI/Groq/OpenRouter).
        HF/local do not require a key.

    compare : bool
        True → provide cosine vs mizan retrieval comparison.

    enable_cache : bool
        Enable LLM & embedding caching.

    cache_path : Optional[str]
        Directory to store caching JSON files.

    on_log : Optional[Callable]
        Callback for logging messages (UI or CLI).

    """

    # =================================================================
    # INIT
    # =================================================================
    def __init__(
        self,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 250,
        overlap: int = 80,
        ranker_metric: str = "mizan",
        ranker_weights: Tuple[float, float] = (0.5, 0.5),
        llm_provider: Optional[str] = None,
        summarizer_model: str = "llama-3.1-70b-specdec",
        api_key: Optional[str] = None,
        compare: bool = False,
        enable_cache: bool = True,
        cache_path: Optional[str] = None,
        on_log: Optional[Callable[[str], None]] = print,
    ):

        self.on_log = on_log or (lambda x: None)

        # ---------------- CACHE ------------------
        self.cache = None
        if enable_cache:
            self.cache = MizanCache(cache_dir=cache_path)

        # ---------------- Embedding ------------------
        self.on_log(f"🔵 Loading embedding model: {embed_model}")

        self.embedder = MizanTextEncoderWrapper(
            backbone_name=embed_model,
            emb_dim=384,
            pooling="mean",
            normalize=True,
            cache=self.cache
        )
        self.embed = self.embedder.encode_one

        # ---------------- Chunker -------------------
        self.chunker = MizanChunker(
            chunk_size=chunk_size,
            overlap=overlap,
            method="word"
        )

        # ---------------- Router (Document Level) -----------------
        self.router = MizanDocumentRouter(
            embed_fn=self.embed, 
            cache_dir=cache_path,
            chunks_per_doc=5,     # NEW
            chunk_chars=4500      # NEW
            )

        # ---------------- Retriever -----------------
        self.retriever = MizanRetriever(
            embed_fn=self.embed,
            cache=self.cache
        )

        # ---------------- Ranker -------------------
        self.ranker = MizanRanker(
            embed_fn=self.embed,
            metric=ranker_metric,
            weight_cosine=ranker_weights[0],
            weight_mizan=ranker_weights[1]
        )

        # ---------------- Summarizer ----------------
        self.summarizer = MizanSummarizer(
            provider=llm_provider,
            model=summarizer_model,
            api_key=api_key,
            cache=self.cache
        )

        self.compare = compare
        self.on_log("✅ Mizan RAG Pipeline initialized.\n")


    # =================================================================
    # INDEX DOCUMENTS
    # =================================================================
    def index_directory(self, docs_dir: str):
        """
        Load all .txt files → chunk → embed → store embeddings in index.

        Includes:
            - Index guard
            - Error handling
            - Logging
        """

        if self.retriever.built:
            self.on_log("⚠️ Index already built — skipping.\n")
            return

        chunks_to_add = []
        total_files = 0
        total_chunks = 0

        for filename in os.listdir(docs_dir):
            if not filename.endswith(".txt"):
                continue

            total_files += 1
            path = os.path.join(docs_dir, filename)

            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception as e:
                self.on_log(f"❌ Failed: {filename} → {e}")
                continue

            chunks = self.chunker.chunk(text)

            for ch in chunks:
                chunks_to_add.append((filename, ch))

            total_chunks += len(chunks)

        self.on_log(f"📄 Files loaded:  {total_files}")
        self.on_log(f"🧩 Chunks ready:  {total_chunks}")
        self.on_log("📥 Embedding & indexing...")

        self.retriever.add_documents(chunks_to_add)

        # ------------ Document Router Indexing ------------
        self.on_log("🔵 Indexing documents for routing...")
        self.router.index_documents(docs_dir)
        self.on_log("✅ Document router ready.\n")

        self.on_log("✅ Indexing complete.\n")



    # =================================================================
    # QUERY PIPELINE
    # =================================================================
    def query(
        self,
        question: str,
        top_k_retrieve: int = 8,
        top_k_rerank: int = 5,
    ) -> Dict:

        # ----------------------------------------------
        # 1) DOCUMENT ROUTING
        # ----------------------------------------------
        best_doc = None
        
        if hasattr(self, "router") and self.router is not None:
            routed = self.router.route(question, top_k=3)

            # routed may be:
            #   - "Alice_in_Wonderland.txt"
            #   - ["Alice_in_Wonderland.txt"]
            if isinstance(routed, list):
                if len(routed) > 0:
                    best_doc = routed[0]
            else:
                best_doc = routed

        # Logging
        self.on_log(f"📘 Routed to document: {best_doc}")

        # ----------------------------------------------
        # 2) RETRIEVAL (document-filtered or global)
        # ----------------------------------------------
        if best_doc:
            cosine_results = self.retriever.search(
                question, top_k=top_k_retrieve,
                metric="cosine", restrict_document=best_doc
            )
            mizan_results = self.retriever.search(
                question, top_k=top_k_retrieve,
                metric="mizan", restrict_document=best_doc
            )
        else:
            cosine_results = self.retriever.search(
                question, top_k=top_k_retrieve, metric="cosine"
            )
            mizan_results = self.retriever.search(
                question, top_k=top_k_retrieve, metric="mizan"
            )

        # ----------------------------------------------
        # 3) RERANKING
        # ----------------------------------------------
        final_ranked = self.ranker.rerank(
            question,
            retrieved=mizan_results,
            top_k=top_k_rerank
        )
        final_chunks = [text for score, doc_id, text in final_ranked]

        # ----------------------------------------------
        # 4) SUMMARY / DIRECT RETURN
        # ----------------------------------------------
        answer = self.summarizer.answer_question(question, final_chunks)
        # answer = final_chunks

        # ----------------------------------------------
        # 5) OPTIONAL COMPARISON
        # ----------------------------------------------
        comparison = None
        if self.compare:
            comparison = []
            for i in range(min(len(cosine_results), len(mizan_results))):
                comparison.append({
                    "rank": i + 1,
                    "cosine_doc": cosine_results[i][1],
                    "cosine_score": cosine_results[i][0],
                    "mizan_doc": mizan_results[i][1],
                    "mizan_score": mizan_results[i][0],
                })

        return {
            "question": question,
            "answer": answer,
            "final_ranked": final_ranked,
            "comparison": comparison,
        }


