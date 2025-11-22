"""
Mizan RAG - Cache System
========================

Enterprise-ready caching engine used across:

    - Chunker
    - Embedder
    - Retriever
    - Summarizer
    - Pipeline Coordinator

Caching Goals:
    ✔ Speed up RAG dramatically
    ✔ Avoid recomputing embeddings
    ✔ Avoid repeated LLM charges
    ✔ Prevent re-indexing unchanged files
    ✔ Support large-scale document corpora
    ✔ Provide a clean API for all modules

IMPORTANT:
This version includes a compatibility layer:
    cache.get(text)
    cache.set(text, embedding)

So it works natively with mizan-embedder.
"""

import os
import json
import pickle
import hashlib
import torch
from typing import Any, Dict, List, Optional


# =====================================================================
#                         UTILITY FUNCTIONS
# =====================================================================

def sha1(text: str) -> str:
    """Return SHA-1 hash for text."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def file_sha1(path: str) -> str:
    """Return SHA-1 hash for the contents of a file."""
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def ensure_dir(path: str):
    """Create directory if missing."""
    os.makedirs(path, exist_ok=True)


# =====================================================================
#                         CACHE FILE BACKENDS
# =====================================================================

def save_pickle(path: str, obj: Any):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================================
#                         MAIN CACHE CLASS
# =====================================================================

class MizanCache:
    """
    Unified caching system for the Mizan RAG ecosystem.

    Caches:
        - Embeddings
        - Chunks
        - Retriever index
        - LLM responses
        - File metadata (hash + modified time)

    Parameters
    ----------
    cache_dir : str
        Directory where all caches live.

    enabled : bool
        If False → cache is completely bypassed.
    """

    def __init__(self, cache_dir: str = ".mizan_cache", enabled: bool = True):
        self.enabled = enabled
        self.cache_dir = cache_dir

        self.paths = {
            "embeddings": os.path.join(cache_dir, "embeddings.jsonl"),
            "chunks": os.path.join(cache_dir, "chunks.json"),
            "index": os.path.join(cache_dir, "index.pkl"),
            "llm": os.path.join(cache_dir, "llm.jsonl"),
            "meta": os.path.join(cache_dir, "file_meta.json"),
        }

        ensure_dir(cache_dir)

        # Cache structures in memory
        self.embedding_cache: Dict[str, List[float]] = {}
        self.chunk_cache: Dict[str, List[str]] = {}
        self.llm_cache: Dict[str, str] = {}
        self.file_meta: Dict[str, Dict] = {}

        self._load_all()

    # =================================================================
    #                   LOADING / SAVING WHOLE CACHE
    # =================================================================

    def _load_all(self):
        """Load all cache components from disk."""
        if not self.enabled:
            return

        # ------------------- Embeddings --------------------
        if os.path.exists(self.paths["embeddings"]):
            with open(self.paths["embeddings"], "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line.strip())
                    self.embedding_cache[obj["text_hash"]] = obj["embedding"]

        # ------------------- Chunks ------------------------
        if os.path.exists(self.paths["chunks"]):
            self.chunk_cache = load_json(self.paths["chunks"])

        # ------------------- LLM responses -----------------
        if os.path.exists(self.paths["llm"]):
            with open(self.paths["llm"], "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line.strip())
                    self.llm_cache[obj["prompt_hash"]] = obj["response"]

        # ------------------- Metadata ----------------------
        if os.path.exists(self.paths["meta"]):
            self.file_meta = load_json(self.paths["meta"])

    def save_chunks(self):
        save_json(self.paths["chunks"], self.chunk_cache)

    def save_file_meta(self):
        save_json(self.paths["meta"], self.file_meta)

    def save_index(self, index_obj: Any):
        save_pickle(self.paths["index"], index_obj)

    def load_index(self) -> Optional[Any]:
        if os.path.exists(self.paths["index"]):
            return load_pickle(self.paths["index"])
        return None

    # =================================================================
    #                           EMBEDDING CACHE
    # =================================================================

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Return cached embedding or None."""
        if not self.enabled:
            return None
        return self.embedding_cache.get(sha1(text))

    # ============================================================
    #       SAFE STORE EMBEDDING (ensures JSON serializable)
    # ============================================================

    def store_embedding(self, text: str, embedding):
        if not self.enabled:
            return

        # ensure list
        if hasattr(embedding, "cpu"):
            embedding = embedding.cpu().tolist()
        elif isinstance(embedding, torch.Tensor):
            embedding = embedding.tolist()

        h = sha1(text)
        self.embedding_cache[h] = embedding

        with open(self.paths["embeddings"], "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "text_hash": h,
                "embedding": embedding
            }) + "\n")

    # =================================================================
    #                           CHUNK CACHE
    # =================================================================

    def get_chunks(self, filepath: str) -> Optional[List[str]]:
        """Return chunks if the file has not changed."""
        if not self.enabled:
            return None

        file_hash = file_sha1(filepath)
        meta = self.file_meta.get(filepath)

        if not meta:
            return None

        if meta["sha1"] != file_hash:
            return None

        return self.chunk_cache.get(filepath)

    def store_chunks(self, filepath: str, chunks: List[str]):
        """Store file chunks and update metadata."""
        if not self.enabled:
            return

        file_hash = file_sha1(filepath)
        modified = os.path.getmtime(filepath)

        self.file_meta[filepath] = {"sha1": file_hash, "mtime": modified}
        self.chunk_cache[filepath] = chunks

        self.save_chunks()
        self.save_file_meta()

    # =================================================================
    #                       LLM CACHE
    # =================================================================

    def get_llm(self, prompt: str) -> Optional[str]:
        """Return cached LLM response if exists."""
        if not self.enabled:
            return None
        return self.llm_cache.get(sha1(prompt))

    def store_llm(self, prompt: str, response: str):
        """Store LLM response."""
        if not self.enabled:
            return
        h = sha1(prompt)
        self.llm_cache[h] = response

        with open(self.paths["llm"], "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "prompt_hash": h,
                "response": response
            }) + "\n")

    # =================================================================
    #               COMPATIBILITY LAYER (IMPORTANT)
    # =================================================================

    def get(self, text: str):
        """
        Compatibility alias for mizan-embedder.
        Equivalent to get_embedding().
        """
        return self.get_embedding(text)

    def set(self, text: str, embedding):
        """
        Compatibility alias for mizan-embedder.
        Equivalent to store_embedding().
        Converts tensor → list automatically.
        """
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        self.store_embedding(text, embedding)

    # =================================================================
    #               DICT-LIKE COMPATIBILITY LAYER
    # =================================================================

    def __getitem__(self, text: str):
        """Allows: cache[text]"""
        return self.get_embedding(text)

    def __setitem__(self, text: str, embedding):
        """
        Allows: cache[text] = embedding
        Automatically converts tensor → list.
        """
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        self.store_embedding(text, embedding)

    # ============================================================
    #   COMPATIBILITY LAYER (Fix for Summarizer calling new API)
    # ============================================================

    # old name: get_llm()
    def get_response(self, prompt: str) -> Optional[str]:
        """Compatibility alias for get_llm()"""
        return self.get_llm(prompt)

    # old name: store_llm()
    def store_response(self, prompt: str, response: str):
        """Compatibility alias for store_llm()"""
        self.store_llm(prompt, response)

