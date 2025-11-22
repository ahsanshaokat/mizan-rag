"""
Mizan RAG - Chunker  (Caching Version)
======================================

Production-grade text chunker used by the Mizan RAG pipeline.

Enhancements in this version:
--------------------------------
✔ Caching support (via MizanCache)
✔ Prevents re-chunking the same file twice
✔ File SHA-1 tracking for modification detection
✔ Clean, safe, extensible architecture
✔ Word-based & character-based chunking
✔ Future token-based chunking compatible

Author: Ahsan Shaokat
"""

from typing import List, Optional
from mizan_rag.utils.cache import MizanCache


class MizanChunker:
    """
    Chunk text into overlapping segments usable by the Mizan RAG pipeline.

    Notes
    -----
    - If a `cache` is provided → chunk results are stored AND reused.
    - Word-based chunking is ideal for semantic retrieval.
    - Character-based chunking is better for messy or unstructured text.
    """

    def __init__(
        self,
        chunk_size: int = 250,
        overlap: int = 80,
        method: str = "word",           # "word" | "char" | future: "token"
        cache: Optional[MizanCache] = None
    ):
        """
        Parameters
        ----------
        chunk_size : int
            Number of tokens (words or chars) per chunk.

        overlap : int
            Number of overlapping tokens between chunks.

        method : str
            Chunking method:
                - "word": whitespace-based word splitting
                - "char": character-based sliding window

        cache : MizanCache, optional
            Enables persistent storage of chunk results.
        """

        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        if overlap < 0:
            raise ValueError("overlap must be >= 0")

        if method not in {"word", "char"}:
            raise ValueError("method must be 'word' or 'char'")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.method = method
        self.cache = cache

    # ======================================================================
    #                INTERNAL CHUNKING METHODS
    # ======================================================================

    def _chunk_words(self, text: str) -> List[str]:
        """Chunk based on whitespace-separated words."""
        words = text.split()
        chunks = []

        step = max(self.chunk_size - self.overlap, 1)
        start = 0

        while start < len(words):
            end = start + self.chunk_size
            chunks.append(" ".join(words[start:end]))
            start += step

        return chunks

    def _chunk_chars(self, text: str) -> List[str]:
        """Chunk based on raw fixed-length character windows."""
        chunks = []
        step = max(self.chunk_size - self.overlap, 1)

        start = 0
        n = len(text)

        while start < n:
            end = min(start + self.chunk_size, n)
            chunks.append(text[start:end])
            start += step

        return chunks

    # ======================================================================
    #                MAIN CHUNKING ENTRYPOINT
    # ======================================================================

    def chunk(self, text: str, source_file: Optional[str] = None) -> List[str]:
        """
        Execute chunking with optional caching.

        Parameters
        ----------
        text : str
            Raw document content.

        source_file : str, optional
            Path to original file.
            If provided AND cache is enabled → caching applies.

        Returns
        -------
        List[str]
            List of chunk strings.
        """

        if not text or not isinstance(text, str):
            return []

        # ------------------------------------------------------------------
        # 1. If caching enabled AND file provided → attempt to load cached chunks
        # ------------------------------------------------------------------
        if self.cache is not None and source_file is not None:
            cached = self.cache.get_chunks(source_file)
            if cached is not None:
                # Cached chunks are valid
                return cached

        # ------------------------------------------------------------------
        # 2. No cache hit → compute fresh chunks
        # ------------------------------------------------------------------
        if self.method == "word":
            chunks = self._chunk_words(text)
        elif self.method == "char":
            chunks = self._chunk_chars(text)
        else:
            raise ValueError(f"Unknown chunking method: {self.method}")

        # ------------------------------------------------------------------
        # 3. Store chunks (if caching is enabled + file path known)
        # ------------------------------------------------------------------
        if self.cache is not None and source_file is not None:
            self.cache.store_chunks(source_file, chunks)

        return chunks
