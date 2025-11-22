"""
Mizan RAG - Public API
======================

This file defines what the user imports when writing:

    from mizan_rag import MizanRAGPipeline, MizanChunker, MizanRetriever, MizanRanker, MizanSummarizer

Only the high-level components are exposed.
"""

# ---------------------------------------------------------
# Core Components
# ---------------------------------------------------------

from .chunker import MizanChunker
from .retriever import MizanRetriever
from .ranker import MizanRanker
from .summarizer import MizanSummarizer
from .pipeline import MizanRAGPipeline

# Utility (optional)
from .loaders import load_text_file

__all__ = [
    "MizanChunker",
    "MizanRetriever",
    "MizanRanker",
    "MizanSummarizer",
    "MizanRAGPipeline",
    "load_text_file",
]
