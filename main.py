"""
main.py
=======

Enterprise CLI runner for the Mizan RAG system.

This example demonstrates:

    ✔ Initializing the MizanRAGPipeline with all optional parameters
    ✔ Indexing .txt files inside ./docs
    ✔ Running an interactive Q/A loop
    ✔ Supporting all LLM providers (OpenAI, Groq, Grok, HF, Local, OpenRouter)
    ✔ Optional caching & logging
    ✔ Optional cosine-vs-mizan comparison table

Use this file as:
    - CLI tool
    - API backend template
    - Desktop app launcher
    - Benchmark/test harness
"""

import os
from mizan_rag.pipeline import MizanRAGPipeline


# ==========================================================
# Optional Logging Function (You can remove or customize)
# ==========================================================
def log(msg: str):
    """Simple logger for console output."""
    print(msg)


def main():

    # ==========================================================
    # Initialize Mizan RAG Pipeline (FULL CONFIG)
    # embed_model="BAAI/bge-large-en-v1.5",
    # ==========================================================
    rag = MizanRAGPipeline(

        # ------------------------------------------------------
        # 🔵 Embeddings (HF or Custom Mizan Model)
        # ------------------------------------------------------
        embed_model="saved/mizan_encoder_v1",

        # ------------------------------------------------------
        # 🧩 Chunk Settings (Optional)
        # ------------------------------------------------------
        chunk_size=800,
        overlap=150,

        # ------------------------------------------------------
        # 🎯 Ranker Settings
        # ------------------------------------------------------
        ranker_metric="mizan",         # "cosine" | "mizan" | "hybrid"
        ranker_weights=(0.4, 0.6),     # Used ONLY if hybrid

        # ------------------------------------------------------
        # 🤖 LLM Provider + Model
        # ------------------------------------------------------
        llm_provider="groq",                       # CHANGE THIS
        summarizer_model="llama-3.3-70b-versatile",  # Groq model

        # ------------------------------------------------------
        # 🔐 API Key (Required for OpenAI/Groq/Grok/OpenRouter)
        # ------------------------------------------------------
        api_key=os.getenv("API_KEY"),

        # ------------------------------------------------------
        # 📊 Comparison Table (Optional)
        # ------------------------------------------------------
        compare=True,   # Show cosine vs mizan retrieval scores

        # ------------------------------------------------------
        # 💾 Enable Caching (Optional)
        # ------------------------------------------------------
        enable_cache=True,
        cache_path=".mizan_cache",  # folder auto-created

        # ------------------------------------------------------
        # 📝 Logging Callback (Optional)
        # ------------------------------------------------------
        on_log=log,
    )

    # ==========================================================
    # Index Documents
    # ==========================================================
    log("\n🔍 Indexing documents...\n")
    rag.index_directory("docs")
    log("\n📚 Indexing complete. Ready!\n")

    # ==========================================================
    # Interactive Q/A Loop
    # ==========================================================
    while True:
        question = input("\nAsk your question (or 'exit'): ").strip()

        if question.lower() == "exit":
            log("\nGoodbye! 👋")
            break

        result = rag.query(question, top_k_retrieve=25, top_k_rerank=10)

        # ------------------------------------------------------
        # 🔎 Top Reranked Retrieval Results
        # ------------------------------------------------------
        print("\n--- 🔎 TOP RERANKED RESULTS ---")
        for score, doc, text in result["final_ranked"]:
            print(f"\n[{doc}] | score={score:.4f}")
            print(text[:300], "...")

        # ------------------------------------------------------
        # 📊 Cosine vs Mizan Comparison (Optional)
        # ------------------------------------------------------
        if result["comparison"]:
            print("\n--- 📊 COSINE vs MIZAN ---")
            for row in result["comparison"]:
                print(
                    f"#{row['rank']:02d} | "
                    f"COS {row['cosine_score']:.4f} ({row['cosine_doc']})  ||  "
                    f"MIZ {row['mizan_score']:.4f} ({row['mizan_doc']})"
                )

        # ------------------------------------------------------
        # 🧠 Final Answer
        # ------------------------------------------------------
        print("\n--- 🧠 FINAL ANSWER ---")
        print(result["answer"])
        print("\n" + "-" * 60)


if __name__ == "__main__":
    main()
