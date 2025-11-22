"""
Real-time RAG pipeline using YOUR trained MizanTextEncoder-base-384
===================================================================

This script:
- Loads your fine-tuned Mizan text encoder
- Uses it inside MizanRAGPipeline
- Indexes real documents from /docs folder
- Runs retrieval using Mizan similarity
- Builds an answer
"""

from mizan_rag import MizanRAGPipeline
import glob
import os


def main():
    # -------------------------------------------------------------
    # 1. Load your OWN embedding model (trained using mizan-embedder)
    # -------------------------------------------------------------
    pipeline = MizanRAGPipeline(
        encoder_type="mizan",                     # ← Use your model, not HF
        encoder_model="intfloat/e5-base",# ← Folder containing tokenizer + model
        dim=384,
        chunk_size=400,
        chunk_overlap=100,
    )

    print("Loaded MizanRAGPipeline with your custom encoder.\n")

    # -------------------------------------------------------------
    # 2. Index real documents from /docs/*.txt
    # -------------------------------------------------------------
    docs_path = "docs"
    if not os.path.exists(docs_path):
        print(f"Folder '{docs_path}' not found. Please create docs/ and add .txt files.")
        return

    txt_files = glob.glob(os.path.join(docs_path, "*.txt"))

    if len(txt_files) == 0:
        print("No .txt files found in docs/. Please add some documents.")
        return

    print(f"Found {len(txt_files)} documents. Indexing...\n")

    for path in txt_files:
        doc_id = os.path.basename(path)
        pipeline.index_text_file(doc_id, path)
        print(f"Indexed: {doc_id}")

    print("\nIndexing complete.\n")

    # -------------------------------------------------------------
    # 3. Ask any real-time question
    # -------------------------------------------------------------
    query = "What are embeddings in machine learning?"
    print(f"QUESTION: {query}\n")

    answer = pipeline.build_answer(
        query_text=query,
        top_k=5,
        metric="mizan"      # ← always use Mizan similarity
    )

    print("=== FINAL ANSWER ===\n")
    print(answer)
    print("\n====================\n")


if __name__ == "__main__":
    main()
