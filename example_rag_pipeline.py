from mizan_rag import MizanRAGPipeline

def main():
    pipeline = MizanRAGPipeline(
        encoder_type="hf",
        encoder_model="all-MiniLM-L6-v2",
        dim=384,
        chunk_size=400,
        chunk_overlap=100,
    )

    doc_text = """Mizan is a scale-aware similarity function. It improves retrieval quality
    in RAG systems by accounting for proportional differences between embeddings,
    unlike cosine similarity which only sees direction. Mizan works well with
    long documents, noisy data, and multi-scale embeddings."""

    pipeline.index_document("doc1", doc_text, extra_metadata={"title": "About Mizan"})

    query = "How does Mizan help RAG retrieval?"
    answer = pipeline.build_answer(query, top_k=3, metric="mizan")
    print(answer)

if __name__ == "__main__":
    main()
