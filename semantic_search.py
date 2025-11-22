import os
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from mizan_vector.metrics import mizan_similarity

# ---------------------------------------------------------
# 1. Load Model (CPU friendly)
# ---------------------------------------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def embed(text):
    """Compute normalized embedding."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        output = model(**enc).last_hidden_state
        emb = output.mean(dim=1)
        emb = F.normalize(emb, dim=1)
        return emb.squeeze(0)  # -> vector


# ---------------------------------------------------------
# 2. Chunking function
# ---------------------------------------------------------
def chunk_text(text, chunk_size=250, overlap=80):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        start += (chunk_size - overlap)

    return chunks


# ---------------------------------------------------------
# 3. Load and chunk documents
# ---------------------------------------------------------
docs_dir = "docs"
chunk_store = []

print("Loading and chunking documents...\n")

for filename in os.listdir(docs_dir):
    if filename.endswith(".txt"):
        path = os.path.join(docs_dir, filename)
        text = open(path, "r", encoding="utf-8", errors="ignore").read()

        chunks = chunk_text(text)
        print(f"{filename}: {len(chunks)} chunks")

        for idx, chunk in enumerate(chunks):
            emb = embed(chunk)
            chunk_store.append({
                "doc": filename,
                "chunk_id": idx,
                "text": chunk,
                "embedding": emb
            })

print(f"\nTotal chunks stored: {len(chunk_store)}\n")


# ---------------------------------------------------------
# 4. Search function
# ---------------------------------------------------------
def search(query, top_k=5, metric="cosine"):
    q_emb = embed(query)
    results = []

    for item in chunk_store:
        doc = item["doc"]
        chunk = item["text"]
        emb = item["embedding"]

        if metric == "cosine":
            score = F.cosine_similarity(q_emb, emb, dim=0).item()
        else:
            score = mizan_similarity(q_emb, emb)

        results.append((score, doc, chunk))

    results = sorted(results, key=lambda x: x[0], reverse=True)
    return results[:top_k]


# ---------------------------------------------------------
# 5. Build simple RAG answer from retrieved chunks
# ---------------------------------------------------------
def build_answer(query, top_k=5, metric="cosine"):
    top_chunks = search(query, top_k=top_k, metric=metric)

    answer = f"QUESTION: {query}\n\n"
    answer += f"TOP MATCHES ({metric}):\n"

    for score, doc, chunk in top_chunks:
        answer += f"\n[{doc}] (score={score:.4f})\n"
        answer += chunk[:350] + "...\n"

    # Combine retrieved chunks into “final answer”
    answer += "\n\nFINAL ANSWER:\n"
    answer += "Based on the retrieved information above, here is a summary:\n\n"

    # Simple summarization (no LLM)
    summary_parts = [chunk for _, _, chunk in top_chunks]
    summary = " ".join(summary_parts)
    answer += summary[:1000] + "..."   # Truncate to readable length

    return answer


# ---------------------------------------------------------
# 6. Interactive loop
# ---------------------------------------------------------
print("===== CHUNKED RAG SEARCH READY =====")

while True:
    query = input("\nEnter your query (or 'exit'): ")
    if query.lower() == "exit":
        break

    print("\n\n=== COSINE RAG ANSWER ===\n")
    print(build_answer(query, metric="cosine"))

    print("\n\n=== MIZAN RAG ANSWER ===\n")
    print(build_answer(query, metric="mizan"))
