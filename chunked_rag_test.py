import os
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from mizan_vector.metrics import mizan_similarity


# ====================================================
# 1. Load Model (CPU Fast + Strong)
# ====================================================
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def embed(text):
    """Return normalized embedding vector."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        output = model(**enc).last_hidden_state
        emb = output.mean(dim=1)
        emb = F.normalize(emb, dim=1)
        return emb.squeeze(0)


# ====================================================
# 2. Chunking Function
# ====================================================
def chunk_text(text, chunk_size=250, overlap=80):
    """Split long text into overlapping chunks."""
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += (chunk_size - overlap)

    return chunks


# ====================================================
# 3. Load + Chunk + Embed Docs
# ====================================================
docs_dir = "docs"
chunk_store = []

print("\nLoading and chunking documents...\n")

for filename in os.listdir(docs_dir):
    if not filename.endswith(".txt"):
        continue

    filepath = os.path.join(docs_dir, filename)
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    chunks = chunk_text(text)

    print(f"{filename} → {len(chunks)} chunks")

    # Embed each chunk
    for idx, chunk in enumerate(chunks):
        emb = embed(chunk)
        chunk_store.append({
            "doc": filename,
            "chunk_id": idx,
            "text": chunk,
            "embedding": emb,
        })

print(f"\nTotal chunks stored: {len(chunk_store)}")
print("\n===== Chunked RAG Search Ready =====\n")


# ====================================================
# 4. Search Function
# ====================================================
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

    # Sort by highest score
    results = sorted(results, key=lambda x: x[0], reverse=True)
    return results[:top_k]


# ====================================================
# 5. Interactive Search Loop
# ====================================================
while True:
    query = input("\nEnter query (or 'exit'): ")
    if query.lower() == "exit":
        break

    print("\n--- TOP RESULTS (COSINE) ---")
    cosine_results = search(query, top_k=5, metric="cosine")
    for score, doc, chunk in cosine_results:
        print(f"\nDOC: {doc}  |  SCORE: {score:.4f}")
        print(chunk[:300], "...")

    print("\n--- TOP RESULTS (MIZAN) ---")
    mizan_results = search(query, top_k=5, metric="mizan")
    for score, doc, chunk in mizan_results:
        print(f"\nDOC: {doc}  |  SCORE: {score:.4f}")
        print(chunk[:300], "...")
