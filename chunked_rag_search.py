import os
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from mizanvector.metrics import mizan_similarity


# ============================================================
# 1. LOAD MODEL
# ============================================================
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def embed(text):
    """Return normalized embedding."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        output = model(**enc).last_hidden_state
        emb = output.mean(dim=1)
        emb = F.normalize(emb, dim=1)
        return emb.squeeze(0)
    

# ============================================================
# 2. TEXT CHUNKING
# ============================================================
def chunk_text(text, chunk_size=300, overlap=80):
    """Split text into overlapping chunks."""
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


# ============================================================
# 3. LOAD DOCUMENTS + CHUNK + EMBED
# ============================================================
docs_dir = "docs"
chunk_store = []  # holds all chunks
print("Loading & chunking documents...\n")

for filename in os.listdir(docs_dir):
    if not filename.endswith(".txt"):
        continue

    path = os.path.join(docs_dir, filename)
    text = open(path, "r", encoding="utf-8", errors="ignore").read()

    chunks = chunk_text(text, chunk_size=250, overlap=80)

    for idx, chunk in enumerate(chunks):
        emb = embed(chunk)

        chunk_store.append({
            "doc": filename,
            "chunk_id": idx,
            "text": chunk,
            "embedding": emb,
        })

    print(f"{filename} → {len(chunks)} chunks created.")

print(f"\nTotal chunks stored: {len(chunk_store)}\n")


# ============================================================
# 4. SEARCH FUNCTION (COSINE + MIZAN)
# ============================================================
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

    # Sort by score desc
    results = sorted(results, key=lambda x: x[0], reverse=True)

    return results[:top_k]


# ============================================================
# 5. INTERACTIVE QUERY
# ============================================================
print("===== CHUNKED RAG SEARCH READY =====\n")

while True:
    query = input("\nEnter query (or 'exit'): ")
    if query.lower() == "exit":
        break

    print("\n--- COSINE RESULTS ---")
    for score, doc, chunk in search(query, metric="cosine"):
        print(f"\nDoc: {doc}  |  Score: {score:.4f}")
        print("Chunk:", chunk[:200], "...")

    print("\n--- MIZAN RESULTS ---")
    for score, doc, chunk in search(query, metric="mizan"):
        print(f"\nDoc: {doc}  |  Score: {score:.4f}")
        print("Chunk:", chunk[:200], "...")
