import os
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from mizan_vector.metrics import mizan_similarity
from groq import Groq

# ====================================================
# 0. Setup Groq LLM Client
# ====================================================

client = Groq(api_key=os.getenv("API_KEY"))  # <--- INSERT YOUR KEY

LLM_MODEL = "llama-3.1-8b-instant"          # <--- BEST FOR FAST RAG


# ====================================================
# 1. Load Embedding Model
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
# 3. Load Docs -> Chunk -> Embed
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

    for idx, chunk in enumerate(chunks):
        emb = embed(chunk)
        chunk_store.append({
            "doc": filename,
            "chunk_id": idx,
            "text": chunk,
            "embedding": emb,
        })

print(f"\nTotal chunks stored: {len(chunk_store)}")
print("\n===== Chunked RAG + Groq LLM Ready =====\n")


# ====================================================
# 4. Retrieval Function (Cosine or Mizan)
# ====================================================
def retrieve(query, top_k=5, metric="cosine"):
    q_emb = embed(query)
    scores = []

    for item in chunk_store:
        emb = item["embedding"]

        if metric == "cosine":
            score = F.cosine_similarity(q_emb, emb, dim=0).item()
        else:
            score = mizan_similarity(q_emb, emb)

        scores.append((score, item))

    scores = sorted(scores, key=lambda x: x[0], reverse=True)
    return scores[:top_k]


# ====================================================
# 5. Groq LLM Answer Builder
# ====================================================
def build_answer(query, retrieved_chunks):
    context = "\n\n".join(
        f"[DOC={item['doc']} | CHUNK={item['chunk_id']}] {item['text']}"
        for score, item in retrieved_chunks
    )

    prompt = f"""
You are the MIZAN RAG assistant.
Use ONLY the context to answer.

QUESTION:
{query}

CONTEXT:
{context}

RULES:
- Do NOT hallucinate.
- If answer is not found, say: "I don't know based on the documents."
- Be concise and factual.

FINAL ANSWER:
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content


# ====================================================
# 6. Main Interactive Loop
# ====================================================
while True:
    query = input("\nEnter query (or 'exit'): ")
    if query.lower() == "exit":
        break

    retrieved = retrieve(query, top_k=5, metric="mizan")

    print("\n--- TOP RETRIEVED CHUNKS (MIZAN) ---")
    for score, item in retrieved:
        print(f"\nDOC: {item['doc']} | SCORE={score:.4f}")
        print(item["text"][:250], "...")

    print("\n--- LLM ANSWER (GROQ) ---")
    answer = build_answer(query, retrieved)
    print(answer)
