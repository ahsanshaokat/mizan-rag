import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from mizan_vector.metrics import mizan_similarity


# -------------------------------------------
# MODELS TO TEST (CPU-friendly)
# -------------------------------------------
MODELS = [
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/e5-base",
    "intfloat/e5-small-v2",
    "distilbert-base-uncased",
    "prajjwal1/bert-mini",
    "sentence-transformers/paraphrase-MiniLM-L3-v2"
]


# -------------------------------------------
# Embedding function (generic for all models)
# -------------------------------------------
def embed(text, tokenizer, model):
    enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**enc).last_hidden_state
        emb = output.mean(dim=1)
        emb = F.normalize(emb, dim=1)
        return emb


# -------------------------------------------
# POSITIVE / NEGATIVE SENTENCES
# -------------------------------------------
pos_sentences = [
    "Embeddings convert text into dense numerical vectors.",
    "Sentence embeddings represent meaning in vector space.",
    "Embeddings allow AI systems to compare semantic similarity.",
    "Transformer models produce contextual embeddings.",
    "Vector similarity helps retrieval rank relevant documents.",
    "Contrastive learning improves embedding separation.",
    "Embeddings capture linguistic semantic relationships.",
    "Semantic search uses embeddings to match meaning.",
    "RAG systems depend on embedding quality for retrieval.",
    "Normalized embeddings stabilize similarity calculations.",
    "High-quality embeddings cluster related ideas.",
    "Embeddings encode semantic structure of language.",
    "Embedding models generalize across many text types.",
    "Neural embeddings support reasoning over text.",
    "Transformers create contextual vector representations.",
    "Semantic distance comes from embedding geometry.",
    "Models use embeddings for intent understanding.",
    "Document retrieval relies on embedding accuracy.",
    "Semantic vectors represent conceptual meaning.",
    "Embedding models map sentences into vector spaces."
]

neg_sentences = [
    "The Amazon rainforest contains diverse plant species.",
    "Mount Everest is the tallest mountain in the world.",
    "A solar eclipse occurs when the moon blocks the sun.",
    "Classical music orchestras perform symphonies.",
    "The Pacific Ocean is the largest body of water.",
    "Horses have been used for transport for thousands of years.",
    "Volcanoes erupt when pressure builds underground.",
    "Many countries celebrate festivals in spring.",
    "Basketball is a sport played by two teams.",
    "The human heart pumps blood throughout the body.",
    "Modern cities rely on public transportation.",
    "Solar panels convert sunlight into electricity.",
    "Birds migrate long distances seasonally.",
    "The Great Wall of China is a historic structure.",
    "Airplanes operate using aerodynamics.",
    "Fish live in rivers and oceans.",
    "Agriculture depends on soil and water.",
    "Most nations maintain armies for defense.",
    "Cars use engines or electric motors.",
    "Clouds form when vapor condenses."
]

query = "What is an embedding model in machine learning?"


# -------------------------------------------
# RUN BENCHMARK FOR ALL MODELS
# -------------------------------------------
def test_model(model_name):
    print("\n\n====================================================")
    print(f"TESTING MODEL: {model_name}")
    print("====================================================")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    query_emb = embed(query, tokenizer, model)

    cos_pos_scores, cos_neg_scores = [], []
    miz_pos_scores, miz_neg_scores = [], []

    for i in range(20):
        pos_emb = embed(pos_sentences[i], tokenizer, model)
        neg_emb = embed(neg_sentences[i], tokenizer, model)

        cos_pos = F.cosine_similarity(query_emb, pos_emb).item()
        cos_neg = F.cosine_similarity(query_emb, neg_emb).item()

        miz_pos = mizan_similarity(query_emb, pos_emb)
        miz_neg = mizan_similarity(query_emb, neg_emb)

        cos_pos_scores.append(cos_pos)
        cos_neg_scores.append(cos_neg)
        miz_pos_scores.append(miz_pos)
        miz_neg_scores.append(miz_neg)

    print("\nAVERAGES:")
    print("------------------------------------")
    print(f"Cosine POS avg : {sum(cos_pos_scores)/20:.4f}")
    print(f"Cosine NEG avg : {sum(cos_neg_scores)/20:.4f}")
    print(f"Mizan POS avg  : {sum(miz_pos_scores)/20:.4f}")
    print(f"Mizan NEG avg  : {sum(miz_neg_scores)/20:.4f}")

    print("\nSEPARATION (higher = better):")
    print("------------------------------------")
    print(f"Cosine SEP : {(sum(cos_pos_scores)/20) - (sum(cos_neg_scores)/20):.4f}")
    print(f"Mizan SEP  : {(sum(miz_pos_scores)/20) - (sum(miz_neg_scores)/20):.4f}")


# Run tests
for m in MODELS:
    test_model(m)
