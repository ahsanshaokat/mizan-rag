import os
import json
from mizan_rag.pipeline import MizanRAGPipeline

# -------------------------------------------------------------
# Ground Truth (Document → Questions)
# -------------------------------------------------------------

GROUND_TRUTH = {
    "Alice_in_Wonderland.txt": [
        "Why does Alice keep changing size during her journey?",
        "Who does Alice meet sitting on a mushroom while smoking a hookah?",
        "Why is Alice frustrated during the Mad Tea Party?",
    ],
    "Frankenstein.txt": [
        "Why does Victor Frankenstein regret creating the Creature?",
        "Why does the Creature feel hatred toward Victor?",
        "What demand does the Creature make to Victor?",
    ],
    "Dracula.txt": [
        "Why does Jonathan Harker become suspicious of Count Dracula?",
        "How does Dracula travel to England?",
        "What method do Van Helsing and his team use to weaken Dracula’s power?",
    ],
    "Pride_and_Prejudice.txt": [
        "Why does Elizabeth Bennet initially dislike Mr. Darcy?",
        "What changes Elizabeth’s opinion of Mr. Darcy?",
        "Why does Lady Catherine visit Elizabeth?",
    ],
    "The_Picture_of_Dorian_Gray.txt": [
        "What does the portrait represent?",
        "Why does Dorian break his friendship with Basil?",
        "Why does Dorian stab the portrait?",
    ],
    "The_Time_Machine.txt": [
        "Who are the Eloi?",
        "Who are the Morlocks?",
        "Why does the Time Traveller fear the Morlocks?",
    ],
    "The_War_of_the_Worlds.txt": [
        "Why are the Martians defeated despite their superior technology?",
        "What are the Martian tripods used for?",
    ],
    "Moby_Dick.txt": [
        "Why is Captain Ahab obsessed with Moby Dick?",
        "What is Ishmael’s purpose for joining the whaling voyage?",
        "Why is Starbuck opposed to Ahab’s mission?",
    ],
    "Ulysses.txt": [
        "What literary technique dominates Ulysses?",
        "Who is the main character the novel follows during one day in Dublin?",
        "What is the central theme of the novel?",
    ],
    "Metamorphosis.txt": [
        "What happens to Gregor Samsa at the beginning of the story?",
        "Why does Gregor fear revealing himself to his family?",
        "Why does Gregor’s family become resentful of him?",
    ]
}

# Flatten Q → Correct Document
QUESTION_TO_DOC = {
    q: doc for doc, qs in GROUND_TRUTH.items() for q in qs
}


# -------------------------------------------------------------
# Run test
# -------------------------------------------------------------

def evaluate_rag():
    print("\n🔵 Initializing RAG Pipeline for Benchmarking...\n")

    rag = MizanRAGPipeline(
        embed_model="saved/mizan_encoder_v1",
        chunk_size=800,
        overlap=150,
        ranker_metric="mizan", 
        compare=True,
        enable_cache=True,
        cache_path=".mizan_cache",
        llm_provider=None,          # Disable LLM for speed
        summarizer_model=None,
        on_log=lambda x: None
    )

    print("🔍 Indexing documents...\n")
    rag.index_directory("docs")

    results = []

    router_correct = 0
    retrieval_correct = 0

    for question, correct_doc in QUESTION_TO_DOC.items():
        print(f"\n=== TESTING QUESTION ===\n{question}")
        print(f"Expected Document: {correct_doc}")

        result = rag.query(question, top_k_retrieve=25, top_k_rerank=10)

        routed_doc = result["final_ranked"][0][1] if result["final_ranked"] else None
        top_mizan = result["final_ranked"][0][0] if result["final_ranked"] else None

        # cosine comparison
        cos_row = result["comparison"][0] if result["comparison"] else None
        cos_score = cos_row["cosine_score"] if cos_row else None

        # router output
        router_output = rag.router.route(question, top_k=1)

        router_is_correct = router_output == correct_doc
        retrieval_is_correct = routed_doc == correct_doc

        if router_is_correct: router_correct += 1
        if retrieval_is_correct: retrieval_correct += 1

        results.append({
            "question": question,
            "expected_doc": correct_doc,
            "router_doc": router_output,
            "router_correct": router_is_correct,
            "top_mizan_doc": routed_doc,
            "mizan_correct": retrieval_is_correct,
            "mizan_top_score": float(top_mizan) if top_mizan is not None else None,
            "cosine_top_score": float(cos_score) if cos_score is not None else None,
        })

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    summary = {
        "router_accuracy": router_correct / len(QUESTION_TO_DOC),
        "retrieval_accuracy": retrieval_correct / len(QUESTION_TO_DOC),
        "total_questions": len(QUESTION_TO_DOC),
        "details": results
    }

    with open("rag_evaluation_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n\n📊 Benchmark Complete!")
    print("💾 Saved report → rag_evaluation_report.json\n")

    return summary


if __name__ == "__main__":
    evaluate_rag()
