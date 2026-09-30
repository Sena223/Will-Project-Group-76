import os
os.environ.setdefault("OPENAI_API_KEY", "dummy-not-used")

import json
import pandas as pd
import ollama
from pyserini.search.lucene import LuceneSearcher
from bert_score import score

THRESHOLD = 4.5
INDEX = "../target/indexes/bm25"
MODEL = "llama3.2:1b"

searcher = LuceneSearcher(INDEX)

topics = pd.read_csv("../data/topics.csv")
groundtruth = pd.read_csv("../data/groundtruth.csv")

results = []
answered_indices = []
all_candidates = []
all_references = []

print(f"Generating answers for {len(topics)} questions...\n")

for i, row in topics.iterrows():

    topic_id = row["topic_id"]
    question_id = row["question_id"]
    question = row["question"]

    print(f"[{i + 1}/{len(topics)}] {question_id}", end=" ")

    hits = searcher.search(question, 3)
    top_score = hits[0].score if hits else 0.0

    # Same threshold as chatbot.py
    if not hits or top_score < THRESHOLD:
        print(f"DECLINED ({top_score:.2f})")

        results.append({
            "topic_id": topic_id,
            "question_id": question_id,
            "question": question,
            "bm25_score": top_score,
            "answered": False,
            "generated_answer": "",
            "bertscore_precision": None,
            "bertscore_recall": None,
            "bertscore_f1": None
        })

        continue

    doc = json.loads(hits[0].lucene_document.get("raw"))
    passage = doc["contents"]

    prompt = f"""Answer the visitor's question using ONLY the information in the passage below. Be concise and friendly. If the passage doesn't fully answer the question, say what you do know from it.

Passage: {passage}

Question: {question}

Answer:"""

    response = ollama.generate(model=MODEL, prompt=prompt)
    generated_answer = response["response"].strip()

    references = groundtruth[
        groundtruth["topic_id"] == topic_id
    ]["passage"].dropna().tolist()

    result_index = len(results)

    results.append({
        "topic_id": topic_id,
        "question_id": question_id,
        "question": question,
        "bm25_score": top_score,
        "answered": True,
        "generated_answer": generated_answer,
        "bertscore_precision": None,
        "bertscore_recall": None,
        "bertscore_f1": None
    })

    # Create candidate/reference pairs.
    # Multiple references for one question are handled later
    # by selecting the highest BERTScore F1.
    for reference in references:
        answered_indices.append(result_index)
        all_candidates.append(generated_answer)
        all_references.append(reference)

    print(f"ANSWERED ({top_score:.2f})")


print("\nRunning BERTScore in one batch...")

P, R, F1 = score(
    all_candidates,
    all_references,
    lang="en",
    verbose=True
)

# Keep best reference match for each answered question
best_scores = {}

for pair_index, result_index in enumerate(answered_indices):

    current = {
        "precision": P[pair_index].item(),
        "recall": R[pair_index].item(),
        "f1": F1[pair_index].item()
    }

    if (
        result_index not in best_scores
        or current["f1"] > best_scores[result_index]["f1"]
    ):
        best_scores[result_index] = current


for result_index, values in best_scores.items():
    results[result_index]["bertscore_precision"] = values["precision"]
    results[result_index]["bertscore_recall"] = values["recall"]
    results[result_index]["bertscore_f1"] = values["f1"]


results_df = pd.DataFrame(results)

os.makedirs("../target/bertscore_results", exist_ok=True)

results_df.to_csv(
    "../target/bertscore_results/results.csv",
    index=False
)

answered = results_df[results_df["answered"] == True]

print("\n========== BERTSCORE SUMMARY ==========")
print(f"Total questions: {len(results_df)}")
print(f"Answered: {len(answered)}")
print(f"Declined: {len(results_df) - len(answered)}")

if len(answered) > 0:
    print(
        f"Average Precision: "
        f"{answered['bertscore_precision'].mean():.4f}"
    )
    print(
        f"Average Recall: "
        f"{answered['bertscore_recall'].mean():.4f}"
    )
    print(
        f"Average F1: "
        f"{answered['bertscore_f1'].mean():.4f}"
    )

print(
    "\nDetailed results saved to "
    "../target/bertscore_results/results.csv"
)