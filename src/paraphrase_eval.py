import pandas as pd
import os

RESULTS = "../target/bertscore_results/results.csv"
OUTPUT_DIR = "../target/paraphrase_results"

df = pd.read_csv(RESULTS)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Number of different question phrasings for each topic
topic_sizes = df.groupby("topic_id").size()

# Only topics with more than one phrasing can be used
# for paraphrase robustness analysis
multi_topics = topic_sizes[topic_sizes > 1].index

para_df = df[df["topic_id"].isin(multi_topics)].copy()

topic_results = []

for topic_id, group in para_df.groupby("topic_id"):

    total = len(group)
    answered = int(group["answered"].sum())
    declined = total - answered

    # Did all paraphrases make the same answer/decline decision?
    consistent_decision = (
        group["answered"].nunique() == 1
    )

    bm25_min = group["bm25_score"].min()
    bm25_max = group["bm25_score"].max()
    bm25_range = bm25_max - bm25_min

    answered_group = group[group["answered"] == True]

    if len(answered_group) > 0:
        mean_bert_f1 = answered_group["bertscore_f1"].mean()
        min_bert_f1 = answered_group["bertscore_f1"].min()
        max_bert_f1 = answered_group["bertscore_f1"].max()
    else:
        mean_bert_f1 = None
        min_bert_f1 = None
        max_bert_f1 = None

    topic_results.append({
        "topic_id": topic_id,
        "num_paraphrases": total,
        "answered": answered,
        "declined": declined,
        "consistent_answer_decision": consistent_decision,
        "bm25_min": bm25_min,
        "bm25_max": bm25_max,
        "bm25_range": bm25_range,
        "mean_bertscore_f1": mean_bert_f1,
        "min_bertscore_f1": min_bert_f1,
        "max_bertscore_f1": max_bert_f1
    })


summary = pd.DataFrame(topic_results)

summary.to_csv(
    f"{OUTPUT_DIR}/paraphrase_topic_results.csv",
    index=False
)

# Overall statistics
total_topics = len(summary)

consistent_topics = int(
    summary["consistent_answer_decision"].sum()
)

inconsistent_topics = (
    total_topics - consistent_topics
)

consistency_rate = (
    consistent_topics / total_topics
    if total_topics else 0
)

average_bm25_range = summary["bm25_range"].mean()
max_bm25_range = summary["bm25_range"].max()

print("\n====== PARAPHRASE ROBUSTNESS ======")

print(f"Topics with multiple phrasings: {total_topics}")

print(
    f"Consistent answer/decline decision: "
    f"{consistent_topics}/{total_topics}"
)

print(
    f"Inconsistent answer/decline decision: "
    f"{inconsistent_topics}/{total_topics}"
)

print(
    f"Decision consistency rate: "
    f"{consistency_rate:.4f}"
)

print(
    f"Average BM25 score range within a topic: "
    f"{average_bm25_range:.4f}"
)

print(
    f"Largest BM25 score range within a topic: "
    f"{max_bm25_range:.4f}"
)

print("\nMost wording-sensitive topics:")

print(
    summary.sort_values(
        "bm25_range",
        ascending=False
    )[
        [
            "topic_id",
            "num_paraphrases",
            "answered",
            "declined",
            "bm25_min",
            "bm25_max",
            "bm25_range"
        ]
    ].head(10).to_string(index=False)
)

print(
    "\nDetailed results saved to "
    "../target/paraphrase_results/"
    "paraphrase_topic_results.csv"
)