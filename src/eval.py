import sys
import pandas as pd
from ranx import evaluate, Qrels, Run

METRICS = ["ndcg@1", "ndcg@3", "ndcg@5", "recall@5", "recall@10"]


def main():
    qrels_df = pd.read_csv("../data/qrels.txt", sep='\t', names=['q_id', '0', 'doc_id', 'score'], header=None)
    qrels_df['q_id'] = qrels_df['q_id'].astype(object)
    qrels_df['doc_id'] = qrels_df['doc_id'].astype(object)
    qrels = Qrels.from_df(qrels_df, q_id_col="q_id", doc_id_col="doc_id", score_col="score")
    run = Run.from_file("../target/runs/rag-bm25.txt", kind="trec")

    scores = evaluate(qrels, run, METRICS, make_comparable=True)
    lines = [f"Run: {run.name}"]
    for metric, value in scores.items():
        lines.append(f"  {metric}: {value:.4f}")
    report = "\n".join(lines)
    print(report)
    with open("../target/trec_eval_results/summary.txt", "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    sys.exit(main())
