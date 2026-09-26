import os
os.environ.setdefault("OPENAI_API_KEY", "dummy-not-used")

import pandas as pd
from pyserini.output_writer import OutputFormat, get_output_writer
from pyserini.search.lucene import LuceneSearcher

DATA_DIR = "../data"
TOPICS = DATA_DIR + "/topics.csv"
INDEX_BM25 = "../target/indexes/bm25"
OUTPUT_PATH_BM25 = "../target/runs/rag-bm25.txt"


def run_bm25():
    topics = pd.read_csv(TOPICS)
    searcher = LuceneSearcher(INDEX_BM25)
    tag = "opera.rag.bm25"
    num_hits = 100
    output_writer = get_output_writer(OUTPUT_PATH_BM25, OutputFormat('trec'), 'w',
                                       max_hits=num_hits, tag=tag, topics=topics)
    with output_writer:
        for question_id, question in topics[['question_id', 'question']].values:
            hits = searcher.search(question, num_hits)
            output_writer.write(question_id, hits)
    print(f"wrote {OUTPUT_PATH_BM25}")


if __name__ == "__main__":
    run_bm25()
