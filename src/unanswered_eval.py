import os
os.environ.setdefault("OPENAI_API_KEY", "dummy-not-used")
import pandas as pd
from pyserini.search.lucene import LuceneSearcher

THRESHOLD = 4.5

searcher = LuceneSearcher("../target/indexes/bm25")
topics = pd.read_csv("../data/topics.csv")
oos = pd.read_csv("../data/out_of_scope_questions.csv")

false_decline = 0
for q in topics['question']:
    hits = searcher.search(q, 1)
    score = hits[0].score if hits else 0.0
    if score < THRESHOLD:
        false_decline += 1

false_answer = 0
for q in oos['question']:
    hits = searcher.search(q, 1)
    score = hits[0].score if hits else 0.0
    if score >= THRESHOLD:
        false_answer += 1

print(f"Threshold = {THRESHOLD}")
print(f"In-scope questions wrongly declined: {false_decline}/{len(topics)}")
print(f"Out-of-scope questions wrongly answered: {false_answer}/{len(oos)}")