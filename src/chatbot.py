import os
os.environ.setdefault("OPENAI_API_KEY", "dummy-not-used")
import json
import ollama
from pyserini.search.lucene import LuceneSearcher

THRESHOLD = 4.5
INDEX = "../target/indexes/bm25"
MODEL = "llama3.2:1b"

searcher = LuceneSearcher(INDEX)

print("Sydney Opera House Visitor Chatbot — type a question, or 'quit' to exit")
while True:
    question = input("\nYou: ").strip()
    if question.lower() in ("quit", "exit"):
        break
    if not question:
        continue

    hits = searcher.search(question, 3)
    top_score = hits[0].score if hits else 0.0

    if not hits or top_score < THRESHOLD:
        print(f"Bot: I'm sorry, I don't have information about that. (top score: {top_score:.2f})")
        continue

    doc = json.loads(hits[0].lucene_document.get('raw'))
    passage = doc['contents']

    prompt = f"""Answer the visitor's question using ONLY the information in the passage below. Be concise and friendly. If the passage doesn't fully answer the question, say what you do know from it.

Passage: {passage}

Question: {question}

Answer:"""

    response = ollama.generate(model=MODEL, prompt=prompt)
    print(f"Bot: {response['response'].strip()}")
    print(f"     (source: {hits[0].docid}, retrieval score: {top_score:.2f})")