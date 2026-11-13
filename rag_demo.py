"""Minimal RAG pipeline: retrieve relevant chunks, then answer from them.

Retrieval is real (TF-IDF + cosine similarity). Generation here is extractive:
the answer is the best-matching chunk. build_prompt() shows how the retrieved
text would be handed to an LLM in a production setup.

Usage:  python rag_demo.py "What did you build at NullClass?"
        python rag_demo.py            (interactive mode)
"""
import json
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB = json.loads((Path(__file__).parent / "knowledge_base.json").read_text(encoding="utf-8"))
TEXTS = [f"{c['title']}. {c['text']}" for c in KB]
VECTORIZER = TfidfVectorizer(stop_words="english", sublinear_tf=True)
MATRIX = VECTORIZER.fit_transform(TEXTS)
MIN_SCORE = 0.1


def retrieve(question, k=3):
    scores = cosine_similarity(VECTORIZER.transform([question]), MATRIX)[0]
    ranked = scores.argsort()[::-1][:k]
    return [(KB[i], float(scores[i])) for i in ranked]


def build_prompt(question, hits):
    """What you would send to an LLM (not called here)."""
    context = "\n\n".join(chunk["text"] for chunk, _ in hits)
    return (f"Answer using only the context below. If it is not there, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}")


def answer(question):
    hits = retrieve(question)
    if hits[0][1] < MIN_SCORE:
        return "I couldn't find that in the knowledge base.", hits
    return hits[0][0]["text"], hits


def show(question):
    text, hits = answer(question)
    print(f"\nQ: {question}\nA: {text}\n\nSources:")
    for chunk, score in hits:
        print(f"  - {chunk['title'].split(':')[0]} (similarity {score:.2f})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
    else:
        while (q := input("\nAsk a question (blank to quit): ").strip()):
            show(q)
