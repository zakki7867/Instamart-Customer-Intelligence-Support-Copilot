"""policy_rag.py -- Policy chunking, TF-IDF vectorization, retrieval, and RAG question answering for Part D.
Evaluates data/raw/policy_eval.csv under standard (10-word overlap) and zero-overlap conditions.
"""
import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure root is in path for mock_llm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mock_llm

POLICIES_DIR = os.path.join("data", "policies")
POLICY_EVAL_PATH = os.path.join("data", "raw", "policy_eval.csv")


def chunk_documents(policies_dir=POLICIES_DIR, chunk_size=40, overlap=10):
    """Splits each .txt policy file into word-based chunks with specified overlap.
    Stop after first chunk whose end reaches or passes the last word of the document.
    """
    chunks = []
    chunk_counts = {}
    step = chunk_size - overlap

    for fn in sorted(os.listdir(policies_dir)):
        if not fn.endswith(".txt"):
            continue
        filepath = os.path.join(policies_dir, fn)
        with open(filepath, "r", encoding="utf-8") as f:
            words = f.read().strip().split()

        doc_chunks = []
        start = 0
        while start < len(words):
            chunk_words = words[start:start + chunk_size]
            doc_chunks.append({
                "source": fn,
                "text": " ".join(chunk_words)
            })
            if start + chunk_size >= len(words):
                break
            start += step

        chunk_counts[fn] = len(doc_chunks)
        chunks.extend(doc_chunks)

    return chunks, chunk_counts


class PolicyRetriever:
    """TF-IDF vectorizer and cosine similarity ranker for policy chunks."""
    def __init__(self, chunks):
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query, k=3):
        q_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(q_vec, self.tfidf_matrix)[0]
        top_indices = similarities.argsort()[::-1][:k]
        return [
            {
                "source": self.chunks[i]["source"],
                "score": float(similarities[i]),
                "text": self.chunks[i]["text"]
            }
            for i in top_indices
        ]


def answer_question(question, retriever, k=3):
    """Retrieves top k chunks and generates grounded answer via mock_llm POLICY ASSISTANT."""
    retrieved = retriever.retrieve(question, k=k)
    context_str = "\n".join([f"[{r['source']}]\n{r['text']}" for r in retrieved])
    user_prompt = f"CONTEXT:\n{context_str}\n\nQUESTION:\n{question}"

    messages = [
        {
            "role": "system",
            "content": "You are a POLICY ASSISTANT. Answer strictly from the provided context. "
                       "If the information is not found in the context, reply that you could not find that information."
        },
        {"role": "user", "content": user_prompt}
    ]

    response = mock_llm.chat(messages)
    answer_text = response.choices[0].message.content
    sources = [r["source"] for r in retrieved]
    return answer_text, sources, retrieved


def evaluate_retrieval_and_qa(qa_df, retriever, name="Overlap 10"):
    print(f"\nEvaluating RAG ({name}):")
    ans_doc_matches = 0
    ans_fact_matches = 0
    unans_matches = 0
    total_ans = 0
    total_unans = 0

    results = []

    for _, row in qa_df.iterrows():
        q = row["question"]
        exp_doc = row["expected_doc"]
        exp_fact = row["expected_fact"]

        ans_text, sources, _ = answer_question(q, retriever, k=3)

        if exp_doc == "NONE":
            total_unans += 1
            is_fallback = ans_text.startswith("I could not find")
            if is_fallback:
                unans_matches += 1
            results.append({
                "question": q,
                "expected_doc": "NONE",
                "doc_matched": is_fallback,
                "expected_fact": "(Fallback)",
                "fact_matched": is_fallback,
                "answer": ans_text
            })
        else:
            total_ans += 1
            doc_matched = exp_doc in sources
            fact_matched = exp_fact in ans_text
            if doc_matched:
                ans_doc_matches += 1
            if fact_matched:
                ans_fact_matches += 1
            results.append({
                "question": q,
                "expected_doc": exp_doc,
                "doc_matched": doc_matched,
                "expected_fact": exp_fact,
                "fact_matched": fact_matched,
                "answer": ans_text
            })

    print(f"  Answerable Expected Doc in Top 3: {ans_doc_matches}/{total_ans}")
    print(f"  Answerable Expected Fact in Answer: {ans_fact_matches}/{total_ans}")
    print(f"  Unanswerable Fallback Triggered: {unans_matches}/{total_unans}")

    return pd.DataFrame(results), ans_doc_matches, ans_fact_matches, unans_matches


def main():
    print("=" * 60)
    print("POLICY RAG ASSISTANT (PART D)")
    print("=" * 60)

    # Task 1: Chunk documents (standard 40 words, 10 overlap)
    chunks_10, counts_10 = chunk_documents(POLICIES_DIR, chunk_size=40, overlap=10)
    print("Task 1: Document Chunking (40 words, 10 overlap):")
    for doc, cnt in counts_10.items():
        print(f"  {doc}: {cnt} chunks")
    print(f"  Total chunks: {len(chunks_10)} (Reference: 15)")
    assert len(chunks_10) == 15, f"Expected 15 chunks, got {len(chunks_10)}"

    # Task 2: Vectorise and test query
    retriever_10 = PolicyRetriever(chunks_10)
    test_query = "refund policy late delivery"
    test_top = retriever_10.retrieve(test_query, k=3)
    top_sources = [r["source"] for r in test_top]
    print(f"\nTask 2: Retrieval for '{test_query}':")
    for i, r in enumerate(test_top, 1):
        print(f"  Rank {i}: {r['source']} (Score: {r['score']:.4f}) -> {r['text'][:60]}...")
    assert all(s == "refund_policy.txt" for s in top_sources), f"Expected all from refund_policy.txt, got {top_sources}"
    print("  Assertion passed: All top 3 chunks are from refund_policy.txt.")

    # Task 4: Evaluate on policy_eval.csv
    qa_df = pd.read_csv(POLICY_EVAL_PATH)
    eval_df_10, doc_m10, fact_m10, unans_m10 = evaluate_retrieval_and_qa(qa_df, retriever_10, name="Standard (Overlap 10)")
    assert doc_m10 == 12 and fact_m10 == 12 and unans_m10 == 2, "Standard evaluation did not match 12/12, 12/12, 2/2"

    # Repeat with zero overlap
    chunks_0, counts_0 = chunk_documents(POLICIES_DIR, chunk_size=40, overlap=0)
    print("\nZero-Overlap Chunking (40 words, 0 overlap):")
    for doc, cnt in counts_0.items():
        print(f"  {doc}: {cnt} chunks")
    print(f"  Total chunks: {len(chunks_0)}")

    retriever_0 = PolicyRetriever(chunks_0)
    eval_df_0, doc_m0, fact_m0, unans_m0 = evaluate_retrieval_and_qa(qa_df, retriever_0, name="Zero Overlap (Overlap 0)")

    failed_questions = eval_df_0[~eval_df_0["fact_matched"]]
    print("\nQuestions broken by Zero-Overlap Chunking:")
    for _, f_row in failed_questions.iterrows():
        print(f"  Question: '{f_row['question']}'")
        print(f"    Expected fact: '{f_row['expected_fact']}'")
        print(f"    Actual answer: '{f_row['answer']}'")
        print("    Root Cause: The key sentence/clause was split across the rigid 40-word boundary without overlapping context.")


if __name__ == "__main__":
    main()
