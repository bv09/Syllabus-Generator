"""Tests cho src/eval_retrieval.py (M5, M6)."""
from pathlib import Path

from eval_retrieval import evaluate_retrieval, is_hit_relevant, load_gold_queries
from rag import Chunk, Hit, Retriever


def test_load_gold_queries(tmp_path):
    p = tmp_path / "queries.jsonl"
    p.write_text('{"id": "q1", "query": "test query", "relevant": [{"file": "f1.pdf"}]}\n', encoding="utf-8")
    queries = load_gold_queries(p)
    assert len(queries) == 1
    assert queries[0]["id"] == "q1"


def test_is_hit_relevant():
    chunk = Chunk("c1", "text", "general", "ch1.pdf", meta={"source_pdf": "ch1.pdf", "page": 5})
    hit = Hit(chunk, 1.0)

    # Khớp file và trang trong khoảng [1, 10]
    assert is_hit_relevant(hit, [{"file": "ch1.pdf", "pages": [1, 10]}])
    # Khớp file nhưng trang ngoài khoảng [6, 10]
    assert not is_hit_relevant(hit, [{"file": "ch1.pdf", "pages": [6, 10]}])
    # Khớp file không cần số trang
    assert is_hit_relevant(hit, [{"file": "ch1.pdf"}])
    # Sai file
    assert not is_hit_relevant(hit, [{"file": "ch2.pdf"}])


def test_evaluate_retrieval_metrics():
    c1 = Chunk("c1", "sắp xếp quicksort", "general", "01_sort.pdf", meta={"source_pdf": "01_sort.pdf"})
    c2 = Chunk("c2", "cây nhị phân", "general", "02_tree.pdf", meta={"source_pdf": "02_tree.pdf"})
    r = Retriever([c1, c2])

    queries = [
        {"id": "q1", "query": "quicksort", "relevant": [{"file": "01_sort.pdf"}]},
        {"id": "q2", "query": "nhị phân", "relevant": [{"file": "02_tree.pdf"}]},
    ]

    metrics = evaluate_retrieval(r, queries, k_values=(1, 2))
    assert metrics["n"] == 2
    assert metrics["mrr"] == 1.0
    assert metrics["hits"][1] == 1.0
    assert metrics["recall"][1] == 1.0
