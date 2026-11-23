"""Đánh giá chất lượng truy hồi (Retrieval Evaluation) và truy xuất nguồn gốc (Citation Traceability).

Chỉ số:
  - Hits@k (k=1, 3, 5, 10): Tỉ lệ truy vấn có ít nhất 1 tài liệu đúng trong top k.
  - Recall@k: Tỉ lệ tài liệu đúng được truy hồi trong top k.
  - MRR (Mean Reciprocal Rank): Điểm thứ hạng nghịch đảo trung bình của kết quả đúng đầu tiên.
  - Citation Traceability: Tỉ lệ tài liệu tham khảo trong đề cương đối chiếu được về corpus.

Chạy:
  python src/eval_retrieval.py --corpus corpus --gold eval/gold/queries.jsonl
  python src/eval_retrieval.py --syllabus outputs/IT3011.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import rag
from rag import Hit, Retriever
from rubric import check_references
from schema import Syllabus

log = logging.getLogger("eval_retrieval")


def load_gold_queries(path: str | Path) -> list[dict]:
    """Đọc tập queries gold chuẩn từ JSONL."""
    p = Path(path)
    if not p.exists():
        return []
    queries = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    queries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    log.warning("Bỏ qua dòng lỗi JSON: %s", e)
    return queries


def is_hit_relevant(hit: Hit, relevant_items: list[dict]) -> bool:
    """Kiểm tra một Hit có khớp với danh sách relevant trong gold không.

    Khớp theo tên file (source_pdf hoặc source) và khoảng trang nếu có.
    """
    chunk = hit.chunk
    chunk_file = (chunk.meta.get("source_pdf") or chunk.source or "").lower()
    chunk_page = chunk.meta.get("page")  # nếu có

    for rel in relevant_items:
        rel_file = rel.get("file", "").lower()
        if rel_file and (rel_file in chunk_file or chunk_file in rel_file):
            pages = rel.get("pages")
            if pages and len(pages) == 2 and chunk_page is not None:
                if pages[0] <= chunk_page <= pages[1]:
                    return True
            else:
                return True
    return False


def evaluate_retrieval(retriever: Retriever, queries: list[dict],
                       k_values: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    """Tính Hits@k, Recall@k, MRR trên tập queries."""
    n = len(queries)
    if n == 0:
        return {"n": 0, "hits": {}, "recall": {}, "mrr": 0.0}

    hits_at_k = {k: 0 for k in k_values}
    recall_at_k = {k: 0.0 for k in k_values}
    reciprocal_ranks = []

    max_k = max(k_values)

    for q in queries:
        query_text = q["query"]
        relevant = q.get("relevant", [])
        if not relevant:
            continue

        course = q.get("course")
        # Truy hồi loại trừ đề cương mẫu cùng mã học phần để chống rò rỉ
        results = retriever.search(query_text, k=max_k, exclude_course=course)

        # Tính rank của hit đúng đầu tiên cho MRR
        first_rel_rank = None
        for rank, h in enumerate(results, start=1):
            if is_hit_relevant(h, relevant):
                first_rel_rank = rank
                break

        reciprocal_ranks.append(1.0 / first_rel_rank if first_rel_rank else 0.0)

        # Hits@k và Recall@k
        for k in k_values:
            top_k = results[:k]
            matched_rels = set()
            hit_found = False
            for h in top_k:
                for idx, rel in enumerate(relevant):
                    if is_hit_relevant(h, [rel]):
                        matched_rels.add(idx)
                        hit_found = True
            if hit_found:
                hits_at_k[k] += 1
            recall_at_k[k] += (len(matched_rels) / len(relevant)) if relevant else 0.0

    return {
        "n": n,
        "hits": {k: round(hits_at_k[k] / n, 4) for k in k_values},
        "recall": {k: round(recall_at_k[k] / n, 4) for k in k_values},
        "mrr": round(sum(reciprocal_ranks) / n, 4) if n else 0.0,
    }


def compute_citation_traceability(syllabus_path: str | Path, corpus_path: str | Path) -> dict:
    """Tính tỉ lệ reference trace được về corpus (M6)."""
    p = Path(syllabus_path)
    if not p.exists():
        return {"error": f"Không tìm thấy file {syllabus_path}"}

    data = json.loads(p.read_text(encoding="utf-8"))
    s = Syllabus.model_validate(data)
    retr = rag.build_retriever(corpus_path, use_dense=False)
    corpus_refs = rag.reference_texts(retr)

    all_readings = [*s.required_readings, *s.recommended_readings]
    total_refs = len(all_readings)
    if total_refs == 0:
        return {"total_refs": 0, "traceable": 0, "rate": 0.0, "findings": []}

    findings = check_references(s, corpus_refs)
    untraceable = sum(1 for f in findings if f.rule == "ref_not_in_corpus")
    traceable = total_refs - untraceable

    return {
        "total_refs": total_refs,
        "traceable": traceable,
        "untraceable": untraceable,
        "rate": round(traceable / total_refs, 4) if total_refs else 0.0,
        "findings": [f.message for f in findings if f.rule == "ref_not_in_corpus"],
    }


def print_evaluation_report(bm25_metrics: dict, hybrid_metrics: dict | None,
                            traceability: dict | None) -> None:
    """In báo cáo đánh giá ra terminal."""
    print(f"\n{'='*65}")
    print("BÁO CÁO ĐÁNH GIÁ TRUY HỒI & NGUỒN GỐC TÀI LIỆU")
    print(f"{'='*65}")

    n = bm25_metrics.get("n", 0)
    print(f"Số lượng truy vấn (n): {n}")
    if n < 30:
        print("  ⚠ LƯU Ý: n nhỏ (n < 30) → KẾT LUẬN YẾU, chỉ mang tính chất tham khảo.")

    print("\n1. So sánh BM25 vs Hybrid (BM25 + Dense Fusion):")
    headers = ["Chỉ số", "BM25-only", "Hybrid (Dense+BM25)"]
    print(f"  {headers[0]:<15} | {headers[1]:<12} | {headers[2]:<20}")
    print(f"  {'-'*15}-+-{'-'*12}-+-{'-'*20}")
    print(f"  {'MRR':<15} | {bm25_metrics.get('mrr', 0.0):<12.4f} | "
          f"{hybrid_metrics.get('mrr', 'N/A') if hybrid_metrics else 'N/A'}")

    for k in [1, 3, 5, 10]:
        bm_hit = bm25_metrics.get("hits", {}).get(k, 0.0)
        hy_hit = hybrid_metrics.get("hits", {}).get(k, "N/A") if hybrid_metrics else "N/A"
        print(f"  {f'Hits@{k}':<15} | {bm_hit:<12.4f} | {hy_hit}")

    for k in [1, 3, 5, 10]:
        bm_rec = bm25_metrics.get("recall", {}).get(k, 0.0)
        hy_rec = hybrid_metrics.get("recall", {}).get(k, "N/A") if hybrid_metrics else "N/A"
        print(f"  {f'Recall@{k}':<15} | {bm_rec:<12.4f} | {hy_rec}")

    if traceability and "error" not in traceability:
        print("\n2. Truy xuất nguồn gốc tài liệu (Citation Traceability - M6):")
        print(f"  Tổng số tài liệu tham khảo: {traceability.get('total_refs', 0)}")
        print(f"  Tài liệu đối chiếu được về corpus: {traceability.get('traceable', 0)}")
        print(f"  Tài liệu không rõ nguồn: {traceability.get('untraceable', 0)}")
        rate = traceability.get("rate", 0.0)
        print(f"  Tỉ lệ truy xuất nguồn (Citation Traceability): {rate*100:.1f}%")
        if traceability.get("findings"):
            print("  Chi tiết nghi vấn:")
            for msg in traceability["findings"]:
                print(f"    - {msg}")
    print(f"{'='*65}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá truy hồi và truy xuất nguồn gốc tài liệu.")
    parser.add_argument("--corpus", default="corpus", help="Thư mục corpus")
    parser.add_argument("--gold", default="eval/gold/queries.jsonl", help="File gold queries JSONL")
    parser.add_argument("--syllabus", default=None, help="File Syllabus JSON để đo citation traceability")
    args = parser.parse_args()

    queries = load_gold_queries(args.gold)
    if not queries:
        print(f"Không có query nào trong {args.gold} hoặc file không tồn tại.")
        return

    # BM25-only
    r_bm25 = rag.build_retriever(args.corpus, use_dense=False)
    if not r_bm25:
        print(f"Không thể khởi tạo retriever từ {args.corpus}.")
        return

    bm25_metrics = evaluate_retrieval(r_bm25, queries)

    # Hybrid nếu có LLM_EMBED_MODEL
    hybrid_metrics = None
    import llm
    if llm.has_embeddings():
        r_hybrid = rag.build_retriever(args.corpus, use_dense=True)
        if r_hybrid:
            hybrid_metrics = evaluate_retrieval(r_hybrid, queries)

    traceability = None
    if args.syllabus:
        traceability = compute_citation_traceability(args.syllabus, args.corpus)

    print_evaluation_report(bm25_metrics, hybrid_metrics, traceability)


if __name__ == "__main__":
    main()
