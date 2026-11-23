import src.rag
from src.rag import Chunk, Retriever


def make():
    return Retriever([
        Chunk("a:0", "Cormen Introduction to Algorithms MIT Press 2022", src.rag.REFERENCE, "ref"),
        Chunk("b:0", "Chương 7 Sắp xếp: merge sort, quick sort, heap sort", src.rag.SYLLABUS, "s", course="IT3011"),
        Chunk("c:0", "Chương 5 Mạng máy tính: giao thức TCP", src.rag.SYLLABUS, "s2", course="IT3080"),
    ])


def test_bm25_finds_vietnamese_topic():
    hits = make().search("sắp xếp quick sort", k=2)
    assert hits[0].chunk.id == "b:0"


def test_kind_filter_and_course_exclusion():
    r = make()
    assert all(h.chunk.kind == src.rag.REFERENCE for h in r.search("algorithms", kinds={src.rag.REFERENCE}))
    assert not [h for h in r.search("sắp xếp", exclude_course="IT3011") if h.chunk.course == "IT3011"]


def test_dense_fusion_runs_with_fake_embedder():
    fake = lambda texts: [[float(len(t) % 7), 1.0] for t in texts]
    assert make().__class__(make().chunks, embed_fn=fake).search("sắp xếp", k=2)
