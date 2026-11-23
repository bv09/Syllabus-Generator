import src.bloom
from src.bloom import Cog
from src.schema import CLO, Bi


def clo(vi, en, code="CLO1"):
    return CLO(code=code, description=Bi(vi=vi, en=en), linked_plos=["PLO 01"], level="R")


def test_basic_levels():
    assert src.bloom.classify("Áp dụng cấu trúc dữ liệu để giải quyết bài toán", "vi").level == Cog.APPLY
    assert src.bloom.classify("Có khả năng đề xuất giải pháp tối ưu", "vi").level == Cog.CREATE
    assert src.bloom.classify("Apply data structures to solve problems", "en").level == Cog.APPLY


def test_compound_takes_highest_and_reports_spread():
    r = src.bloom.classify("Hiểu và đánh giá được hiệu quả của giải pháp", "vi")
    assert r.level == Cog.EVALUATE and r.spread == 3


def test_noun_false_positive_guard():
    # "linked lists" / "tính toán" là danh từ, không được coi là động từ
    assert src.bloom.classify("Understand linked lists and stacks", "en").level == Cog.UNDERSTAND
    r = src.bloom.classify("Nhận diện và hiểu rõ các yêu cầu tính toán trong hệ thống", "vi")
    assert r.level == Cog.UNDERSTAND


def test_no_verb_returns_none():
    assert src.bloom.classify("Cấu trúc dữ liệu và giải thuật", "vi") is None


def test_method_ceiling():
    assert src.bloom.method_ceiling("Trắc nghiệm (Quiz)") == Cog.APPLY
    assert src.bloom.method_ceiling("Đồ án") == Cog.CREATE
    assert src.bloom.method_ceiling("???") is None
