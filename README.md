# Sinh Đề Cương Học Phần Bằng LLM + RAG + Bloom + Rubric

Hệ thống sinh đề cương chi tiết học phần tự động từ tài liệu môn học (PDF giáo trình, chuẩn đầu ra PLO, quy chế) kết hợp kiểm soát chất lượng bằng thang nhận thức Bloom và hệ thống Rubric kiểm tra tính đúng đắn.

---

## 1. Bài Toán & Giải Pháp

- **Vấn đề:** Giảng viên thường mất nhiều thời gian biên soạn đề cương chi tiết theo chuẩn kiểm định (AUN-QA, ABET), dễ mắc lỗi không tương thích giữa mức độ CLO và phương pháp đánh giá (constructive alignment), hoặc LLM tự sinh (bịa) tài liệu tham khảo không có trong thư viện trường.
- **Giải pháp:**
  - **Inference-time RAG:** Truy hồi nội dung từ kho tài liệu PDF/văn bản thực tế của bộ môn; không fine-tune.
  - **Kiểm soát Bloom bằng code:** Tự động gán mức nhận thức (Anderson & Krathwohl 2001) và kiểm tra "trần" nhận thức của phương pháp đánh giá hoàn toàn bằng thuật toán xác định.
  - **Tài liệu tham khảo thực 100%:** Chỉ cho phép chọn tài liệu theo ID có trong corpus; không tự bịa thông tin.

---

## 2. Kiến Trúc Hệ Thống

```
inputs/<MÃ>.json + data/*.pdf
        │
        ▼
   pdf_ingest.py ──▶ corpus/<MÃ>/books/ (gợi ý chương từ bookmark/tên file)
        │
        ▼
  RAG Retriever ──▶ A: Mô tả + CLO  ──▶ Validator (rubric.py) ──▶ Retry nếu lỗi
  (BM25 + Dense)        │
                        ▼
                    B: Các chương   ──▶ Truy hồi theo chương + post-filter
                        │
                        ▼
                    C: Đánh giá     ──▶ Tự dựng Blueprint ma trận bằng code
                        │
                        ▼
                    D: Tài liệu     ──▶ Chỉ CHỌN ID từ corpus (kind=reference)
                        │
                        ▼
                  Syllabus JSON ──▶ render.py ──▶ Đề cương Word (.docx)
```

---

## 3. Cài Đặt & Sử Dụng

### Cài đặt môi trường
```bash
python -m venv .venv
source .venv/bin/activate  # Hoặc .venv\Scripts\activate trên Windows
pip install -r requirements.txt  # Hoặc pip install pypdf docxtpl openai python-dotenv pydantic pytest
```

Tạo file `.env`:
```env
LLM_API_KEY=your_api_key_here
LLM_MODEL=       
LLM_JUDGE_MODEL=   # model chấm điểm (nên khác model sinh)
LLM_EMBED_MODEL=   # tuỳ chọn: bật dense hybrid retrieval
```

### Chạy hệ thống
```bash
# 1. Sinh đề cương và nạp PDF giáo trình của môn:
python src/main.py inputs/it3011.json --pdfs path/to/chapter_pdfs/ --corpus corpus --evaluate

# 2. Đánh giá chất lượng truy hồi RAG trên tập benchmark gold:
python src/eval_retrieval.py --corpus corpus --gold eval/gold/queries.jsonl --syllabus outputs/IT3011.json

# 3. Xuất đề cương sang Word (yêu cầu có templates/syllabus_template.docx):
python src/render.py outputs/IT3011.json templates/syllabus_template.docx outputs/IT3011.docx

# 4. Chạy toàn bộ kiểm thử tự động:
pytest -v
```

---

## 4. Cấu Trúc Thư Mục

```
syllabus-gen/
├── corpus/                      # Kho tài liệu RAG
│   ├── _shared/                 # Chuẩn đầu ra PLO, quy chế chung
│   │   ├── plo/
│   │   └── reg/
│   ├── _syllabus/               # Đề cương cũ các môn khác làm tham khảo
│   └── <MÃ_MÔN>/               # Tài liệu riêng từng môn (tích lũy dần)
│       ├── ref/                 # Danh mục tài liệu tham khảo (mỗi dòng 1 cuốn)
│       ├── toc/                 # Mục lục giáo trình
│       └── books/               # PDF giáo trình theo chương (kèm metadata)
├── eval/
│   └── gold/
│       └── queries.jsonl        # Tập query chuẩn do con người gán nhãn
├── docs/
│   ├── BLOOM.md                 # Hướng dẫn động từ Bloom & CLO mẫu
│   └── EVALUATION.md            # Phương pháp đo Hits@k, MRR, Traceability
├── inputs/                      # Cấu hình môn học điền tay (inputs/it3011.json)
├── outputs/                     # Đề cương JSON và báo cáo đánh giá
├── templates/                   # Mẫu Word (.docx) do nhà trường cấp
├── src/
│   ├── bloom.py                 # Từ điển Bloom và luật trần nhận thức
│   ├── bloom_extra.py           # Tập từ mơ hồ (VAGUE) - cảnh báo, không chặn
│   ├── eval_retrieval.py        # Đo Hits@k, Recall@k, MRR, Citation Traceability
│   ├── evaluate.py              # Đánh giá đề cương bằng Rubric + LLM-judge
│   ├── generate.py              # Điều phối 4 giai đoạn sinh đề cương
│   ├── ingest_docx.py           # Đọc đề cương mẫu docx thành Syllabus object
│   ├── judge.py                 # LLM-as-judge chấm các tiêu chí định tính
│   ├── llm.py                   # Lớp bọc gọi LLM API tương thích OpenAI
│   ├── main.py                  # Điểm khởi chạy CLI chính
│   ├── pdf_ingest.py            # Nạp PDF, phát hiện chương từ bookmark/tên file
│   ├── prompts.py               # Gom tập trung prompts 4 giai đoạn
│   ├── rag.py                   # Retriever BM25 + Cosine Fusion brute-force
│   ├── render.py                # Điền dữ liệu vào template docx
│   ├── rubric.py                # 9 tiêu chí rubric xác định (S1-S5, B1-B3, G1)
│   └── schema.py                # Khai báo schema Pydantic song ngữ
└── tests/                       # Bộ kiểm thử tự động pytest
```

---

## 5. Ước Tính Chi Phí & Số Lần Gọi LLM (Free-Tier Friendly)

Hệ thống được thiết kế tối ưu cho gói miễn phí (Free Tier như Google AI Studio, Groq, hoặc GitHub Models):

| Giai đoạn | Số lần gọi LLM | Mục đích | Cơ chế fallback nếu lỗi |
|---|:---:|---|---|
| **A: Mô tả & CLO** | 1 | Sinh mô tả môn học và danh sách CLO | Tối đa 3 lần retry kèm phản hồi lỗi cụ thể |
| **B: Các chương** | 1 | Sinh mục con (topics) và hoạt động dạy học | Cấu trúc chương lấy từ PDF, không để LLM bịa |
| **C: Đánh giá** | 1 | Thiết kế thành phần và phương pháp đánh giá | Tối đa 3 lần retry nếu lệch tổng 100% |
| **D: Tài liệu** | 1 | Chọn ID tài liệu phù hợp từ corpus | Chỉ chọn từ danh sách có sẵn; 0 nếu trống |
| **LLM-Judge** *(tuỳ chọn)*| 1 | Hội đồng AI chấm 3 tiêu chí L1-L3 | Chỉ chạy khi có cờ `--judge` |

- **Tổng số lần gọi LLM trong điều kiện bình thường:** **4 lần** (5 lần nếu bật judge).
- **Trường hợp xấu nhất (mọi giai đoạn đều retry tối đa):** ~12 - 14 lần gọi.
- **Tiêu thụ token trung bình:** ~3.000 - 6.000 prompt tokens và ~1.500 completion tokens cho toàn bộ 1 môn học.

---

## 6. Trạng Thái Hiện Tại (Đã Kiểm Chứng vs Chưa Kiểm Chứng)

### Đã kiểm chứng thực tế (100% pass unit tests)
- [x] Parser PDF tự động, trích xuất cấu trúc chương từ bookmarks (outline) và tên file.
- [x] Phát hiện file scan (trang text rỗng) và báo cáo tình trạng PDF.
- [x] Phân loại corpus theo thư mục con `corpus/<MÃ>/{ref,toc,books}` và phân lập không rò rỉ giữa các môn.
- [x] RAG BM25 + Cosine Brute-force kết hợp Reciprocal Rank Fusion ($k=60$).
- [x] Gán metadata `chapter` và `source_pdf` cho các đoạn trích.
- [x] Bộ cảnh báo từ mơ hồ `bloom_extra.py` (cảnh báo: "biết", "nắm được", "appreciate", "know").
- [x] Tập trung hóa prompt 4 giai đoạn vào `prompts.py`.
- [x] 9 tiêu chí Rubric xác định (S1-S5, B1-B3, G1) và parser đề cương mẫu docx IT3011.
- [x] Đo lường Hits@k, Recall@k, MRR và Citation Traceability (`eval_retrieval.py`).
- [x] Bộ test pipeline mô phỏng LLM giả (Mock) kiểm tra cơ chế tự sửa lỗi khi vi phạm rubric.

### Chưa kiểm chứng / Cần dữ liệu thực
- [ ] Chạy thực tế với API key thật trên một tập môn học lớn (hiện tại test hoàn toàn bằng LLM Mock để tiết kiệm quota).
- [ ] Render Word qua `render.py` với file mẫu thật (đang chờ file mẫu từ người dùng tại `templates/`).
- [ ] Đánh dấu tập gold queries lớn ($n \ge 30$) trên các file PDF bài giảng thực tế để có kết luận thống kê mạnh mẽ.
