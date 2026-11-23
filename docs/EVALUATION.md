# Đánh Giá Đề Cương & Hệ Thống Truy Hồi

Tài liệu này mô tả chi tiết phương pháp đánh giá của dự án `syllabus-gen`: đánh giá chất lượng truy hồi (Retrieval Evaluation), truy xuất nguồn gốc tài liệu (Citation Traceability), và đánh giá đề cương bằng rubric / LLM-judge.

---

## 1. Đánh Giá Truy Hồi (Retrieval Evaluation - `src/eval_retrieval.py`)

Hệ thống RAG sử dụng phương pháp tìm kiếm lai (Hybrid: BM25 + Dense Cosine Brute-force kết hợp qua Reciprocal Rank Fusion k=60).

### Các chỉ số đo lường

1. **Hits@k (k = 1, 3, 5, 10):**
   - Định nghĩa: Tỉ lệ các truy vấn mà trong đó có ít nhất một đoạn trích đúng (ground truth) xuất hiện trong top $k$ kết quả trả về.
   - Công thức: $\text{Hits@}k = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}(\text{rank}(q) \le k)$.
2. **Recall@k:**
   - Định nghĩa: Tỉ lệ tài liệu đúng được truy hồi thành công trong top $k$ trên tổng số tài liệu đúng của truy vấn đó.
3. **MRR (Mean Reciprocal Rank):**
   - Định nghĩa: Giá trị trung bình của nghịch đảo thứ hạng của kết quả đúng đầu tiên.
   - Công thức: $\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_1(q)}$, với $\text{rank}_1(q)$ là thứ hạng của tài liệu đúng đầu tiên (bằng 0 nếu không tìm thấy trong top $k$).

### Chống rò rỉ dữ liệu (Data Leakage)
Khi thực hiện benchmark đánh giá truy hồi:
- Đề cương mẫu cùng mã môn học (ví dụ `syllabus_IT3011_mau.docx`) luôn được loại trừ khỏi ngữ cảnh (`exclude_course`) để tránh việc retriever chỉ đơn giản là chép lại chính xác đoạn văn từ đề cương đích.

### Lưu ý về cỡ mẫu ($n$)
> **CẢNH BÁO:** Nếu số lượng câu hỏi gold queries $n < 30$, các chỉ số định lượng có phương sai lớn. **Kết luận chỉ mang tính định tính/thử nghiệm sơ bộ**, không được tuyên bố là kết quả khẳng định về chất lượng sản phẩm.

---

## 2. Truy Xuất Nguồn Gốc Tài Liệu (Citation Traceability - M6)

### Vấn đề
LLM thường có xu hướng "bịa" (hallucinate) tên sách, tác giả, nhà xuất bản hoặc năm phát hành nếu không được kiểm soát chặt chẽ.

### Cơ chế kiểm soát
- Hệ thống chỉ cho phép mục **Tài liệu tham khảo** (References) được chọn từ các chunk có `kind = reference` trong corpus của chính môn học đó (hoặc `_shared/`).
- **Citation Traceability Rate:**
  $$\text{Traceability} = \frac{\text{Số tài liệu tham khảo đối chiếu được về corpus}}{\text{Tổng số tài liệu tham khảo trong đề cương}}$$
- Nếu tỉ lệ này $< 100\%$, rubric kích hoạt lỗi nghiêm trọng `ref_not_in_corpus` (Gate failure) và đề cương không được thông qua tự động.

---

## 3. Rubric Xác Định (Deterministic Rubric)

Gồm 9 tiêu chí tính toán 100% bằng code:
- **S1 (Cổng chặn):** Tổng tỉ trọng đánh giá = 100%, các tỉ trọng > 0.
- **S2 (Cổng chặn):** Ma trận đánh giá (blueprint) phủ đầy đủ mọi CLO, không có ô trống phương pháp.
- **S3 (Cổng chặn):** Số chương liên tục từ 1..N, mỗi chương có mục con và gắn ít nhất một CLO; cảnh báo nếu >50% chương gắn toàn bộ CLO.
- **S4 (Cổng chặn):** Mọi liên kết PLO nằm trong danh mục PLO được duyệt của CTĐT.
- **S5:** Đầy đủ thông tin song ngữ Việt - Anh, không để trống hoặc chép nguyên văn vi sang en.
- **B1 (Cổng chặn):** Mọi CLO bắt đầu bằng động từ hành động đo lường được theo Bloom A&K 2001, hai ngôn ngữ đồng mức nhận thức.
- **B2:** Tiến triển nhận thức giữa các CLO (có ít nhất một CLO đạt mức Vận dụng trở lên, không phẳng lì cùng một mức).
- **B3:** Phương pháp đánh giá có trần nhận thức đủ bao quát mức Bloom của CLO (ví dụ: CLO Sáng tạo không thể chỉ đo bằng trắc nghiệm).
- **G1 (Cổng chặn):** 100% tài liệu có nguồn gốc từ corpus.

---

## 4. Đánh Giá Bằng LLM (LLM-as-Judge: L1 - L3)

Dành cho các khía cạnh sư phạm không thể kiểm tra bằng regex/code:
- **L1 (Nội dung chương):** Tính đầy đủ, đúng đắn, bám sát ngữ cảnh tài liệu và mô tả học phần.
- **L2 (Chất lượng CLO):** Tính cụ thể, khả năng đo lường, mức độ phù hợp với số tín chỉ.
- **L3 (Mạch sư phạm):** Thứ tự bố trí các chương, tính khả thi trong quỹ thời gian (LT/BT/TN).

### Các hạn chế đã biết của LLM-as-judge:
1. **Thiên vị tự ưu tiên (Self-enhancement bias):** Nếu dùng cùng một họ mô hình để vừa sinh vừa chấm, điểm số thường bị thổi phồng. Khắc phục: đặt `LLM_JUDGE_MODEL` khác họ với `LLM_MODEL`.
2. **Thiên vị độ dài và văn phong:** Mô hình có xu hướng cho điểm cao đối với các phản hồi dài, hoa mỹ. Khắc phục: Rubric yêu cầu LLM trích xuất `evidence` cụ thể trước khi chấm điểm 1-4 theo neo chuẩn (anchors).
3. **Biến thiên ngẫu nhiên (Variance):** Cần đặt `temperature = 0.0`. Dù vậy, kết quả vẫn cần được đối chiếu với điểm số của ít nhất 2 giảng viên con người để tính hệ số đồng thuận (Cohen's Kappa).
