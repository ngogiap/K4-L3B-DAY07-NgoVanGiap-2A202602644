# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Ngô Văn Giáp
**Nhóm:** 6h50
**Ngày:** 20/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> *Viết 1-2 câu:* Nghĩa là hai vector biểu diễn văn bản hướng về cùng một phía trong không gian nhiều chiều. Điều này thể hiện rằng hai đoạn văn bản có ý nghĩa (ngữ nghĩa) rất giống nhau hoặc liên quan chặt chẽ với nhau, bất kể độ dài văn bản.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Chính sách đổi trả hàng trên Shopee áp dụng trong bao lâu?"
- Câu B: "Thời gian cho phép hoàn trả sản phẩm của Shopee là mấy ngày?"
- Tại sao tương đồng: Mặc dù dùng từ vựng khác nhau ("đổi trả hàng" vs "hoàn trả sản phẩm", "bao lâu" vs "mấy ngày"), nhưng ý nghĩa và mục tiêu hỏi về thời hạn trả hàng là giống hệt nhau.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Apple là một loại trái cây rất tốt cho sức khỏe."
- Câu B: "Apple vừa ra mắt dòng điện thoại iPhone mới."
- Tại sao khác: Từ "Apple" xuất hiện ở cả hai câu, nhưng ngữ cảnh hoàn toàn khác biệt (một bên là quả táo, một bên là công ty công nghệ), do đó vector ngữ nghĩa của chúng sẽ nằm cách xa nhau.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> *Viết 1-2 câu:* Vì độ tương tự cosine chỉ quan tâm đến góc (hướng) giữa hai vector, bỏ qua độ lớn (magnitude) của vector. Nhờ đó, nó đánh giá độ tương đồng ngữ nghĩa chính xác hơn mà không bị ảnh hưởng bởi sự khác biệt về độ dài của hai văn bản.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* Bước tiến (stride) của mỗi chunk = chunk_size - overlap = 500 - 50 = 450. Chunk đầu tiên lấy 500 ký tự, phần còn lại là 9500 ký tự. Số chunk tiếp theo cần = ceil(9500 / 450) = ceil(21.11) = 22. Tổng cộng = 1 + 22 = 23 chunks. (Hoặc công thức: ceil((10000 - 50) / 450) = 23).
> *Đáp án:* 23 chunks.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> *Viết 1-2 câu:* Với overlap=100, stride=400, số lượng chunk = ceil((10000 - 100) / 400) = 25 chunks (số chunk tăng lên). Cần độ chồng chéo cao hơn để đảm bảo các thông tin ngữ cảnh quan trọng không bị cắt đứt đột ngột ở biên giữa hai chunk, giúp câu văn giữ được ý nghĩa trọn vẹn.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> *Viết 2-3 câu:* Dùng biểu thức chính quy (regex) `re.split(r'(?<=[.!?])(?:\s|\n)', text)` để chia văn bản tại các dấu kết thúc câu (dùng lookbehind `(?<=...)` để giữ lại dấu câu). Trường hợp ngoại lệ được xử lý bằng cách dùng `.strip()` để loại bỏ các chuỗi rỗng vô nghĩa trước khi nhóm (group) nhiều câu lại thành một chunk theo giới hạn `max_sentences_per_chunk`.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> *Viết 2-3 câu:* Thuật toán đệ quy thử cắt chuỗi theo separator ưu tiên cao nhất trước (vd: `\n\n`), rồi gom (merge) các mẩu nhỏ lại; nếu mẩu gom vẫn vượt quá `chunk_size` thì đệ quy cắt tiếp bằng separator ưu tiên thấp hơn (như `\n`, ` `). Base case (trường hợp cơ sở) là khi chuỗi hiện tại đã ngắn hơn hoặc bằng `chunk_size`, hoặc khi hết danh sách separator thì bắt buộc phải cắt cứng theo đúng số ký tự `chunk_size`.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> *Viết 2-3 câu:* Lưu trữ bằng cách sinh embedding cho nội dung từng đoạn và đóng gói thành một dictionary (chứa `id`, `content`, `embedding`, `metadata`) rồi thêm vào mảng `self._store`. Hàm `search` duyệt toàn bộ `_store`, tính độ tương tự giữa vector truy vấn và vector của từng bản ghi bằng tích vô hướng (dot product) và sắp xếp giảm dần để lấy top k.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> *Viết 2-3 câu:* Hàm thực hiện tiền lọc (pre-filtering) bằng cách duyệt qua kho lưu trữ và tạo một list con (filtered_records) chỉ chứa các chunk thỏa mãn toàn bộ cặp key-value trong `metadata_filter` rồi mới thực hiện `search`. Xóa tài liệu được thực hiện bằng cách khởi tạo lại (list comprehension) mảng `_store` chỉ chứa các chunk có `doc_id` khác với id cần xóa.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> *Viết 2-3 câu:* Prompt được cấu trúc thành các phần rõ ràng: Chỉ thị hệ thống, khối "Context:", sau đó là khối "Question:", và kết thúc bằng "Answer: " để mồi (prime) cho mô hình sinh chữ. Ngữ cảnh (context) được inject bằng cách gọi hàm search để lấy danh sách top k chunk, đánh số thứ tự `[1]`, `[2]` cho từng chunk, nối (join) chúng lại bằng dấu xuống dòng và chèn trực tiếp vào chuỗi (string formatting) prompt.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts =============================
platform win32 -- Python 3.13.1, pytest-9.1.1, pluggy-1.6.0 -- D:\HOC TAP\PYTHON\Aithucchien\Day7\K4-L3B-DAY07-NgoVanGiap-2A202602644\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\HOC TAP\PYTHON\Aithucchien\Day7\K4-L3B-DAY07-NgoVanGiap-2A202602644
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.06s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "hello world" | "hello world" | Cao | 1.0000 | Đúng |
| 2 | "Thời gian hoàn trả là bao lâu?" | "Bao nhiêu ngày thì được trả hàng?" | Cao | 0.8679 (Cao) | Đúng |
| 3 | "Tôi thích ăn táo." | "Apple vừa ra mắt điện thoại." | Thấp | 0.6826 (Thấp) | Đúng |
| 4 | "Chính sách đổi trả hàng." | "Chính sách đổi trả hàng!" | Cao | 0.8970 (Cao) | Đúng |
| 5 | "Mèo thích ăn cá." | "Hệ điều hành Windows." | Thấp | 0.5798 (Thấp) | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> *Viết 2-3 câu:* Bất ngờ nhất là cặp số 2: mặc dù hai câu sử dụng bộ từ vựng hoàn toàn khác nhau ("thời gian" vs "bao nhiêu ngày", "hoàn trả" vs "trả hàng"), điểm tương tự vẫn rất cao (0.8679). Điều này chứng minh sức mạnh của mô hình ngôn ngữ (Gemini): vector embeddings biểu diễn thành công **ngữ nghĩa (semantic meaning)** của câu thay vì chỉ so sánh hiện tượng trùng lặp từ vựng (lexical matching).

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Người mua có bao nhiêu ngày để gửi yêu cầu trả hàng/hoàn tiền kể từ khi đơn hàng giao thành công? | `quy-dinh-chung-tra-hang-hoan-tien#3`: "Đối với các đơn hàng khác: 15 ngày kể từ lúc đơn hàng được cập nhật..." | 0.8411 | Có | 15 ngày kể từ lúc đơn hàng được cập nhật giao hàng thành công. |
| 2 | Khi hệ thống ghi nhận đã trả hàng thành công nhưng Shop chưa nhận được hàng, Người Bán phải phản hồi Shopee trong bao lâu? | `quan-ly-don-tra-hang-hoan-tien-seller#6` (có lọc `audience: seller`): "Hệ thống ghi nhận đã trả hàng... Hạn phản hồi: Sau 2 ngày..." | 0.8713 | Có | Cần phản hồi trong vòng 2 ngày kể từ ngày hệ thống cập nhật. |
| 3 | Nếu người mua nhận hoàn tiền qua Ví ShopeePay, sau khi Shopee chấp nhận hoàn tiền thì mất bao lâu để nhận được tiền? | `thoi-gian-nhan-tien-hoan#0`: "[Trả hàng/ Hoàn tiền] Thời gian nhận tiền hoàn và cách kiểm tra..." | 0.8240 | Có | Trong vòng 24 giờ kể từ khi Shopee chấp nhận hoàn tiền. |
| 4 | Khi đóng gói hàng hoàn trả, người mua có được dán/viết thông tin trả hàng lên hộp của nhà sản xuất không? | `cach-dong-goi-hang-hoan-tra#6`: "Lưu ý: Không dán, viết lên hộp của nhà sản xuất..." | 0.8192 | Có | Không được dán hay viết thông tin lên hộp của nhà sản xuất. |
| 5 | Sau khi người mua gửi yêu cầu Trả hàng/Hoàn tiền, Shopee phản hồi kết quả xử lý trong khoảng thời gian nào? | `quy-dinh-chung-tra-hang-hoan-tien#4`: "Yêu cầu Trả hàng/Hoàn tiền của bạn sẽ được phản hồi trong vòng 3-5 ngày làm việc..." | 0.8830 | Có | Phản hồi kết quả xử lý trong vòng 3-5 ngày làm việc. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5 (Nhờ mô hình Gemini hiểu được ngữ nghĩa).

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> *Viết 2-3 câu:* Qua phần demo, mình nhận thấy bộ lọc `metadata_filter` rất quan trọng để loại bỏ nhiễu (ví dụ lọc `seller` cho câu 2). Ngoài ra, khi chuyển sang dùng API Gemini thật, điểm số truy xuất tăng vọt lên 5/5 so với mức 0-1/5 của Mock Embedder; điều này chứng tỏ "nút thắt cổ chai" lớn nhất trong RAG là chất lượng của Embedding Model chứ không chỉ là chiến lược Chunking.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
