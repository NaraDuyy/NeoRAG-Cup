# Giải pháp: HybridPipeline

Pipeline nằm trong [`src/neorag/pipelines/hybrid.py`](../src/neorag/pipelines/hybrid.py), phần truy xuất lexical trong [`src/neorag/retrieval.py`](../src/neorag/retrieval.py). Mỗi câu hỏi gọi `claude-haiku-5-5` hai lần; mọi kết quả được cache trong `.cache/hybrid.sqlite`.

## 1. Kiến trúc

```
câu hỏi ──► (1) Viết lại câu hỏi (LLM) ──► truy vấn gốc + truy vấn đã chuẩn hóa + truy vấn con, CLB được hỏi
                                                 │
            (2) Truy xuất                        ▼
                BM25 trên văn bản bỏ dấu + bigram âm tiết, mở rộng viết tắt   ─┐
                Dense bge-m3 (gateway)                                         ├─► RRF ─► xếp hạng toàn corpus
                Bảng xếp hạng chỉ gồm đoạn của CLB được hỏi (trọng số 0.5)    ─┘
                                                 │
            (3) Sinh câu trả lời (LLM)           ▼ 12 đoạn đầu
                LLM trả về  ID: <đoạn đã dùng>  và  TRẢ LỜI: <câu trả lời ngắn>
                                                 │
retrieved_ids = đoạn LLM đã dùng + phần còn lại theo RRF (đủ 10)    answer = câu trả lời
```

## 2. Các quyết định kỹ thuật

| Vấn đề trong dữ liệu | Cách xử lý |
|---|---|
| ~20% câu hỏi viết không dấu, viết tắt (`clb mm`, `tv`, `nk`, `t10`) | BM25 so khớp trên âm tiết đã bỏ dấu; `expand_query` mở viết tắt và tên CLB không chính thức (`clb game` → GCC); LLM viết lại câu hỏi có dấu cho bge-m3 |
| Corpus nhỏ (118 đoạn), mỗi đoạn tự đủ ngữ cảnh | Không chunking; index nguyên đoạn kèm `club` và `topic` |
| Câu hỏi so sánh, nhiều đoạn | LLM tách thành truy vấn con, mỗi truy vấn con đóng góp một bảng xếp hạng vào RRF |
| Câu hỏi thường nhắm một CLB | Thêm bảng xếp hạng chỉ gồm đoạn của CLB được hỏi (và Cộng đồng S2B) vào RRF |
| Điểm retrieval tính MRR@5, NDCG@5 | LLM chỉ ra đoạn đã dùng để trả lời; các đoạn đó được đưa lên đầu `retrieved_ids` (thay cho reranker riêng, tiết kiệm một lần gọi) |
| ROUGE-L phạt câu dài, String Presence so khớp chuỗi | Prompt yêu cầu 1–3 câu, nhắc lại chủ thể, giữ nguyên ngày tháng, số liệu, tên riêng |
| Câu hỏi không có đáp án (Abstention) | Prompt yêu cầu nói rõ "Tài liệu hiện có không có thông tin về ..." và không đưa thông tin như thể là câu trả lời |
| Ngân sách 5 USD | 2 lần gọi LLM mỗi câu, cache SQLite, chấm retrieval riêng bằng `scripts/eval_retrieval.py` |

## 3. Kết quả

Retrieval trên tập train (122 câu có `relevant_ids`), trung bình của Recall@5, MRR@5, NDCG@5:

| Cấu hình | Recall@5 | MRR@5 | NDCG@5 | Trung bình |
|---|---:|---:|---:|---:|
| Baseline (bge-m3 cosine) | 0.782 | 0.692 | 0.691 | 0.722 |
| BM25 bỏ dấu | 0.790 | 0.780 | 0.736 | 0.769 |
| + mở rộng viết tắt | 0.798 | 0.790 | 0.742 | 0.776 |
| + ưu tiên CLB được hỏi (`--offline`) | 0.837 | 0.805 | 0.771 | 0.805 |
| + bge-m3 cho mọi truy vấn | 0.835 | 0.782 | 0.764 | 0.794 |
| + bge-m3 chỉ cho truy vấn có dấu (`--no-rewrite`) | 0.867 | 0.830 | 0.802 | 0.833 |
| + viết lại câu hỏi bằng LLM | 0.928 | 0.873 | 0.851 | 0.884 |
| + LLM chọn đoạn đã dùng (pipeline đầy đủ) | 0.971 | 1.000 | 0.977 | **0.983** |

bge-m3 hiểu kém câu viết không dấu (nhóm "không dấu/viết tắt" tụt từ 0.821 xuống 0.637 khi embed nguyên câu gốc), nên dense chỉ dùng các truy vấn có dấu; câu không dấu được LLM viết lại có dấu trước khi embed.

Generation trên train (`--no-judge`):

| Prompt | String Presence | ROUGE-L | BLEU-4 | Độ dài median | Latency median |
|---|---:|---:|---:|---:|---:|
| v1: 1-3 câu, nói rõ phần tài liệu không nêu | 0.895 | 0.626 | 0.468 | 272 ký tự | 2.4 s |
| v2: dừng khi đã trả lời đủ, không rào đón | 0.894 | **0.644** | **0.503** | 208 ký tự | 2.2 s |

Đáp án chuẩn có độ dài median 196 ký tự. Bản v1 hay thêm câu "Tài liệu không nêu thêm ..." kể cả khi đã trả lời đủ, làm giảm ROUGE-L. Hai câu train-0001 và train-0019 nằm trong ví dụ của prompt nên điểm train hơi lạc quan.

Public Test: baseline 52.40, ngưỡng 53.40. Điểm của HybridPipeline: _chưa nộp_.

## 4. Cách chạy

```bash
# Chấm riêng retrieval, chia theo kiểu câu hỏi
python scripts/eval_retrieval.py --offline       # chỉ BM25, không gọi API
python scripts/eval_retrieval.py --no-rewrite    # thêm bge-m3
python scripts/eval_retrieval.py                 # thêm bước viết lại câu hỏi

# Chạy pipeline đầy đủ trên train, chấm nhanh không dùng judge
neorag run train --pipeline neorag.pipelines.hybrid:HybridPipeline --no-judge

# Chấm đầy đủ bằng judge (tốn ngân sách, nên chạy ít lần)
neorag evaluate train

# Tạo file nộp Public Test / Private Test
neorag run public_test --pipeline neorag.pipelines.hybrid:HybridPipeline
neorag run private_test --pipeline neorag.pipelines.hybrid:HybridPipeline

# Chatbot (mặc định dùng HybridPipeline)
python -m streamlit run app/chat.py
```
