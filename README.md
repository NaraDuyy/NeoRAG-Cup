<div align="center">

<img src="docs/assets/neorag-banner.png" alt="ProPTIT × NeoRAG Cup 2026" width="720">

<p><a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"></a> <a href="LICENSE"><img src="https://img.shields.io/badge/code-MIT-green" alt="License MIT"></a></p>

</div>

# HybridPipeline: chatbot hỏi đáp cộng đồng S2B

Bài dự thi **NeoRAG Cup 2026** của **Hoàng Duy Anh**. Đây là một hệ thống RAG trả lời câu hỏi về CLB Lập Trình PTIT (ProPTIT), CLB Multimedia PTIT và CLB Nhà Sáng Tạo Game PTIT (GCC). Hệ thống chỉ dùng `claude-haiku-5-5` qua gateway của Ban tổ chức (BTC).

| Bộ đề | Điểm (raw score) | Retrieval | Generation |
|---|---:|---:|---:|
| **Private Test** (bài chính thức, v11) | **89.80** | 97.60 | 81.99 |
| Public Test (cao nhất, v10) | 90.19 | 95.40 | 84.98 |
| Baseline của BTC (Public) | 52.40 | 42.31 | 62.48 |

Code ở nhánh `main` chạy ra đúng bài Private 89.80 đã nộp.

## Kiến trúc

```
câu hỏi
  │
  ├─(1) Haiku viết lại câu hỏi: thêm dấu, mở viết tắt (gợi ý nghĩa cho viết tắt lạ),
  │     tách câu so sánh thành truy vấn con, nhận diện CLB được hỏi
  │
  ├─(2) Truy xuất lai:  BM25 trên văn bản bỏ dấu + bigram âm tiết, mở viết tắt (614 mục)
  │                     bge-m3 cho các truy vấn có dấu
  │                     bảng xếp hạng riêng của CLB được hỏi
  │                     → gộp RRF → 30 ứng viên
  │
  ├─(3) Haiku đọc 30 đoạn + 3 ví dụ văn phong lấy từ train, trả về
  │       "ID: <các đoạn đã dùng>"  và  "TRẢ LỜI: <1–3 câu>"
  │
  └─(4) retrieved_ids = tối đa 5 id LLM ghi
                        + phần còn lại xếp bằng cross-encoder bge-reranker-v2-m3 (chạy local) gộp RRF
```

Mỗi câu hỏi gọi Haiku 2 lần. Mọi kết quả gọi mô hình được cache trong `.cache/`, nên chạy lại cùng câu hỏi không tốn thêm ngân sách.

### Các quyết định chính

| Đặc điểm dữ liệu | Cách xử lý |
|---|---|
| Nhiều câu hỏi không dấu, viết tắt (train 21%, Public 48%) | BM25 so khớp trên âm tiết đã bỏ dấu; LLM viết lại câu có dấu trước khi đưa cho bge-m3. bge-m3 hiểu kém câu không dấu (nhóm này tụt từ 0.82 xuống 0.64) |
| `relevant_ids` tính cả đoạn tổng quan, bổ trợ | LLM ghi các đoạn đã dùng và cả đoạn bổ trợ vào dòng `ID:`; các id này được đưa lên đầu. Retrieval trên train tăng từ 0.884 lên 0.983, MRR đạt 1.0 |
| Đề test cần nhiều đoạn hơn train | Cho LLM thấy 30 ứng viên thay vì 12; phần sau các id được xếp bằng cross-encoder. Retrieval Public tăng từ 93.54 lên 95.40 |
| Đáp án ngắn, key fact chép nguyên văn | Prompt yêu cầu trả lời 1–3 câu và chép nguyên cụm từ của tài liệu |
| Câu trả lời được một phần, câu không có đáp án | Tách hai quy tắc: không có thông tin thì từ chối bằng một câu; có một phần thì viết "Tài liệu không ghi …, chỉ cho biết …" (v11). Generation Private tăng 3.24 |

Ablation đầy đủ, các thử nghiệm đã bỏ và lý do: [docs/solution.md](docs/solution.md), [docs/experiment_log.md](docs/experiment_log.md).

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows  (macOS, Linux: source .venv/bin/activate)
python scripts/install.py --local   # cài cả PyTorch + sentence-transformers cho cross-encoder
copy .env.example .env              # Windows  (macOS, Linux: cp .env.example .env)
```

Mở `.env` và điền `OPENAI_API_KEY` bằng key BTC cấp.

Lần chạy đầu tiên sẽ tải model `BAAI/bge-reranker-v2-m3` (khoảng 2.3 GB) từ Hugging Face. Trên CPU, mỗi câu hỏi mới mất khoảng 15 giây, chủ yếu do cross-encoder; câu đã có trong cache trả lời gần như ngay.

## Chạy

### Chatbot web (dùng cho Demo Product)

```bash
python app/server.py              # mở http://localhost:8000
python app/server.py --port 8080
```

Giao diện riêng ở [`app/web/`](app/web/), backend [`app/server.py`](app/server.py). Backend chỉ dùng thư viện chuẩn của Python, không cần cài thêm gói nào.

- **Hiện tiến độ từng bước** theo thời gian thực: viết lại câu hỏi → truy xuất → xếp lại → trả lời, kèm thời gian của mỗi bước. Server gửi tiến độ qua Server-Sent Events.
- Hiện **câu hỏi đã được hiểu lại** (thêm dấu, mở viết tắt).
- **Nguồn tham khảo**: 10 đoạn trong `retrieved_ids`, đoạn LLM đã trích có viền cam. Bấm vào một đoạn để đọc toàn văn.
- Câu hỏi gợi ý cho từng kiểu câu: không dấu, so sánh, có/không, cần nhiều đoạn, không có trong tài liệu.
- Lịch sử hội thoại lưu trong trình duyệt; có giao diện sáng/tối và dùng được trên điện thoại.
- Lỗi gateway (key hết hạn, rate limit, mất mạng) hiện thành thông báo dễ hiểu.

Backend gọi đúng `pipeline.run()`. Nó chỉ bọc các bước để báo tiến độ, nên câu trả lời và `retrieved_ids` giống hệt khi chạy `neorag run`. Muốn đổi pipeline thì đặt biến môi trường `NEORAG_PIPELINE="module.path:ClassName"`.

Chatbot Streamlit mẫu của BTC vẫn dùng được: `python -m streamlit run app/chat.py`.

### Tạo bài nộp và chấm điểm

```bash
neorag run private_test --pipeline neorag.pipelines.hybrid:HybridPipeline   # outputs/private_test_submission.zip
neorag run public_test  --pipeline neorag.pipelines.hybrid:HybridPipeline
neorag run train --pipeline neorag.pipelines.hybrid:HybridPipeline --no-judge   # chấm nhanh, không gọi judge
python scripts/eval_retrieval.py --offline        # chỉ BM25, không gọi API
python scripts/eval_final_retrieval.py --show     # retrieved_ids cuối cùng, chia nửa tune/hold
```

### Kiểm thử

```bash
python -m pytest
python -m ruff check . && python -m ruff format --check .
```

## Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| [`src/neorag/pipelines/hybrid.py`](src/neorag/pipelines/hybrid.py) | `HybridPipeline`: viết lại câu hỏi, truy xuất lai, prompt trả lời, ghép `retrieved_ids` |
| [`src/neorag/retrieval.py`](src/neorag/retrieval.py) | BM25 bỏ dấu, mở viết tắt, RRF |
| [`src/neorag/fewshot.py`](src/neorag/fewshot.py) | Chọn ví dụ văn phong từ train (leave-one-out) |
| [`src/neorag/embeddings.py`](src/neorag/embeddings.py) | Cross-encoder reranker chạy local, có cache |
| [`src/neorag/resources/abbreviations.tsv`](src/neorag/resources/abbreviations.tsv) | Từ điển 614 viết tắt |
| [`app/server.py`](app/server.py), [`app/web/`](app/web/) | Chatbot web |
| [`scripts/`](scripts/) | Script đo retrieval, chia tune/hold, so cặp, dựng từ điển viết tắt |
| [`eval/abbrev_dev.jsonl`](eval/abbrev_dev.jsonl) | 27 câu viết tắt tự soạn từ corpus, dùng để kiểm tra overfit |
| [`docs/solution.md`](docs/solution.md) | Kiến trúc, ablation, kết quả từng phiên bản |
| [`docs/experiment_log.md`](docs/experiment_log.md) | Nhật ký thử nghiệm: giả thuyết, cách đo, chi phí, kết luận, kể cả các bước sai |

## Dữ liệu bổ sung

Từ điển viết tắt được trích tự động bằng [`scripts/build_abbreviations.py`](scripts/build_abbreviations.py) từ hai dataset chuẩn hóa văn bản tiếng Việt. Hai dataset này không chứa thông tin về S2B; benchmark vẫn tính theo `id` của `data/corpus.jsonl`.

| Nguồn | License |
|---|---|
| [yammdd/vietnamese-error-correction-corpus](https://huggingface.co/datasets/yammdd/vietnamese-error-correction-corpus) | MIT |
| [ViLexNorm](https://github.com/ngxtnhi/ViLexNorm) (Nguyen et al., EACL 2024) | CC BY-NC-SA 4.0 |

## Về cuộc thi

NeoRAG Cup 2026 là cuộc thi nội bộ do Team AI – CLB Lập Trình PTIT tổ chức. Tài liệu gốc của BTC:

| Tài liệu | Nội dung |
|---|---|
| [Hướng dẫn sử dụng](docs/guide.md) | Cài đặt môi trường, cấu hình key, chạy baseline, tạo bài nộp |
| [Sử dụng gateway](docs/gateway.md) | Kết nối mô hình ngôn ngữ và embedding, ngân sách, giới hạn |
| [Quy định](docs/rules.md) | Mô hình được phép, các hành vi bị cấm |
| [Nộp bài](docs/submission.md), [aichallenge](docs/aichallenge.md) | Định dạng và quy trình nộp bài |
| [Đánh giá](docs/evaluation.md) | Định nghĩa metric, công thức điểm |
| [Dữ liệu](data/README.md) | Corpus, bộ câu hỏi, nguồn |

Tiêu chí chấm: Benchmark 40% (Private Test), Pitching 25%, Technical 20%, Demo Product 15%.

## Giấy phép

Mã nguồn phát hành theo [giấy phép MIT](LICENSE). Bộ dữ liệu chỉ dùng cho mục đích học tập trong khuôn khổ NeoRAG Cup 2026; bản quyền nội dung gốc thuộc về các CLB và các nguồn ghi trong `data/sources.csv`.
