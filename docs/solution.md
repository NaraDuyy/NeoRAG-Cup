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
| ~20% câu hỏi viết không dấu, viết tắt (`clb mm`, `tv`, `nk`, `t10`) | BM25 so khớp trên âm tiết đã bỏ dấu; `expand_query` mở viết tắt (từ điển 614 mục trích từ dataset ngoài) và tên CLB không chính thức (`clb game` → GCC); LLM viết lại câu hỏi có dấu cho bge-m3, kèm gợi ý nghĩa của các viết tắt lạ |
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

Đáp án chuẩn có độ dài median 196 ký tự. Bản v1 hay thêm câu "Tài liệu không nêu thêm ..." kể cả khi đã trả lời đủ, làm giảm ROUGE-L. Hai câu train-0001 và train-0019 nằm trong ví dụ của prompt v1/v2 nên điểm train của hai bản này hơi lạc quan.

### Tinh chỉnh văn phong câu trả lời (v3–v5)

So với đáp án chuẩn, câu trả lời v2 lệch văn phong ở mấy điểm:
- câu có/không ít khi mở đầu bằng "Có." / "Không.";
- câu so sánh không nêu các đối tượng theo thứ tự câu hỏi;
- hay diễn đạt lại cụm từ của tài liệu;
- đôi khi tự suy đoán thêm.

Để tránh overfit, [`scripts/eval_split.py`](../scripts/eval_split.py) chia train thành hai nửa cố định, cân theo kiểu câu hỏi: `tune` để đọc lỗi và `hold` chỉ để chấm. Hai câu train-0001 và train-0019 bị bỏ khỏi cả hai nửa. Hai ví dụ trong prompt được thay bằng ví dụ tự viết từ những đoạn corpus mà train và Public không hỏi tới (PROPTIT Marathon, trò chơi trên itch.io của GCC).

| Prompt | Thay đổi so với bản trước | hold ROUGE-L / SP | tune ROUGE-L / SP | Judge trên cả train: Generation / Abstention |
|---|---|---:|---:|---:|
| v2 | | 0.638 / 0.884 | 0.644 / 0.901 | 0.8556 / 1.000 |
| v3 | mở đầu "Có."/"Không.", "Theo <nguồn>", so sánh theo thứ tự, câu thiếu dữ liệu nêu phần có trước, không suy đoán | 0.645 / 0.914 | 0.682 / 0.878 | |
| v4 | bỏ "Theo <nguồn>" (LLM tự đặt nguồn, câu dài ra), dùng lại đúng cụm từ của tài liệu | 0.654 / 0.938 | 0.672 / 0.925 | 0.8532 / **0.909** |
| v5 | câu không có đáp án phải mở đầu bằng "Tài liệu hiện có không có thông tin về ..." | 0.638 / 0.932 | 0.686 / 0.925 | **0.8640** / 1.000 |

Benchmark trên cả train tăng từ 0.9191 (v2) lên 0.9230 (v5). Tính riêng nửa hold, Generation của judge tăng từ 0.8493 lên 0.8538.
- **Bài học từ v4.** Khi câu không có đáp án mở đầu bằng điều tài liệu có ("Tài liệu chỉ cho biết ..."), judge Abstention không coi đó là từ chối trả lời (train-0049). Đề Public chỉ có 3 câu không có đáp án, nên sai 1 câu mất khoảng 2.8 điểm tổng. Vì vậy v5 bắt buộc câu từ chối phải đứng đầu.
- **Mức chắc chắn của v5.** String Presence tăng rõ trên cả hai nửa. Nhưng ROUGE-L trên nửa hold không đổi, nên chưa đạt ngưỡng đề ra (hold ROUGE-L +0.02). Mức tăng tổng (khoảng +0.2 đến +0.4 điểm) nằm trong biên nhiễu của judge.

Public Test: baseline 52.40, ngưỡng 53.40. HybridPipeline với prompt v2: **88.64**.

### Từ điển viết tắt từ dataset ngoài

Từ điển viết tắt ban đầu (~20 mục) được soạn tay sau khi đọc câu hỏi train, nên dễ overfit. Bản hiện tại ([`src/neorag/resources/abbreviations.tsv`](../src/neorag/resources/abbreviations.tsv)) do [`scripts/build_abbreviations.py`](../scripts/build_abbreviations.py) trích từ hai dataset chuẩn hóa văn bản tiếng Việt:

| Nguồn | Quy mô | Nhãn | License |
|---|---|---|---|
| [yammdd/vietnamese-error-correction-corpus](https://huggingface.co/datasets/yammdd/vietnamese-error-correction-corpus) | ~70k cặp `input`/`target` | LLM sinh | MIT |
| [ViLexNorm](https://github.com/ngxtnhi/ViLexNorm) (Nguyen et al., EACL 2024) | 10.467 cặp `original`/`normalized` | người gắn | CC BY-NC-SA 4.0 |

- Mỗi cặp câu được căn theo âm tiết bỏ dấu (`difflib`). Một token bị thay bằng một cụm dài hơn và khác nó sau khi bỏ dấu thì được tính là một viết tắt.
- Một ánh xạ được giữ khi có trong ViLexNorm, hoặc gặp ít nhất 3 lần trong yammdd.
- Bỏ các trường hợp sau:
  - token có chữ số, hoặc là một âm tiết phổ biến (`nhin`, `ca`);
  - nghĩa chứa âm tiết hiếm, tức nhãn lỗi (`khongo`);
  - lỗi gõ thiếu 1 ký tự (`lic` → `lịch`);
  - nghĩa nhiều âm tiết không bắt đầu bằng chữ đầu của viết tắt, do phép căn chuỗi gộp nhầm từ đứng trước (`dc` → "bằng được"). Có tính các chữ thay thế của teencode (`z` ~ `v`/`d`, `k` ~ `c`...);
  - từ tục tĩu và từ xúc phạm (`súc vật`).
- Gộp thêm các mục soạn tay (`mm`, `mpc`, `pro`, `ttv`...) và `t1`…`t12`, trong đó `t2`–`t7` mang cả nghĩa "tháng N" lẫn "thứ N". Kết quả: 614 viết tắt.
- Không bỏ nghĩa nào. `Abbreviations.rank_by_corpus` chỉ đưa các nghĩa có trong corpus lên trước. Nếu bản thân token là một từ trong corpus (`cv` trong "vòng CV"), giữ nguyên token cũng được tính là một nghĩa.
- **Viết tắt đã biết / viết tắt lạ.** Các mục soạn tay (nguồn `manual`, gồm `t1`…`t12`) là viết tắt đã biết, đã được giải thích trong prompt viết lại câu hỏi:
  - không sinh gợi ý cho chúng;
  - BM25 chỉ mở chúng theo nghĩa soạn tay đầu tiên (`hv` → học viện, `dk` → đăng ký, `t5` → tháng 5), đúng như bản trước khi có từ điển.
- Với viết tắt lạ (`bn`, `bh`, `trc`, `chx`...), tối đa 3 nghĩa ứng viên được nối vào **user message** của bước viết lại để Haiku chọn theo ngữ cảnh. System prompt không đổi.
- Khi có bước viết lại, BM25 trên câu gốc chỉ mở viết tắt lạ có một nghĩa. Các nghĩa trùng nhau sau khi bỏ dấu ("học viện" / "học viên") được gộp làm một. Ở chế độ `--offline`, BM25 mở mọi nghĩa có trong corpus.

Bản đầu tiên (commit `9a26c03`) gợi ý cho **mọi** viết tắt, kể cả `ko`, `hv`, `mm`, và Public tụt từ 88.64 xuống **87.96**: retrieval giảm từ 93.75 xuống 92.59, generation giảm từ 83.53 xuống 83.32.
- Đúng 17/42 câu Public có gợi ý, và đúng 17 câu đó có câu trả lời thay đổi.
- Mọi gợi ý đều là viết tắt Haiku đã hiểu đúng. Gợi ý chỉ làm user message khác đi, nên bản viết lại bị xáo trộn ngẫu nhiên. Ví dụ ở public-0012, LLM hiểu "bên code" thành CLB Lập Trình.

Sau khi chỉ gợi ý viết tắt lạ:
- tập từ khóa BM25 trùng khớp với bản 88.64 trên cả 42 câu Public và 126 câu train;
- 42/42 câu trả lời Public giống hệt bản 88.64, vì dùng lại cache.

Để đo việc chống overfit, bộ [`eval/abbrev_dev.jsonl`](../eval/abbrev_dev.jsonl) gồm 27 câu viết tắt / không dấu, soạn trực tiếp từ corpus (không dựa vào câu hỏi train hay public), trong đó có 5 câu đa nghĩa (`t5`, `hv`, `tt`). Kết quả retrieval của `scripts/eval_retrieval.py` (chưa tính bước LLM chọn đoạn):

| Bộ câu hỏi | Chế độ | Từ điển soạn tay | Gợi ý mọi viết tắt (`9a26c03`) | Chỉ gợi ý viết tắt lạ (hiện tại) |
|---|---|---:|---:|---:|
| train (122) | `--offline` | 0.805 | 0.805 | 0.805 |
| train (122) | đầy đủ | 0.884 | 0.888 | 0.884 |
| dev viết tắt (27) | `--offline` | 0.715 | 0.766 | **0.766** |
| dev viết tắt (27) | đầy đủ | 0.836 | 0.870 | **0.847** |
| Public (42) | nộp bài | 88.64 | 87.96 | = 88.64 (câu trả lời giống hệt) |

- Phần tăng trên dev đến từ các viết tắt mà từ điển soạn tay không có (`bn`, `bh`, `trc`, `chx`, `lm`, `fb`).
- Bản hiện tại kém bản gợi ý mọi viết tắt trên dev (0.847 so với 0.870), nhưng không làm xáo trộn câu hỏi chỉ chứa viết tắt quen thuộc, vốn chiếm toàn bộ Public.
- Nhóm đa nghĩa (5 câu) chưa cải thiện. Ví dụ, LLM hiểu `t5 ... co sinh hoat` là "tháng 5" chứ không phải "thứ 5", vì bản thân câu hỏi mơ hồ.

## 4. Cách chạy

```bash
# Chấm riêng retrieval, chia theo kiểu câu hỏi
python scripts/eval_retrieval.py --offline       # chỉ BM25, không gọi API
python scripts/eval_retrieval.py --no-rewrite    # thêm bge-m3
python scripts/eval_retrieval.py                 # thêm bước viết lại câu hỏi
python scripts/eval_retrieval.py --questions eval/abbrev_dev.jsonl   # bộ câu viết tắt riêng

# Sinh lại từ điển viết tắt (cần tải CSV của hai dataset về trước)
python scripts/build_abbreviations.py --yammdd <thư mục> --vilexnorm <thư mục>

# So các biến thể prompt trên hai nửa tune/hold của train
neorag run train --pipeline neorag.pipelines.hybrid:HybridPipeline --no-judge --out outputs/train_pred_v5.jsonl
python scripts/eval_split.py outputs/eval_train_pred/per_query.jsonl outputs/eval_train_pred_v5/per_query.jsonl

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
