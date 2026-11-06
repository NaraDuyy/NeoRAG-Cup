# Nhật ký thử nghiệm

Mỗi mục ghi theo thứ tự:
- **Giả thuyết**: điều mình nghĩ đang xảy ra;
- **Vì sao thử**: bằng chứng dẫn tới giả thuyết;
- **Cách đo**: đo trên dữ liệu nào, so với cái gì;
- **Chi phí** (USD, ước tính);
- **Kết quả**;
- **Kết luận**: giữ / bỏ, và học được gì.

Mốc so sánh là **v7b** (commit `460bb59`): Public 89.20, Benchmark train 92.69.

Ngân sách cho đợt này: **$2.60** (đã tiêu $1.40 trước đó, tổng không quá $4.00).

---

## Nguyên tắc chung của đợt thử nghiệm

1. **Không tin chênh lệch nhỏ.** Haiku không cho đặt `temperature`, nên mỗi lần sinh lại câu trả lời cho ra câu chữ khác. Ở plan v7, ROUGE-L của từng câu dao động tới ±0.4 dù ý không đổi. Vì vậy thử nghiệm đầu tiên là chạy lại đúng v7b để đo **mức nhiễu nền**. Một thay đổi chỉ đáng tin khi vượt rõ mức nhiễu đó.
2. **Chọn trên nửa `hold`, đọc lỗi trên nửa `tune`.** `scripts/eval_split.py` chia train cố định theo kiểu câu hỏi. Mình chỉ đọc câu trả lời sai ở nửa tune để nghĩ ra cách sửa. Nửa hold chỉ dùng để chấm, nhờ vậy giảm việc "học thuộc" train.
3. **Đo rẻ trước, đo đắt sau.** Mỗi biến thể chạy qua `scripts/quick_eval.py`:
   - tính tại chỗ (không tốn API): Recall/MRR/NDCG@5, String Presence, ROUGE-L;
   - Response Relevancy: chỉ 1 lần gọi judge ngắn mỗi câu.

   Groundedness và Noise cần đưa 10 đoạn văn vào judge nên đắt. Chỉ chấm judge đầy đủ cho bản vào vòng cuối.
4. **Public là tín hiệu thật nhưng nhiễu.** Public chỉ có 42 câu, judge lệch khoảng ±1 điểm mỗi lần chấm. Vì vậy không chọn cấu hình chỉ dựa trên một lần nộp Public.

---

## E0. Ước lượng chi phí mỗi loại lượt chạy

- **Vì sao:** cần biết $2.60 đủ cho bao nhiêu thử nghiệm. Trang tra cứu ngân sách cần dán key, mình không được làm việc đó. Cache lại chỉ lưu câu trả lời, không lưu prompt.
- **Cách làm:**
  - đếm số lần gọi đã có trong cache: 1597 lần gọi chat của pipeline, 1618 lần gọi chat của judge;
  - tái dựng độ dài prompt trung bình của từng loại lần gọi trên train v7b;
  - chia $1.40 đã tiêu theo tỉ lệ ký tự, trong đó ký tự output tính gấp 5 vì token output đắt hơn.

| Loại lần gọi | Input trung bình (ký tự) | Ghi chú |
|---|---:|---|
| Viết lại câu hỏi | ~1 000 | |
| Trả lời (12 đoạn + 3 ví dụ) | ~9 700 | chiếm phần lớn chi phí |
| Judge Groundedness | ~5 300 | đưa cả 10 đoạn `retrieved_ids` vào |
| Judge Noise | ~2 000 | |
| Judge Response Relevancy | ~500 | rất rẻ |

- **Kết quả:** khoảng $0.07 cho mỗi triệu ký tự quy đổi. Từ đó:
  - 1 lượt trả lời trên train ≈ **$0.10**;
  - chấm judge đầy đủ trên train ≈ **$0.075**;
  - chỉ chấm RR ≈ **$0.01**;
  - 1 lượt Public ≈ **$0.035**.
- **Kết luận:** $2.60 đủ cho khoảng 15–20 vòng "chạy train + chấm nhanh", cộng vài lần chấm judge đầy đủ. Mình giữ khoảng $0.3 dự phòng cho Private Test và các lần chạy lại.

## E1. Vì sao Public thấp hơn train? Phân tích không tốn API

- **Bằng chứng:**

| | Train (v7b) | Public (v7b) |
|---|---:|---:|
| Retrieval | 98.76 | 93.54 |
| ROUGE-L | 67.74 | 63.00 |
| Response Relevancy | 66.28 | 58.93 |

- **Giả thuyết:** Public khác train ở một đặc điểm nào đó của câu hỏi.
- **Phát hiện:**
  - **Public có 20/42 câu viết không dấu (48%), train chỉ có 26/115 (23%).**
  - Trên train, RR của câu không dấu là 0.482, của câu có dấu là 0.716. Lý do nằm ở cách judge tính RR: Haiku sinh 3 câu hỏi (có dấu) từ câu trả lời, rồi bge-m3 so chúng với **câu hỏi gốc không dấu**. Hai câu cùng nghĩa nhưng một bên không dấu thì cosine thấp.
  - Tính theo tỉ lệ dấu của Public: 0.52 × 0.716 + 0.48 × 0.482 ≈ **0.60**, khớp với 58.9 đo được. Như vậy khoảng cách RR **giải thích được hoàn toàn bằng tỉ lệ câu không dấu**. Pipeline không làm sai gì, và đội nào cũng chịu như nhau.
  - Ngược lại, câu không dấu trên train **không** kém ở retrieval (Recall 0.974 so với 0.985) hay ROUGE-L (0.730 so với 0.662). Vậy khoảng cách retrieval và ROUGE-L trên Public **không** do thiếu dấu. Nhiều khả năng Public có nhiều câu cần nhiều đoạn hơn, nhưng không có nhãn Public thì không kiểm chứng được. Quy định cấm gán nhãn thủ công cho Public.
  - Câu trả lời ngắn được RR cao hơn: với câu không dấu, nửa ngắn được 0.528, nửa dài được 0.436. Đây là một hướng thử: viết gọn, đúng trọng tâm. Nhưng nó có thể làm giảm String Presence và ROUGE-L, nên phải đo cả ba.
  - Trên Public, LLM ghi nhiều id ở dòng `ID:` hơn so với train: chỉ 6/42 câu ghi 1 id, so với 37/126 câu trên train. Điều này hợp với giả thuyết Public có nhiều câu cần nhiều đoạn.

## E2. Đợt 1: nhiễu nền, 20 ứng viên, 5 ví dụ (đã chạy, ~$0.38)

| Biến thể | Thay đổi | Vì sao thử | Chi phí |
|---|---|---|---:|
| `rep1` | v7b y nguyên, chỉ thêm 1 dòng trống cuối user message để không trúng cache | Đo nhiễu nền: hai lần chạy cùng cấu hình chênh nhau bao nhiêu | ~$0.10 |
| `cand20` | `n_candidates` 12 → 20 | Độ phủ đoạn đúng trên train tăng từ 0.980 lên 0.988 khi mở từ 12 lên 20 đoạn. Nếu Public cần nhiều đoạn hơn, LLM cần thấy nhiều đoạn hơn để ghi đủ vào dòng `ID:` | ~$0.15 |
| `shot5` | `few_shot` 3 → 5 | Few-shot là thay đổi duy nhất tăng ROUGE-L trên cả hai nửa (v7). Thêm ví dụ có thể tăng tiếp, nhưng cũng có thể làm LLM chép văn phong quá mức | ~$0.11 |

**Kết quả** (`scripts/quick_eval.py --rr`; RR = Response Relevancy chỉ chấm cho câu có đáp án):

| Biến thể | Nửa | Retrieval | SP | ROUGE-L | RR |
|---|---|---:|---:|---:|---:|
| v7b | tune / hold | 0.9940 / 0.9845 | 0.934 / 0.935 | 0.703 / 0.647 | 0.671 / 0.650 |
| rep1 (chạy lại v7b) | tune / hold | 0.9899 / 0.9843 | 0.917 / 0.923 | 0.680 / 0.652 | 0.670 / 0.650 |
| cand20 | tune / hold | 0.9907 / 0.9836 | 0.934 / 0.929 | 0.698 / 0.662 | 0.673 / 0.660 |
| shot5 | tune / hold | 0.9905 / 0.9843 | 0.934 / 0.957 | 0.669 / 0.680 | 0.676 / 0.655 |

So cặp từng câu bằng `scripts/paired.py`, mỗi ô là chênh lệch trung bình ± sai số chuẩn (SE) trên 114 câu có đáp án:

| so với | SP | ROUGE-L | TB(SP, RL) |
|---|---|---|---|
| rep1 − v7b (**nhiễu thuần**) | −0.0145 ± 0.0093 | −0.0091 ± 0.0101 | −0.0118 ± 0.0077 |
| cand20 − v7b | −0.0029 ± 0.0080 | +0.0047 ± 0.0104 | +0.0009 ± 0.0073 |
| shot5 − v7b | +0.0109 ± 0.0117 | −0.0008 ± 0.0127 | +0.0050 ± 0.0095 |
| cand20 − rep1 | +0.0116 ± 0.0089 | +0.0138 ± 0.0117 | +0.0127 ± 0.0080 |
| shot5 − rep1 | +0.0254 ± 0.0135 | +0.0083 ± 0.0107 | +0.0168 ± 0.0094 |

**Kết luận và bài học**
- **Nhiễu nền lớn.** Chạy lại y hệt v7b mà trung bình SP và ROUGE-L đã lệch 0.012. Muốn thấy một cải tiến ở mức 2 SE (~0.02) thì nó phải tăng ROUGE-L hoặc SP khoảng 2 điểm phần trăm. Kiểu chỉnh prompt nhỏ hầu như không làm được vậy. Nói cách khác, **tập train 115 câu không đủ lớn để chọn giữa các biến thể chênh nhau ít.**
- **Không nhìn từng nửa riêng.** shot5 trông rất tốt trên hold (ROUGE-L +0.033, SP +0.022) nhưng kém trên tune (ROUGE-L −0.034). Đó chính là nhiễu: nếu chỉ nhìn hold, mình sẽ chọn nhầm. Từ đây, mình so cặp từng câu trên cả train và kèm SE.
- RR gần như không đổi giữa các lần chạy (±0.001), nên nó là chỉ số ổn định, so được.
- cand20 và shot5 đều nhích lên một chút so với trung bình của hai lần chạy v7b, nhưng chưa vượt nhiễu. Chưa giữ. Cả hai sẽ được thử lại ở chỗ có tín hiệu rõ hơn (E3).

## E3. Đổi hướng: dùng phần retrieval của Public làm tín hiệu

- **Lập luận:**
  - Trên train, retrieval đã gần trần (98.8), nên không còn chỗ để thấy cải tiến.
  - Trên Public, retrieval chỉ 93.5. Đây là chỗ mất nhiều điểm nhất: (100 − 93.5) × 0.5 ≈ **3.2 điểm tổng**.
  - Quan trọng hơn, **điểm retrieval không có nhiễu**. Recall, MRR và NDCG tính thẳng từ `retrieved_ids`, không qua LLM judge. Nhiễu ±1 của Public chỉ nằm ở phần generation.
  - Vì vậy, nộp Public cho các thay đổi về retrieval cho tín hiệu sạch, dù chỉ trên 42 câu.
- **Rủi ro overfit:** chỉ thử các thay đổi có cấu trúc chung (số ứng viên, cách xếp phần sau dòng `ID:`), không thử thay đổi nào nhắm vào câu cụ thể. Không đọc hay gán nhãn câu Public (quy định cấm).
- **Lưu ý khi nộp:** bảng xếp hạng lấy **lần nộp gần nhất**. Sau khi thử xong phải nộp lại bản tốt nhất.

**Các file Public đã tạo** (chi phí ~$0.05, chỉ cand20 cần sinh câu trả lời mới):

| File | Thay đổi so với v7b | Top 5 khác tập | Câu trả lời khác | Đọc kết quả thế nào |
|---|---|---:|---:|---|
| `public_test_pred_tail_submission.zip` | Phần sau các id LLM ghi xếp theo thứ tự gộp cross-encoder + RRF. Ứng viên giữ nguyên | 22/42 | **0/42** | Câu trả lời y hệt v7b, nên chênh lệch Recall/MRR/NDCG là tác động **thuần** của reranker. Groundedness cũng có thể đổi chút, vì judge dùng 10 `retrieved_ids` làm ngữ cảnh |
| `public_test_pred_cand20_submission.zip` | LLM thấy 20 ứng viên thay vì 12 | 8/42 | 39/42 | Retrieval tăng nghĩa là LLM tìm thêm được đoạn đúng. Phần generation lẫn nhiễu |
| `public_test_pred_cand20tail_submission.zip` | Cả hai | 26/42 | 39/42 | Nộp sau cùng, nếu cả hai cái trên đều tăng |

- Cả 3 file vẫn mở đầu bằng câu từ chối ở đúng 3 câu như v7b, nên không có rủi ro Abstention mới.
- **Kỳ vọng:** trên train, xếp lại phần đuôi không có lợi vì train đã gần trần. Nhưng trên Public có 22/42 câu đổi tập top 5. Nếu Public đúng là nhiều câu cần nhiều đoạn, đây là chỗ reranker có thể giúp. Nếu Recall giảm thì giả thuyết "Public cần nhiều đoạn" yếu đi.

### E3a. Kết quả nộp `tail` (33106)

| Chỉ số | v7b lần 1 (33063) | v7b lần 2 (33067) | **tail (33106)** |
|---|---:|---:|---:|
| Recall@5 | 90.71 | 90.71 | **91.31** |
| MRR@5 | 97.62 | 97.62 | 97.62 |
| NDCG@5 | 92.29 | 92.29 | **92.69** |
| **Retrieval** | 93.54 | 93.54 | **93.87** |
| String Presence | | 95.30 | 95.30 |
| ROUGE-L | | 63.00 | 63.00 |
| Groundedness | | 97.44 | 98.72 |
| Response Relevancy | | 59.56 | 59.51 |
| 1 − Noise | | 93.59 | 91.88 |
| Abstention | | 100 | 100 |
| Generation | | 84.82 | 84.74 |
| **Tổng** | 89.20 | 89.18 | **89.30** |

**Đọc kết quả:**
- **Reranker giúp retrieval trên Public: +0.33 retrieval, tức +0.17 điểm tổng.** Đây là tín hiệu sạch vì câu trả lời giống hệt v7b. Giả thuyết E3 đúng: trên train phần đuôi không quan trọng vì LLM đã ghi gần đủ đoạn đúng, còn trên Public thì có.
- **Đo được nhiễu của judge trên Public.** Câu trả lời giống hệt nhau mà 1 − Noise lệch 1.7 (93.59 → 91.88), Groundedness lệch 1.3. SP và ROUGE-L không đổi vì tính tất định. Hai lần nộp v7b chỉ lệch 0.02 điểm tổng, nhưng từng chỉ số judge có thể lệch tới khoảng 1.5. Nghĩa là: **chênh lệch Generation dưới khoảng 0.5 trên Public chưa nói lên gì.**
- **Kết luận: giữ reranker cho phần sau dòng `ID:`** (`reranker="BAAI/bge-reranker-v2-m3"`, `rerank_candidates=False`). Đây là thay đổi đầu tiên có bằng chứng trên dữ liệu chưa thấy.
- Lưu ý vận hành: ngày Private, máy chạy phải có sẵn model (đã tải, 2.3 GB). Mỗi câu tốn khoảng 10 giây CPU.

### E3b. Kết quả nộp `cand20` (33107)

| Chỉ số | v7b (33067) | tail (33106) | **cand20 (33107)** |
|---|---:|---:|---:|
| Recall@5 | 90.71 | 91.31 | **91.90** |
| MRR@5 | 97.62 | 97.62 | 97.62 |
| NDCG@5 | 92.29 | 92.69 | 92.68 |
| **Retrieval** | 93.54 | 93.87 | **94.07** |
| String Presence | 95.30 | 95.30 | 93.59 |
| ROUGE-L | 63.00 | 63.00 | 62.91 |
| Generation | 84.82 | 84.74 | 84.44 |
| Tổng | 89.18 | 89.30 | 89.25 |

**Đọc kết quả:**
- **Cho LLM thấy 20 đoạn thay vì 12 làm Recall tăng +1.19.** Đây là bằng chứng trực tiếp cho giả thuyết ở E1: trên Public có những đoạn đúng nằm ở hạng 13–20 mà LLM trước đây không được thấy. Trên train thì không thấy hiệu ứng này, vì độ phủ ở 12 đoạn đã là 0.98.
- String Presence giảm 1.7 là do câu trả lời được sinh lại. Trên 39 câu có đáp án, 1.7 điểm chỉ tương đương khoảng một key fact. Ở train, chạy lại y hệt đã lệch SP khoảng 1.5 (E2), nên mức này nằm trong nhiễu sinh lại.
- tail và cand20 tác động lên hai chỗ khác nhau: tail xếp phần đuôi, cand20 mở rộng ứng viên. Vì vậy mình kỳ vọng gộp lại sẽ cộng dồn. File `cand20tail` có câu trả lời giống hệt cand20, nên so với cand20 sẽ thấy tác động thuần của tail.

### E3c. Kết quả nộp `cand20tail`: 89.43, cao nhất tới lúc này

| Chỉ số | v7b | tail | cand20 | **cand20tail** |
|---|---:|---:|---:|---:|
| Recall@5 | 90.71 | 91.31 | 91.90 | **92.50** |
| NDCG@5 | 92.29 | 92.69 | 92.68 | **93.48** |
| **Retrieval** | 93.54 | 93.87 | 94.07 | **94.53** |
| Generation | 84.82 | 84.74 | 84.44 | 84.32 |
| **Tổng** | 89.18 | 89.30 | 89.25 | **89.43** |

- **Cộng dồn đúng như dự đoán:** tail +0.33, cand20 +0.53, gộp lại +0.99 (còn hơn tổng hai phần một chút).
- Generation của cand20tail và cand20 dùng cùng câu trả lời. SP, ROUGE-L, Groundedness và Noise giống hệt nhau, chỉ RR lệch 0.7. Như vậy RR trên Public cũng có nhiễu, vì bước sinh câu hỏi ngược của judge mỗi lần mỗi khác.
- Generation thấp hơn v7b khoảng 0.5, chủ yếu do String Presence (95.30 → 93.59) khi sinh lại câu trả lời. Không có lý do để nghĩ 20 đoạn làm câu trả lời kém đi. Trên train, cand20 không làm SP giảm (E2). Đây là nhiễu sinh lại, nhưng mình sẽ theo dõi ở các lần nộp sau.
- **Bước tiếp (E3d):** nếu 12 → 20 giúp, thì thử 30 (`cand30tail`, reranker chấm 30 đoạn đầu). Lập luận: còn khoảng 7.5% đoạn đúng chưa vào top 5. Nếu một phần trong số đó nằm ở hạng 21–30 thì 30 ứng viên sẽ bắt được. Rủi ro: prompt dài gấp khoảng 2.5 lần bản 12 đoạn, LLM dễ ghi nhầm đoạn không liên quan vào dòng `ID:`. Nếu vậy MRR sẽ giảm.

### E3d. Kết quả nộp `cand30tail` (33112): 89.91

| Chỉ số | v7b | cand20tail | **cand30tail** |
|---|---:|---:|---:|
| Recall@5 | 90.71 | 92.50 | **93.17** |
| NDCG@5 | 92.29 | 93.48 | **94.13** |
| **Retrieval** | 93.54 | 94.53 | **94.97** |
| String Presence | 95.30 | 93.59 | 91.88 |
| ROUGE-L | 63.00 | 62.91 | 64.23 |
| 1 − Noise | 93.59 | 91.88 | 95.30 |
| Generation | 84.82 | 84.32 | 84.85 |
| **Tổng** | 89.18 | 89.43 | **89.91** |

- **Retrieval tiếp tục tăng** khi LLM được thấy nhiều ứng viên hơn: 12 → 20 → 30 đoạn cho 93.54 → 94.53 → 94.97 (có tail). Mức tăng nhỏ dần, nên không cần thử 40.
- **Điểm cần cảnh giác: String Presence giảm đều** (95.30 → 93.59 → 91.88) khi tăng ứng viên. Một lần giảm có thể là nhiễu sinh lại, nhưng hai lần giảm cùng chiều thì có thể có lý do thật: nhiều đoạn hơn thì LLM có nhiều thứ để "tóm tắt", câu trả lời bớt bám đúng chữ của key fact. ROUGE-L và Noise lại tăng, nên phần generation tổng gần như không đổi. Kiểm tra trên train ở E4, vì train có 115 câu và có đáp án để đọc.

## E4. Doc2query: mở rộng index bằng câu hỏi giả

- **Giả thuyết:** một phần đoạn đúng bị xếp thấp vì cách người hỏi diễn đạt khác cách tài liệu viết (train-0028: "y tế" so với "Medical"; câu hỏi đời thường so với văn bản trang trọng). Nếu mỗi đoạn có sẵn vài câu hỏi mẫu, BM25 và bge-m3 sẽ khớp được những cách nói đó.
- **Vì sao chọn cách này:** đây là kỹ thuật doc2query đã được kiểm chứng trong tìm kiếm thông tin. Nó chỉ cần chạy **một lần** trên corpus, khoảng $0.02, rồi dùng lại được cho Private. Nó chỉ đọc corpus, không nhìn câu hỏi train hay test, nên không rò rỉ đáp án.
- **Cách làm** ([`scripts/build_doc2query.py`](../scripts/build_doc2query.py)):
  - Haiku viết 5 câu hỏi cho mỗi đoạn, yêu cầu phủ nhiều ý và đa dạng cách nói (trang trọng, đời thường, từ đồng nghĩa). Tổng cộng 589 câu, lưu ở `src/neorag/resources/doc2query.jsonl`.
  - Hai cách dùng: `bm25` nối các câu hỏi vào văn bản của đoạn khi index BM25; `dense` thêm một bảng xếp hạng mới, trong đó điểm của đoạn là cosine cao nhất giữa câu hỏi người dùng và các câu hỏi giả của đoạn.
- **Đo trước bước LLM** (`eval_retrieval.py --doc2query ...`, không tốn API):

| | dev (27, chưa thấy) | tune | hold | Phủ@12 hold |
|---|---:|---:|---:|---:|
| Không dùng | 0.847 | 0.898 | 0.868 | 0.975 |
| `bm25` | 0.953 | **0.912** | **0.899** | 0.975 |
| `dense` | 0.941 | 0.903 | 0.889 | 0.977 |
| `bm25 dense` | **0.968** | 0.902 | 0.895 | **0.983** |

- **Đọc kết quả:** tăng trên cả ba bộ. Mức tăng lớn nhất ở dev, bộ 27 câu tự soạn từ corpus mà pipeline chưa từng được chỉnh theo. Đây là dấu hiệu tốt về khả năng tổng quát, khác hẳn những thay đổi trước đây vốn chỉ nhích trên train.
- Mình chọn `bm25 dense`. Nó tốt nhất trên dev và có độ phủ ứng viên cao nhất, mà độ phủ chính là thứ quyết định khi LLM chọn đoạn (bài học từ E3). Nó chỉ kém `bm25` một chút trên tune/hold, trong phạm vi nhiễu.
- **Đang chạy:**
  - Public `c30d` (cand30 + tail + doc2query): xem retrieval trên Public.
  - Train `c30d` và `c30` (cand30 + tail, không doc2query): so SP/ROUGE-L với v7b để kiểm tra nghi vấn "nhiều ứng viên làm giảm SP".
  - Chi phí khoảng $0.06 + 2 × $0.2.

## E5. MBR decoding cho câu trả lời

- **Giả thuyết:** E2 cho thấy mỗi lần sinh lại câu trả lời lệch ngẫu nhiên khoảng ±0.01–0.02 ROUGE-L/SP. Nếu sinh 3 câu và chọn câu "đồng thuận" nhất, tức câu có ROUGE-L trung bình cao nhất so với 2 câu kia, thì sẽ loại được những lần sinh lệch: thiếu key fact, lan man, ghi sai id. Đó là Minimum Bayes Risk decoding.
- **Hợp lệ:** việc chọn chỉ so các câu trả lời với nhau, không dùng đáp án chuẩn. Chạy được y hệt trên Private.
- **Cài đặt:** `HybridPipeline(mbr=3)` sinh 3 mẫu, mẫu thứ i có thêm i dòng trống cuối tin nhắn để không trúng cache của nhau. Hàm chọn là `mbr_select`. Với `mbr_ids="vote"`, các id được quá nửa số mẫu ghi được đưa lên đầu.
- **Mẹo tiết kiệm:** mẫu 0 chính là v7b, mẫu 1 chính là `rep1` của E2, cả hai đã có trong cache. Vì vậy chỉ tốn 1 lượt mới, khoảng $0.10.
- **Kết quả trên train** (cấu hình v7b, 114 câu có đáp án):

| | SP | ROUGE-L | RR | MRR |
|---|---:|---:|---:|---:|
| mẫu 0 (v7b) | 0.934 | 0.676 | 0.660 | 1.000 |
| mẫu 1 (rep1) | 0.920 | 0.666 | 0.660 | 1.000 |
| mẫu 2 (rep2) | 0.928 | 0.670 | | **0.996** |
| **MBR 3 mẫu** | 0.930 | **0.680** | 0.661 | 1.000 |
| MBR + id theo phiếu | 0.930 | 0.680 | 0.661 | 1.000 |

  - So cặp, MBR trừ trung bình 3 mẫu đơn, TB(SP, RL) = **+0.0058 ± 0.0032** (≈1.8 SE).
  - So với "oracle" (luôn chọn đúng mẫu tốt nhất theo đáp án chuẩn, việc không làm được khi thi): −0.028. Nghĩa là chọn bằng đồng thuận chỉ lấy lại được một phần nhỏ khoảng cách giữa các mẫu.
- **Kết luận:** có lợi nhưng nhỏ, khoảng +0.1 điểm tổng, đổi lại tốn gấp 3 lần số lần gọi trả lời. Giá trị chính là **giảm rủi ro** trên Private: một lần sinh tệ (như mẫu 2 ghi sai id đầu) bị loại. Vì Private chỉ chạy một lần, giảm phương sai là có ích. Bỏ phiếu id không khác gì so với dùng id của câu được chọn. Quyết định cuối cùng phụ thuộc ngân sách còn lại khi chạy Private.

### E4a. Kết quả nộp `c30d` (cand30 + tail + doc2query): 89.70, không giữ doc2query

| Chỉ số | cand30tail | **c30d** |
|---|---:|---:|
| Recall@5 | 93.17 | 92.70 |
| NDCG@5 | 94.13 | 93.45 |
| **Retrieval** | **94.97** | 94.59 |
| Generation | 84.85 | 84.80 |
| **Tổng** | **89.91** | 89.70 |

- **Doc2query không giúp trên Public**, thậm chí làm retrieval giảm 0.38 (30/42 câu đổi tập top 5).
- **Bài học, lần thứ hai gặp:** một cải tiến làm tăng thứ hạng **trước bước LLM chọn đoạn** (dev 0.847 → 0.968) chưa chắc làm tăng `retrieved_ids` **cuối cùng**. Khi LLM đã được thấy 30 ứng viên, độ phủ gần như đủ. Lúc đó thứ quyết định là LLM ghi id nào và phần đuôi được xếp ra sao. Doc2query đổi thứ tự RRF, kéo theo thay đổi thứ tự reranker gộp ở phần đuôi, và lần này đổi theo hướng xấu. Lần đầu gặp bài học này là reranker chọn ứng viên ở v8.
- Từ nay, chỉ chấp nhận thay đổi retrieval khi nó tăng **retrieval cuối cùng**, đo bằng `eval_final_retrieval.py` trên train hoặc nộp Public, không dựa vào `eval_retrieval.py`.
- Code doc2query được giữ, tắt mặc định (`doc2query=()`).

---

## Các bước sai và sự cố (ghi lại để rút kinh nghiệm)

| # | Chuyện gì xảy ra | Hậu quả | Nguyên nhân gốc | Lần sau làm gì |
|---|---|---|---|---|
| S1 | Nhìn riêng nửa hold, shot5 trông rất tốt (ROUGE-L +0.033, SP +0.022). Suýt chọn nó | Không mất gì vì đã dừng lại kiểm tra; nửa tune lại giảm 0.034 | Không có mức nhiễu nền để so. Một lần chạy lại y hệt đã lệch tới ±0.02 | Luôn chạy một bản lặp lại (rep) trước. So cặp từng câu và kèm sai số chuẩn (`scripts/paired.py`) |
| S2 | Doc2query tăng mạnh retrieval trước bước LLM (dev 0.847 → 0.968). Mình kỳ vọng Public tăng, nhưng Public lại **giảm** 0.38 | Mất 1 lượt nộp Public + ~$0.07 sinh câu trả lời | Đo sai tầng: `eval_retrieval.py` chấm thứ tự RRF, nhưng điểm thật là `retrieved_ids` sau khi LLM chọn đoạn. **Đây là lần thứ hai mắc lỗi này** (lần đầu: reranker chọn ứng viên ở v8) | Thay đổi retrieval chỉ được chấp nhận khi đo trên retrieval **cuối cùng** |
| S3 | Khởi chạy train `c30d` cùng lúc với Public `c30d`, trước khi biết kết quả Public | Lượt train chạy dở thành vô ích, bị dừng giữa chừng. Tốn một phần của ~$0.25 | Muốn tiết kiệm thời gian nên chạy song song các bước phụ thuộc nhau | Bước đắt mà phụ thuộc kết quả bước trước thì phải chờ kết quả rồi mới chạy |
| S4 | `TaskStop` chỉ dừng shell; tiến trình con `neorag.exe` và `python.exe` (3 GB RAM, đang gọi API) vẫn chạy tiếp | Có thể tốn thêm API nếu không phát hiện | Trên Windows, dừng shell cha không dừng cây tiến trình con | Sau khi dừng, kiểm tra bằng `Get-CimInstance Win32_Process`, rồi `taskkill /PID <id> /T /F` đúng cây tiến trình của mình. Không đụng tiến trình của dự án khác (đang có `Url Investigation` chạy song song) |
| S5 | Sửa code bằng Python trong heredoc của Bash. `"\b"` bị biến thành ký tự backspace, `"\n"` thành xuống dòng thật | File lỗi cú pháp, mất thời gian sửa. **Lặp lại lần thứ ba** sau khi đã ghi chú | Bash và Python cùng xử lý escape | Sửa có escape thì dùng công cụ Edit, không dùng heredoc. Sau khi sửa, `grep -c $'\x08'` để dò ký tự lạ |
| S6 | `scripts/install.py --local` bị lỗi giữa chừng vì console Windows (cp1252) không in được tiếng Việt | Phải chạy lại cài đặt | Script in tiếng Việt mà không ép UTF-8 | Chạy mọi script của repo với `PYTHONIOENCODING=utf-8` trên máy này |
| S7 | Lần đầu mở trang cuộc thi bằng Chrome thì ra 404 | Phải chờ bạn đăng nhập | Chrome của extension chưa đăng nhập aichallenge; trang cuộc thi ẩn với khách | Kiểm tra trạng thái đăng nhập trước khi điều hướng |
| S8 | Ở plan v6, bảng bằng chứng ban đầu ghi rằng "LLM khai thừa id 7 và 35". Kiểm lại thì sai | Phải sửa plan | Suy luận từ `retrieved_ids` cuối mà không tách bước nào gây ra | Tái dựng từng bước từ cache (RRF, dòng `ID:`, phần đuôi) trước khi kết luận nguyên nhân |
| S9 | Giả định "câu không dấu kéo retrieval Public xuống". Kiểm lại thì không đúng (train: Recall câu không dấu 0.974 so với 0.985) | Không mất gì | Thấy một khác biệt lớn (48% so với 23% câu không dấu) là vội gán nó cho mọi chỉ số | Kiểm tra từng chỉ số riêng trước khi kết luận nguyên nhân |
| S10 | Nộp c30d làm bảng xếp hạng tụt từ 89.91 xuống 89.70, vì bảng xếp hạng lấy lần nộp **gần nhất** | Phải tốn thêm 1 lượt để nộp lại cand30tail | Đặc tính của trang thi | Sau mỗi lượt nộp thử kém hơn, nộp lại ngay bản tốt nhất. Giữ ít nhất 1 lượt/ngày cho việc này |

## E6. Kiểm tra trên train: 30 ứng viên có làm giảm String Presence không?

- **Vì sao:** trên Public, SP giảm qua ba lần nộp liên tiếp khi tăng ứng viên (95.30 → 93.59 → 91.88). Nghi đây là tác dụng phụ thật. Train có 115 câu kèm đáp án, nhiều hơn 39 câu của Public, nên kiểm chứng ở đây.
- **Chi phí:** khoảng $0.2.

| | Retrieval cuối | SP | ROUGE-L | RR | Độ dài TB |
|---|---:|---:|---:|---:|---:|
| v7b (12 ứng viên) | 0.9893 | 0.934 | 0.676 | 0.660 | 240 |
| rep1 (chạy lại v7b) | 0.9872 | 0.920 | 0.666 | 0.660 | 244 |
| cand20 | 0.9872 | 0.931 | 0.680 | 0.667 | 249 |
| **c30** (30 ứng viên + tail) | **0.9906** | **0.942** | 0.677 | 0.666 | 244 |

- So cặp c30 trừ v7b: TB(SP, RL) = +0.005 ± 0.010, tức không khác biệt.
- **Kết luận:** 30 ứng viên **không** làm SP giảm trên train, câu trả lời cũng không dài ra. Mức SP giảm trên Public là nhiễu sinh lại trên 39 câu.
- **Bài học:** đừng đọc xu hướng từ 3 điểm dữ liệu nhiễu trên tập nhỏ. Hãy kiểm chứng trên tập lớn hơn trước khi thêm một cơ chế "chữa" cho vấn đề có thể không tồn tại.

## Chốt sau đợt thử nghiệm

**Cấu hình mặc định mới** (`HybridPipeline()`): `n_candidates=30`, `reranker="BAAI/bge-reranker-v2-m3"`, `rerank_top=30`, few-shot 3, MBR tắt, doc2query tắt. Chạy pipeline mặc định trên Public cho ra file khớp 42/42 với `cand30tail` đã nộp.

| Thay đổi | Bằng chứng | Giữ? |
|---|---|---|
| Reranker xếp phần sau dòng `ID:` (tail) | Public retrieval +0.33 với câu trả lời y hệt | **Có** |
| 12 → 20 → 30 ứng viên cho LLM | Public retrieval 93.54 → 94.53 → 94.97 (có tail). Train không làm giảm SP | **Có (30)** |
| Doc2query | Tăng trước bước LLM nhưng Public cuối giảm 0.38 | Không (code giữ, tắt) |
| MBR 3 mẫu | +0.006 ± 0.003 TB(SP, RL) trên train, nhưng gấp 3 chi phí trả lời | Không bật mặc định. Cân nhắc nếu ngân sách Private dư |
| 5 ví dụ few-shot | Trong nhiễu | Không |

**Public:** 89.20 (v7b) → **89.94** (lần nộp lại cand30tail). Retrieval 93.54 → 94.97.

**Chi phí đợt này:** khoảng $1.0–1.1 trong $2.60, ước tính theo E0. Phần còn lại khoảng $1.5. Private với cấu hình mới tốn khoảng $0.0015 mỗi câu cho bước trả lời (prompt 30 đoạn dài khoảng 2 lần bản 12 đoạn).

## E7. Trọng số reranker cho phần đuôi (`rerank_weight`)

- **Vì sao:** tail với w=1 đã giúp trên Public (E3a). Câu hỏi tiếp theo là nên nghiêng về reranker nhiều hơn hay ít hơn. Thử nghiệm này **không tốn API**: câu trả lời và điểm reranker của cấu hình v9 đều đã có trong cache, chỉ tốn lượt nộp.
- **Trên train:** w = 0 / 1 / 2 / 100 cho retrieval cuối là tune 0.990 / 0.989 / 0.993 / 0.989 và hold 0.988 / 0.988 / 0.984 / 0.984. Chênh nhau ±0.003, train đã sát trần nên không phân biệt được. Phải nhờ Public.
- **Trên Public** (câu trả lời y hệt v9):

| w | Recall@5 | NDCG@5 | Retrieval | Generation (nhiễu) | Tổng |
|---|---:|---:|---:|---:|---:|
| 1 (v9) | 93.17 | 94.13 | 94.97 | 84.85 | 89.91 / 89.94 |
| **2** | **93.97** | **94.61** | **95.40** | 84.58 | **89.99** |

- **Kết luận:** chọn w=2 làm mặc định. Pipeline mặc định khớp 42/42 với file đã nộp.
- **Cảnh giác overfit:** đây là chỉnh một tham số dựa trên 42 câu Public. Recall +0.8 tương đương khoảng 1 đoạn đúng thêm vào top 5 trên toàn bộ 42 câu, nên tín hiệu khá mỏng. Tuy vậy chiều tăng nhất quán: w=0 → 1 → 2 đều tăng. Mình chỉ thử vài giá trị tròn, không dò mịn. Ngày mai thử thêm w=100 (gần như xếp hẳn theo reranker). Nếu không tăng nữa thì dừng ở 2.
- **Lượt nộp:** hôm nay đã dùng 9/10. Lượt thứ 10 để dành.

## E8. Lượt nộp cuối ngày: w=100, và phát hiện rủi ro Abstention

- **Thử:** w=100 (phần đuôi gần như xếp hẳn theo reranker). Câu trả lời y hệt w=2. Không tốn API.
- **Kết quả Public:** retrieval 95.43, gần như bằng w=2 (95.40), tức hết tăng. Không đổi mặc định.
- **Bất ngờ: Abstention 100 → 66.67, tổng tụt còn 87.40** dù câu trả lời **giống hệt** bản w=2 (89.99). Judge chấm một trong 3 câu từ chối lúc đạt lúc không. Mỗi lần như vậy mất khoảng **2.6 điểm tổng**, lớn hơn mọi cải tiến của cả đợt cộng lại.

### Tìm câu gây rủi ro (chi phí < $0.01)

- Chạy riêng judge Abstention 8 lần cho mỗi câu từ chối trên Public. Mỗi lần thêm khoảng trắng cuối câu để không trúng cache. So bản đầy đủ với bản chỉ giữ câu đầu.

| Câu | Bản đầy đủ (v9) | Bản v7b | Chỉ câu đầu |
|---|---:|---:|---:|
| public-0002 | 8/8 | 8/8 | 8/8 |
| **public-0024** | **5/8** | **1/8** | 8/8 |
| public-0027 | 8/8 | 8/8 | 8/8 |

- public-0024 hỏi so sánh học phí. Câu "Tài liệu chỉ cho biết …" kèm theo lại kể "đóng một khoản nhỏ hằng tháng gọi là quỹ CLB" và "0.1 điểm khuyến khích". Judge coi đó là đưa ra thông tin cụ thể như lời giải.
- Trên 11 câu không có đáp án của train (mỗi câu 4 lần): bản đầy đủ đạt 43/44. Lần trượt rơi vào **train-0019, cũng là câu hỏi về phí/quỹ**. Bản chỉ câu đầu đạt 44/44.
- **Bài học:** với câu không có đáp án, điểm **chỉ tính Abstention**; ROUGE-L và SP không áp dụng. Phần thông tin kèm theo câu từ chối không mang lại điểm mà chỉ thêm rủi ro. Quy tắc "có thể thêm 'Tài liệu chỉ cho biết …'" được đặt ra từ v5 vì muốn câu từ chối hữu ích hơn cho người đọc, nhưng chưa tính tới rủi ro này.

### Quyết định: `trim_refusal=True` (câu trả lời mở đầu bằng "Tài liệu hiện có không có thông tin" chỉ giữ câu đầu)

| | Mất | Được |
|---|---|---|
| Public | 0: cả 3 câu bị cắt đều là câu từ chối thật | Tránh rủi ro ~37% × 2.6 ≈ **1 điểm** kỳ vọng |
| Train | 4 câu **có đáp án một phần** mà LLM trả lời bằng câu từ chối (train-0027, 0053, 0062, 0119) mất phần thông tin: SP −0.022, ROUGE-L −0.007, ≈ **−0.24 điểm** | Tránh rủi ro ≈ 2.3% × 11 câu × 0.76 ≈ 0.19 điểm |

- Trên phân bố giống train, hai bên gần như hòa. Trên Public thì có lợi rõ. Thêm nữa, việc cắt **loại bỏ một khoản mất điểm lớn và ngẫu nhiên**, mà Private chỉ chạy một lần. Vì vậy mình chọn cắt.
- Muốn tắt thì dùng `HybridPipeline(trim_refusal=False)`.
- Bản v10 trên Public (`public_test_pred_v10_submission.zip`) chỉ khác w=2 ở 3 câu từ chối. Chưa nộp vì hôm nay đã hết lượt.

## E9. Truy xuất bằng chứng theo câu trả lời (`evidence_weight`)

- **Quan sát dẫn tới giả thuyết:** phân tích các đoạn đúng còn thiếu trên train cho thấy BTC gắn nhãn theo kiểu "**mọi đoạn có chứa thông tin của đáp án**", chứ không theo "đoạn nói về chủ đề câu hỏi".
  - Ví dụ train-0099: câu hỏi "ai đứng lớp, ra trường còn gắn bó không" không nhắc gì tới nội quy. Vậy mà đoạn [71] "Nghĩa vụ thành viên theo nội quy" vẫn là đoạn đúng, vì nó ghi "…duy trì truyền thống khóa trên dạy khóa dưới, quay lại giúp đỡ CLB khi cần", trùng thông tin với đáp án.
  - Tìm theo câu hỏi thì đoạn này nằm hạng 44 (BM25 38–72, bge-m3 48–61). Cố chỉnh retrieval theo câu hỏi để bắt nó là overfit.
- **Giả thuyết:** dùng chính **câu trả lời** của LLM làm truy vấn (BM25 + bge-m3) sẽ tìm ra những đoạn chứa cùng thông tin. Gộp thứ hạng này vào phần sau các id LLM ghi sẽ bắt được loại đoạn đúng mà câu hỏi không nhắc tới.
- **Không tốn API:** câu trả lời đã có trong cache, chỉ thêm tìm kiếm cục bộ và một lần embed bge-m3. Câu từ chối thì bỏ qua bước này.
- **Kết quả trên train** (retrieval cuối, câu trả lời y hệt):

| `evidence_weight` | tune | hold | Recall tăng | Recall giảm |
|---|---:|---:|---|---|
| 0 | 0.9929 | 0.9843 | | |
| 0.5 | 0.9930 | 0.9880 | 0048 | |
| 1 | 0.9889 | 0.9879 | 0048 | 0118 |
| **2** | 0.9929 | **0.9921** | 0015, 0048, **0099** | 0118 |

- Câu bị hỏng (train-0118): câu trả lời có thêm chi tiết phụ "theo cách giới thiệu các ban trong đợt tuyển Gen 14 (tháng 8/2026)", và chi tiết này kéo các đoạn về đợt tuyển Gen 14 lên. Đây đúng là rủi ro đã dự đoán: câu trả lời lan man thì kéo theo đoạn lan man.
- Trọng số không tăng đơn điệu (w=1 làm tune giảm), nên tín hiệu trên train còn nhiễu. Trên Public, file `evid2` đổi tập top 5 ở 30/42 câu, nhưng id đứng đầu và câu trả lời giữ nguyên. Đã nộp để kiểm chứng.

### E9a. Kết quả nộp `evid2`: tổng 90.12 nhưng **retrieval giảm**, không giữ

| | w2 (89.99) | evid2 (90.12) |
|---|---:|---:|
| Recall@5 | 93.97 | 93.97 |
| NDCG@5 | 94.61 | 94.26 |
| **Retrieval** | **95.40** | 95.28 |
| 1 − Noise | 94.44 | 97.22 |
| Abstention | 100 | 100 |
| Generation | 84.58 | 84.96 |

- **Bẫy dễ mắc:** nhìn tổng 90.12, cao nhất từ trước tới giờ, thì dễ kết luận "truy xuất theo câu trả lời có tác dụng". Thật ra phần chịu ảnh hưởng của thay đổi là retrieval lại **giảm 0.12**: Recall không đổi, NDCG giảm. Phần tăng nằm ở generation, đến từ 3 câu từ chối đã cắt (E8) và nhiễu judge (1 − Noise lệch 2.8 dù câu trả lời gần như giống hệt). Hai thứ này bản v10 cũng có.
- **Quy tắc rút ra:** mỗi thay đổi chỉ được đánh giá trên **phần chỉ số mà nó tác động tới**. Thay đổi ở retrieval thì đọc Recall/MRR/NDCG, không đọc tổng.
- Trên train, hold tăng +0.008 (sửa 3 câu, hỏng 1). Trên Public thì không. Có thể Public có nhiều câu trả lời nhắc chi tiết phụ (như train-0118) hơn, nên câu trả lời kéo đoạn lạc đề lên.
- **Quyết định:** `evidence_weight=0` (tắt; code giữ lại). Nộp **v10** (w2 + cắt câu từ chối) làm bản trên bảng xếp hạng.

### E9b. Kết quả nộp `v10` (w2 + cắt câu từ chối): **90.19**, bản trên bảng xếp hạng

| | w2 | evid2 | **v10** |
|---|---:|---:|---:|
| Retrieval | 95.40 | 95.28 | **95.40** |
| String Presence | 91.88 | 91.88 | 91.88 |
| ROUGE-L | 64.23 | 63.79 | 63.79 |
| Groundedness | 97.86 | 97.44 | 96.58 |
| Response Relevancy | 59.07 | 59.45 | 59.33 |
| 1 − Noise | 94.44 | 97.22 | 98.29 |
| Abstention | 100 | 100 | 100 |
| **Tổng** | 89.99 | 90.12 | **90.19** |

- Retrieval của v10 trùng khớp w2 (95.40), đúng như dự kiến vì `retrieved_ids` giống hệt.
- **Suy luận ngược từ điểm:** ROUGE-L chỉ tính trên câu **có đáp án**, và chỉ 3 câu từ chối bị cắt. Vậy việc ROUGE-L giảm 64.23 → 63.79 cho biết **một trong 3 câu đó thực ra có đáp án**, kiểu "đáp án một phần" như train-0027/0062/0119. Thiệt hại khoảng −0.44 ROUGE-L, tức −0.04 điểm tổng, khớp với đánh đổi đã ước lượng ở E8. Đổi lại, 3 lần nộp gần nhất có cắt hoặc không có câu kèm phí đều được Abstention 100.
- Hai lần nộp có cùng `retrieved_ids` hoặc cùng câu trả lời vẫn lệch nhau ở Groundedness, Noise, RR tới khoảng ±3. Mức chênh 90.12 so với 90.19 hoàn toàn nằm trong nhiễu judge.

## E10. Câu trả lời ngắn hơn có tăng Response Relevancy không? → **Không** (bác bỏ với ~$0.02)

- **Giả thuyết ban đầu (từ E1):** trên train, nửa câu trả lời ngắn có RR cao hơn nửa dài (0.755 so với 0.697 ở câu có dấu). RR đang mất nhiều điểm nhất (−3.39 điểm tổng). Vậy dặn LLM viết ngắn hơn có thể tăng RR.
- **Vì sao không chạy ngay:** sinh lại câu trả lời tốn khoảng $0.2, và còn ảnh hưởng tới SP/ROUGE-L. Trước đó có thể kiểm tra **rẻ hơn** bằng cách can thiệp trực tiếp: lấy chính câu trả lời v10, cắt máy còn 1 hoặc 2 câu đầu, rồi chấm lại RR. Chỉ tốn lần gọi RR.

| Bản | Độ dài TB | SP | ROUGE-L | RR |
|---|---:|---:|---:|---:|
| v10 nguyên bản | 225 | 0.920 | 0.671 | 0.667 |
| Giữ 2 câu đầu | 204 | 0.873 | 0.663 | 0.668 |
| Giữ 1 câu đầu | 99 | 0.439 | 0.403 | 0.588 |

- **Phân tích nguyên nhân:**
  - 24 câu bị cắt có RR 0.641 → 0.642, tức **không đổi**.
  - Trong nhóm câu có dấu, câu **hỏi** ngắn có RR 0.750, câu hỏi dài 0.702. Câu hỏi nhiều ý thì 3 câu hỏi sinh ngược khó khớp với nó.
  - Câu hỏi ngắn thường đi kèm câu trả lời ngắn. Mối liên hệ "trả lời ngắn ↔ RR cao" là **tương quan giả**, do độ dài câu hỏi gây ra.
- **Kết luận:** RR do câu hỏi quyết định (có dấu hay không, mấy ý), pipeline gần như không tác động được. Không sinh lại câu trả lời.
- **Bài học:**
  - **Tương quan không phải nhân quả.** Trước khi tốn tiền thay đổi theo một tương quan quan sát được, hãy tìm cách **can thiệp trực tiếp** rẻ nhất để kiểm tra. Ở đây, cắt máy thay cho viết lại.
  - Cắt máy không giống hệt dặn LLM viết ngắn: LLM có thể viết gọn mà vẫn đủ ý. Nhưng nếu bỏ hẳn cả câu thứ 3 mà RR vẫn không đổi, thì việc viết gọn hơn càng khó làm RR tăng.
