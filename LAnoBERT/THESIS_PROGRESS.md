# Theo dõi tiến độ luận văn

## 1. Phạm vi nghiên cứu

Đề tài triển khai cơ sở tham chiếu LAnoBERT cho phát hiện bất thường log,
sau đó mở rộng theo kiến trúc Hybrid-RAG:

```text
LAnoBERT baseline
    + Field-Value constraints
    + Ontology retrieval
    + Gated fusion
    -> early anomaly warning
```

Các thực nghiệm mục tiêu:

| Mã | Cấu hình |
|---|---|
| E0 | LAnoBERT baseline |
| E1 | LAnoBERT + Field-Value |
| E2 | LAnoBERT + Ontology |
| E3 | LAnoBERT + Gated Hybrid-RAG |

Các chỉ số chính:

- AUROC
- F1
- Precision và Recall
- Early Warning Rate
- Detection Lead Time
- Độ trễ suy luận

## 2. Đã đọc và tổng hợp kế hoạch

Đã đọc các tài liệu trong thư mục `Results-Gemini-newnew`, đặc biệt là:

- `result-6.md`: đặc tả kỹ thuật Hybrid-RAG.
- `result-7.md`: đặc tả phần mềm và ranh giới hệ thống.
- `result-9.md`: lộ trình triển khai và kế hoạch luận văn.

### Checkpoint 17 — smoke training có loss và gradient hợp lệ

Log training Kaggle mới cho thấy smoke run đã học thực sự:

```text
step đầu: loss=0.7508, grad_norm=2.91
loss khoảng epoch 0.2: 0.08916
loss khoảng epoch 0.5: 0.05263
loss cuối log: 0.03416, grad_norm=0.3112
epoch cuối log: 0.9245
```

Learning rate giảm từ xấp xỉ `1e-4` xuống `1.4e-6` theo cosine schedule và
`grad_norm` không bị cố định ở 0. Đây là bằng chứng smoke run đã tránh được
lỗi tokenizer 5 token/loss 0 trước đó.

Tuy nhiên, log này chưa đủ để báo cáo kết quả: cần phần cuối chứa
`eval_loss`, `train_loss`, trạng thái hoàn tất và output inference. Vì đây là
smoke config giới hạn dữ liệu/epoch, chỉ dùng để xác nhận pipeline kỹ thuật;
E0 vẫn phải chạy riêng bằng `configs/bgl_baseline.yaml`.

### Checkpoint 18 — xác minh run Kaggle thực tế dùng toàn bộ corpus

Output đầy đủ của run cho thấy:

```text
[dataset] pre-tokenizing 3,496,193 lines...
[train] examples: train=3,461,232 eval=34,961
0/54082 ... 54082/54082
eval_loss=0.03614
[train] saved final model -> outputs/BGL/model/final
```

Vì vậy run này **không phải smoke run 10,000 dòng** như cấu hình runtime mới.
Nó là run 1 epoch trên toàn bộ corpus, với 3,461,232 examples train và
34,961 examples eval. Run đã hoàn tất thành công về kỹ thuật:

```text
eval_loss=0.03614
loss cuối=0.03658
grad_norm cuối=0.6546
```

Nhưng vẫn chưa phải E0 chính thức vì mới có `epoch=1`, artifact nằm trong
`outputs/BGL/model/` thay vì thư mục baseline, và log cho biết Transformers đã
bỏ qua `warmup_ratio` cùng `logging_dir`. Không dùng artifact hoặc metric của
run này để báo cáo E0. Cần chạy lại bằng `configs/bgl_baseline.yaml`, xác nhận
đúng config/commit trước khi bắt đầu, và ghi nhận rõ nếu runtime Transformers
không hỗ trợ warmup ratio.

### Checkpoint 19 — cảnh báo scoring do smoke evaluation chỉ chứa normal

Kaggle xuất hiện nhiều cảnh báo sklearn:

```text
No positive samples in y_true
No positive class found in y_true
```

Nguyên nhân không phải model không tạo anomaly score. File BGL test được sắp
xếp như sau:

```text
903,310 dòng normal (label 0)
348,460 dòng anomaly (label 1)
```

Trong smoke config, `max_eval_samples: 10000` lấy 10,000 dòng đầu tiên. Vì
toàn bộ 10,000 dòng này nằm trong đoạn normal, `y_true` chỉ có class 0. Vì
vậy AUROC, recall và F1 của smoke inference không có ý nghĩa và không được
báo cáo.

E0 phải dùng `configs/bgl_baseline.yaml` với:

```yaml
max_eval_samples: null
```

để đánh giá toàn bộ 1,251,770 mẫu, gồm cả 903,310 normal và 348,460 anomaly.
Trước mỗi inference cần kiểm tra log:

```text
[infer] test lines: 1251770  labels: 1251770
```

và phân bố label phải là:

```text
label 0: 903310
label 1: 348460
```

### Checkpoint 20 — notebook chuyển sang E0 baseline theo paper

Đã cập nhật `kaggle_github_step_by_step.ipynb` để chạy trực tiếp baseline E0,
không còn tạo hoặc dùng `configs/kaggle_runtime.yaml`. Notebook hiện:

1. clone repository và in commit;
2. dùng `configs/bgl_baseline.yaml`;
3. chuẩn bị BGL và kiểm tra các file split/preprocess;
4. train tokenizer riêng tại `outputs/BGL/baseline_tokenizer`;
5. xác nhận vocabulary 1000 và MLM probe với xác suất mask 20%;
6. xác nhận đúng phân bố test labels `903310` normal + `348460` anomaly;
7. train full normal corpus trong 10 epochs với batch size 32;
8. chạy inference trên toàn bộ `1,251,770` test lines;
9. in các report loss/probability/top-k và đóng gói artifact E0.

Các điểm này bám theo paper: chỉ dùng normal logs khi train, không dùng NSP,
mask 20%, test có cả normal và anomaly, và anomaly score được tính bằng
predictive loss/probability với top-k aggregation. Smoke config vẫn tồn tại
cho chẩn đoán kỹ thuật nhưng không còn nằm trong đường chạy baseline notebook.

Đồng thời đã sửa tương thích `warmup_ratio`: nếu phiên bản Transformers không
có tham số này nhưng có `warmup_steps`, training sẽ tính số bước warmup tương
đương từ kích thước tập train, batch size, số epoch và truyền `warmup_steps`.
Như vậy Kaggle không còn âm thầm bỏ qua warmup của cấu hình E0.

### Checkpoint 21 — sửa truy cập nested config trong notebook

Cell xác nhận baseline từng dùng `baseline_cfg.get('train.num_train_epochs')`.
`Config.get()` chỉ đọc key cấp cao nhất; API truy cập nested dotted path của
project là `get_path()`. Vì vậy biểu thức cũ trả về `None` và gây:

```text
TypeError: int() argument must be a string ... not 'NoneType'
```

Đã sửa notebook dùng:

```python
baseline_cfg.get_path('train.num_train_epochs')
baseline_cfg.get_path('train.per_device_train_batch_size')
baseline_cfg.get_path('inference.max_eval_samples')
```

Đây chỉ là sửa validation cell, không thay đổi cấu hình hoặc kết quả baseline.

Các giai đoạn được ưu tiên triển khai:

1. Giữ LAnoBERT làm baseline bất biến.
2. Xây dựng Field-Value branch.
3. Xây dựng Ontology branch ở mức deterministic trước.
4. Xây dựng Gated Fusion.
5. Đánh giá cảnh báo sớm.
6. Sau khi có baseline đo được mới cân nhắc Qdrant.

## 3. Các phần đã thực hiện

### 3.1. Field-Value branch

File: [lanobert/field_value.py](lanobert/field_value.py)

Đã triển khai:

- `LogFields`: schema cho timestamp, severity, host, service, event code
  và response time.
- `ConstraintConfig`: cấu hình ngưỡng SLA.
- `ConstraintEngine`: kiểm tra:
  - response-time violation;
  - heartbeat interval violation;
  - critical severity;
  - blacklisted host.
- `HardState`: kết quả có thể audit.
- `HardState.vector`: vector nhị phân dạng số để đưa vào fusion.

### 3.2. Gated Fusion

File: [lanobert/fusion.py](lanobert/fusion.py)

Đã triển khai:

- Projection riêng cho baseline, semantic context và hard state.
- Adaptive gate dùng softmax để phân bổ trọng số giữa ba nhánh.
- Output anomaly score hợp nhất.
- `freeze_baseline()` để đóng băng LAnoBERT và chỉ huấn luyện lớp fusion.

### 3.3. Parser trường cấu trúc

File: [lanobert/structured.py](lanobert/structured.py)

Đã triển khai:

- Đọc log dạng `key=value`.
- Nhận diện timestamp BGL dạng:
  `YYYY-MM-DD-HH.MM.SS.microseconds`.
- Nhận diện severity.
- Nhận diện host/IP.
- Nhận diện response time từ các tên trường phổ biến.

### 3.4. Ontology retriever cục bộ

File: [lanobert/ontology.py](lanobert/ontology.py)

Đã triển khai:

- `OntologyDocument`.
- `TfidfOntologyRetriever`.
- Truy xuất top-k tài liệu bằng cosine similarity.

Đây là baseline ontology deterministic, chưa sử dụng Qdrant. Mục đích là
kiểm chứng thiết kế trước khi thêm hạ tầng vector database.

### 3.5. Metric cảnh báo sớm

File: [lanobert/early_detection.py](lanobert/early_detection.py)

Đã triển khai:

- `early_warning_rate()`.
- `detection_lead_time()`.
- Incident count.
- Detected incident count.
- Mean lead time.
- Median lead time.

### 3.6. Kiểm thử

Các file test:

- [tests/test_field_value.py](tests/test_field_value.py)
- [tests/test_fusion.py](tests/test_fusion.py)
- [tests/test_structured_and_early.py](tests/test_structured_and_early.py)
- [tests/test_ontology.py](tests/test_ontology.py)

Kết quả đã xác nhận:

```text
7 passed in 18.04s
```

Ngoài ra đã chạy smoke test trực tiếp bằng Python 3 và compile check cho
package `lanobert`, đều thành công.

## 4. Sửa lỗi tương thích macOS

### 4.1. Lock dữ liệu

File: [scripts/ensure_data.sh](scripts/ensure_data.sh)

macOS không có `flock` mặc định. Đã thay cơ chế lock bằng lock directory:

```bash
mkdir "$LOCK"
```

Các tiến trình khác sẽ chờ bằng vòng lặp `sleep` cho đến khi lock được giải
phóng. Script cũng có cleanup khi thoát.

Đã thêm xử lý file `.prep.lock` cũ còn sót lại từ phiên bản dùng `flock`.

### 4.2. Python executable

Các file:

- [scripts/ensure_data.sh](scripts/ensure_data.sh)
- [scripts/run_pipeline.sh](scripts/run_pipeline.sh)

Đã hỗ trợ tự động chọn:

```text
python nếu tồn tại
python3 nếu không có python
```

## 5. Trạng thái chạy pipeline BGL

Lệnh đã chạy:

```bash
cd /Users/ruby/dev/lv-new/LAnoBERT
bash scripts/run_pipeline.sh configs/bgl.yaml
```

Trạng thái tại thời điểm ghi tài liệu:

```text
[1-2/5] split + preprocess
```

Đã hoàn tất bước split:

```text
train normal: 3,496,193 dòng
test total:   1,251,770 dòng
test anomaly:   348,460 dòng
test normal:    903,310 dòng
```

Đang hoặc đã bắt đầu xử lý:

```text
data/BGL/BGL_train_normal.raw
    -> data/BGL/BGL_train_normal_parsed.log
```

File train đã được tạo trong quá trình xử lý và có kích thước khoảng 129 MB
tại thời điểm kiểm tra. Cần kiểm tra lại file đầu ra và log tiến trình trước
khi chuyển sang preprocessing test.

Không nên chạy thêm pipeline khác dùng cùng `data/BGL` khi lock sau đây còn
tồn tại:

```text
data/BGL/.prep.lock
```

## 6. Việc cần làm tiếp theo

### Bước 1: Xác nhận preprocessing BGL

Kiểm tra terminal đang chạy pipeline. Nếu tiến trình đã dừng, chạy lại:

```bash
cd /Users/ruby/dev/lv-new/LAnoBERT
bash scripts/ensure_data.sh configs/bgl.yaml
```

Script sẽ bỏ qua các file đã hoàn tất và tiếp tục file còn thiếu.

Kiểm tra đầu ra:

```bash
ls -lh data/BGL/BGL_train_normal_parsed.log
ls -lh data/BGL/BGL_test_parsed.log
ls -lh data/BGL/BGL_test_label.log
```

### Bước 2: Chạy tokenizer và baseline training

Sau khi preprocessing hoàn tất:

```bash
cd /Users/ruby/dev/lv-new/LAnoBERT
python3 -m lanobert.tokenizer --config configs/bgl.yaml
python3 -m lanobert.train --config configs/bgl.yaml
```

Hoặc tiếp tục toàn bộ pipeline:

```bash
bash scripts/run_pipeline.sh configs/bgl.yaml
```

### Bước 3: Chạy baseline inference

```bash
python3 -m lanobert.inference --config configs/bgl.yaml
```

Kết quả dự kiến nằm trong:

```text
outputs/BGL/results/
```

### Bước 4: Ghi nhận E0

Lưu các chỉ số baseline:

- AUROC.
- Best F1.
- Precision.
- Recall.
- Threshold.
- Thời gian suy luận.

### Bước 5: Tích hợp E1, E2 và E3

Sau khi E0 chạy ổn định:

1. E1: áp dụng `ConstraintEngine` cho từng log.
2. E2: truy xuất ontology cục bộ bằng `TfidfOntologyRetriever`.
3. E3: đưa baseline representation, context vector và hard-state vector
   vào `GatedFusion`.
4. Xuất cùng một schema kết quả cho cả bốn cấu hình.

### Bước 6: Đánh giá cảnh báo sớm

Đưa score, label và timestamp theo thứ tự thời gian vào:

```python
from lanobert.early_detection import detection_lead_time
```

Ghi nhận:

- số incident;
- số incident được cảnh báo trước;
- Early Warning Rate;
- mean lead time;
- median lead time.

### Bước 7: Chạy ablation và mở rộng

Chỉ sau khi E0--E3 chạy được trên fixture nhỏ mới:

- chạy toàn bộ BGL;
- lặp lại trên HDFS và Thunderbird nếu phù hợp;
- thêm latency/QPS;
- cân nhắc thay TF-IDF bằng embedding cục bộ;
- cân nhắc Qdrant;
- đóng gói artifact và cấu hình tái lập.

## 7. Nguyên tắc theo dõi kết quả

- Không thay đổi kiến trúc LAnoBERT baseline khi so sánh.
- Giữ chronological split, không random shuffle dữ liệu thời gian.
- Lưu config, seed và kết quả cho từng experiment.
- Không kết luận Hybrid-RAG tốt hơn nếu chưa có E0 đối chứng trên cùng dữ liệu.
- Không đưa Qdrant vào trước khi ontology baseline cục bộ có số liệu.

## 8. Quy tắc cập nhật tiến độ

Sau mỗi thay đổi đáng kể, lần chạy thực nghiệm, lỗi môi trường hoặc kết quả
kiểm thử, phải cập nhật file này để giữ một nhật ký tiến độ duy nhất cho
luận văn. Mỗi cập nhật nên ghi rõ:

- ngày/giờ hoặc mốc thực hiện;
- file hoặc module đã thay đổi;
- lệnh đã chạy;
- kết quả thành công/thất bại;
- bước tiếp theo và điều kiện để thực hiện bước đó.

## 9. Kaggle sử dụng trực tiếp GitHub

Đã tạo notebook:

- [kaggle_github_step_by_step.ipynb](kaggle_github_step_by_step.ipynb)

Notebook clone trực tiếp repository:

```text
https://github.com/rubyhcm/lv-new.git
```

Notebook được chia thành các cell độc lập để kiểm tra lần lượt:

1. GPU và môi trường Python.
2. Clone repository và ghi nhận commit.
3. Cài dependencies.
4. Chạy pytest và kiểm tra shell script.
5. Kiểm tra hoặc tải dữ liệu BGL.
6. Split và preprocess.
7. Train tokenizer.
8. Train LAnoBERT baseline.
9. Inference và đọc báo cáo.
10. Đóng gói artifact.

Đã xác minh cục bộ:

```text
Notebook JSON hợp lệ
Các code cell Python compile thành công
```

Raw BGL không được giả định là có trong GitHub. Khi chạy Kaggle cần một trong
hai phương án:

- bật Internet và đặt `DOWNLOAD_BGL = True`;
- cung cấp BGL từ một Kaggle Dataset riêng.

Artifact dự kiến sau khi chạy là:

```text
/kaggle/working/lanobert-results.zip
```

### Checkpoint 11 — notebook Kaggle gửi về chưa có execution logs

File `lv-new-1.ipynb` được đính kèm để kiểm tra hiện chỉ chứa phần Markdown
tiêu đề của notebook, không chứa các code cell hoặc output đã chạy. File output
đi kèm cũng chỉ là:

```json
[]
```

Do đó chưa thể đối chiếu các bước split, train, inference hoặc các metric
AUROC/F1. Cần tải notebook đã lưu cùng output từ Kaggle, hoặc gửi riêng log
của các cell train/inference.

Để xuất đúng trên Kaggle:

1. Chọn `File -> Download notebook`.
2. Đảm bảo notebook đã chạy và output vẫn hiển thị trước khi tải.
3. Nếu dùng `Save Version`, bật tùy chọn lưu output của notebook.
4. Gửi file `.ipynb` đã download, không chỉ bản notebook template.

Các output cần có để kiểm tra:

```text
[train] loaded tokenizer ...
[train] examples ...
[train] start
...
[eval:...] AUROC=...
best_F1=...
```

Nếu không muốn tải lại notebook, gửi nội dung của các cell hiển thị:

```python
print(runtime_config.read_text())
```

và toàn bộ output của cell `train` và `inference`.

### Checkpoint 12 — training không hợp lệ do tokenizer chỉ có 5 token

Log Kaggle cho thấy:

```text
[train] loaded tokenizer (vocab=5)
loss=0
grad_norm=0
eval_loss=nan
```

`vocab=5` đúng bằng số special token mặc định:

```text
[PAD], [UNK], [CLS], [SEP], [MASK]
```

Vì vậy model 86 triệu tham số đã chạy gần 5 giờ nhưng không học được dữ liệu.
Kết quả này phải loại bỏ, không dùng làm E0 baseline. Nguyên nhân cần xác
minh là file normalized train corpus rỗng/sai hoặc tokenizer cũ được tái sử
dụng sau khi đổi dữ liệu.

Đã thêm guard vào `lanobert/tokenizer.py`: nếu tokenizer học được không quá 5
token, pipeline sẽ dừng ngay với lỗi rõ ràng thay vì train model giả.

Trên Kaggle, cần xóa artifact tokenizer cũ và chẩn đoán trước khi train lại:

```python
import shutil
from pathlib import Path

train_file = PROJECT / 'data/BGL/BGL_train_normal_parsed.log'
tokenizer_dir = PROJECT / 'outputs/BGL/tokenizer'

print('train file:', train_file)
print('exists:', train_file.exists())
print('size MB:', train_file.stat().st_size / (1024**2) if train_file.exists() else None)
with train_file.open(errors='replace') as handle:
    samples = [next(handle, '').strip() for _ in range(5)]
print('samples:', samples)
print('non-empty samples:', sum(bool(x) for x in samples))

if tokenizer_dir.exists():
    shutil.rmtree(tokenizer_dir)
```

Chỉ chạy tokenizer lại khi sample có nội dung log hợp lệ và `size MB` khác 0.

### Checkpoint 13 — xác định lỗi tương thích Transformers làm mất vocabulary

Kiểm tra file local `outputs/BGL/tokenizer/BGL_LogBERT-vocab.txt` cho thấy file
hoàn toàn hợp lệ:

```text
1000 dòng
5 special tokens
995 learned tokens
```

Corpus normalized cũng hợp lệ:

```text
2,550,501 dòng
181,682,861 bytes
```

Nguyên nhân `vocab=5` là API của Transformers mới. Trong Transformers 5.16.1,
constructor của `BertTokenizerFast` nhận tham số `vocab`, không còn xử lý đúng
tham số `vocab_file`. Code cũ truyền `vocab_file=...`; tham số này bị bỏ qua
âm thầm, nên tokenizer chỉ còn `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`.

Đã sửa `lanobert/tokenizer.py` để truyền `vocab=vocab_file`. Kiểm tra local:

```text
BertTokenizerFast(vocab_file=...) -> vocab_size=5
BertTokenizerFast(vocab=...)      -> vocab_size=1000
```

Do đó cần clone commit mới trên Kaggle và chạy lại tokenizer/train. Không cần
preprocess BGL lại nếu file `BGL_train_normal_parsed.log` đã tồn tại hợp lệ.

Đã thêm một validation cell ngay trước cell train trong
`kaggle_github_step_by_step.ipynb`. Cell này kiểm tra project/config path,
corpus tồn tại và không rỗng, `tokenizer.vocab_size > 5`, sample có content
token, và MLM probe batch có label khác `-100`. Nếu một điều kiện thất bại,
notebook dừng trước khi khởi tạo training model.

## 10. Checklist chạy Kaggle tiếp theo

### Trên máy local

Đảm bảo các thay đổi mới nhất đã được push lên branch `main`:

```bash
cd /Users/ruby/dev/lv-new/LAnoBERT
git add README.md THESIS_PROGRESS.md kaggle_github_step_by_step.ipynb
git commit -m "Add GitHub-based Kaggle workflow"
git push origin main
```

### Trên Kaggle

1. Tạo notebook mới.
2. Bật `Settings -> Accelerator -> GPU`.
3. Bật `Internet` nếu muốn notebook tự tải BGL.
4. Mở hoặc tải `kaggle_github_step_by_step.ipynb`.
5. Chạy các cell tuần tự, không chạy đồng thời.
6. Trong cell cấu hình, dùng:

```python
DOWNLOAD_BGL = True
EPOCHS = 1
MAX_EVAL_SAMPLES = 10000
```

7. Xác nhận cell clone in ra commit mới nhất từ GitHub.
8. Xác nhận test pass trước khi chạy dữ liệu lớn.
9. Xác nhận các file preprocessing tồn tại trước khi train.
10. Sau khi inference hoàn tất, tải:
    `/kaggle/working/lanobert-results.zip`.

Sau lần chạy thử thành công, chạy lại với:

```python
DOWNLOAD_BGL = True
EPOCHS = 10
MAX_EVAL_SAMPLES = None
```

## 11. Trạng thái Kaggle thực tế

### Checkpoint 1 — cấu hình notebook

Đã xác nhận cell đầu tiên trên Kaggle chạy thành công với:

```text
GitHub: https://github.com/rubyhcm/lv-new.git
Working clone: /kaggle/working/lv-new
```

Điều này xác nhận notebook đã nhận đúng repository GitHub và thư mục làm việc
dự kiến. Bước tiếp theo là kiểm tra GPU/PyTorch, sau đó mới clone repository
và chạy các bước cài đặt, test, dữ liệu và huấn luyện.

### Checkpoint 2 — GPU và framework

Cell kiểm tra môi trường Kaggle đã cho kết quả:

```text
Python: 3.13.15
PyTorch: 2.11.0+cu128
CUDA available: True
GPU: Tesla T4
```

Môi trường GPU đã sẵn sàng để chạy LAnoBERT. Chưa chạy training ở checkpoint
này; cần clone và kiểm tra đúng commit GitHub trước.

### Checkpoint 3 — clone GitHub

Notebook đã clone thành công repository:

```text
Project: /kaggle/working/lv-new/LAnoBERT
Commit: 61664262630341c2d2022a889e87897bffdc09a2
```

Commit này là mốc mã nguồn cần ghi cùng các kết quả thực nghiệm Kaggle để
đảm bảo khả năng tái lập. Bước tiếp theo là cài dependencies trong môi trường
Kaggle.

### Checkpoint 4 — lỗi đường dẫn requirements

Cell cài dependencies thất bại vì notebook cũ dùng đường dẫn:

```text
/kaggle/working/LAnoBERT/requirements.txt
```

Trong khi cell clone thực tế tạo project tại:

```text
/kaggle/working/lv-new/LAnoBERT
```

Nguyên nhân là repository GitHub có thư mục gốc `lv-new`, bên trong mới có
`LAnoBERT`. Notebook đã được sửa để dùng biến `PROJECT` được xác định ở cell 3:

```python
%pip install -q -r {PROJECT / 'requirements.txt'} pytest
```

Để tiếp tục notebook đang mở, chạy lại cell 4 với lệnh trên. Không cần clone
lại repository.

### Checkpoint 4b — cài dependencies thành công

Cell 4 đã chạy lại thành công với đường dẫn `PROJECT`. Output chỉ còn:

```text
Note: you may need to restart the kernel to use updated packages.
```

Đây là thông báo chuẩn của `%pip` trong Jupyter, không phải lỗi cài đặt.
Không cần restart kernel nếu các import kiểm tra bên dưới đều thành công.

Lệnh kiểm tra:

```python
import torch
import transformers
import pydantic
import sklearn
import pytest

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("pydantic:", pydantic.__version__)
print("sklearn:", sklearn.__version__)
print("pytest:", pytest.__version__)
```

Bước tiếp theo là chuyển vào thư mục project và kiểm tra config cùng dữ liệu
BGL.

### Checkpoint 5 — project và dữ liệu BGL

Cell kiểm tra project đã cho kết quả:

```text
Current directory: /kaggle/working/lv-new/LAnoBERT
Config: True
BGL directory: False
```

Mã nguồn và config đã được nhận đúng. `BGL directory: False` là bình thường
vì raw log lớn không được commit vào GitHub. Cần tải BGL trong Kaggle bằng
cell dữ liệu với `DOWNLOAD_BGL = True` và Kaggle Internet được bật.

### Checkpoint 5b — BGL đã import vào Kaggle Input

Ảnh Kaggle xác nhận dữ liệu đã được import tại dataset:

```text
/kaggle/input/dataset-input-0409/BGL/
```

Trong đó có `BGL.log` và các file split/preprocess đã tạo sẵn. Cell 5/6 vẫn
không thấy vì notebook đang tìm tại:

```text
/kaggle/working/lv-new/LAnoBERT/data/BGL/
```

Đã cập nhật notebook để tự quét:

```text
/kaggle/input/*/BGL/BGL.log
```

và copy các file BGL vào thư mục project trước khi chạy các bước tiếp theo.
Trong notebook Kaggle hiện tại, có thể chạy thủ công đoạn sau ở cell dữ liệu:

```python
from pathlib import Path
import shutil

source_bgl = Path('/kaggle/input/dataset-input-0409/BGL')
target_bgl = PROJECT / 'data/BGL'
target_bgl.mkdir(parents=True, exist_ok=True)
for source in source_bgl.iterdir():
    target = target_bgl / source.name
    if source.is_file() and not target.exists():
        shutil.copy2(source, target)
print(*sorted(str(p) for p in target_bgl.iterdir()), sep='\n')
```

Sau đó kiểm tra:

```python
raw_log = PROJECT / 'data/BGL/BGL.log'
assert raw_log.exists()
```

### Checkpoint 6 — lỗi clone khi chạy lại cell 3

Khi chạy lại cell 3, Git báo:

```text
fatal: Unable to read current working directory: No such file or directory
```

Nguyên nhân: kernel đang đứng trong thư mục clone cũ, nhưng cell 3 xóa thư mục
đó trước khi gọi `git clone`. Notebook đã được sửa để chạy:

```python
os.chdir('/kaggle/working')
```

trước khi xóa và clone lại repository.

Để sửa notebook Kaggle hiện tại ngay lập tức, chạy cell này trước khi chạy lại
cell clone:

```python
import os
os.chdir('/kaggle/working')
```

Sau đó chạy lại toàn bộ cell 3. Không cần restart kernel và không cần xóa
dataset BGL trong `/kaggle/input`.

### Checkpoint 6b — cell dữ liệu không tìm thấy BGL

Cell dữ liệu báo:

```text
AssertionError: Thiếu data/BGL/BGL.log; không tìm thấy BGL trong /kaggle/input.
```

Nguyên nhân là cell dùng đường dẫn tương đối sau khi clone lại, hoặc dataset
có cấu trúc thư mục sâu hơn mẫu `*/BGL/BGL.log`. Notebook đã được sửa để:

- chuyển về `PROJECT` trước khi kiểm tra;
- dùng đường dẫn tuyệt đối `PROJECT / 'data/BGL/BGL.log'`;
- quét đệ quy `/kaggle/input/**/BGL.log`;
- copy toàn bộ thư mục chứa `BGL.log` vào project.

Trong notebook Kaggle hiện tại, chạy thủ công trước cell dữ liệu:

```python
PROJECT = Path('/kaggle/working/lv-new/LAnoBERT').resolve()
os.chdir(PROJECT)
print('PROJECT:', PROJECT)
print('BGL candidates:', list(Path('/kaggle/input').rglob('BGL.log')))
```

Nếu danh sách có file BGL, chạy:

```python
candidates = list(Path('/kaggle/input').rglob('BGL.log'))
assert candidates, 'Kaggle chưa mount dataset chứa BGL.log'
source_bgl = candidates[0].parent
target_bgl = PROJECT / 'data/BGL'
target_bgl.mkdir(parents=True, exist_ok=True)
for source in source_bgl.iterdir():
    if source.is_file():
        shutil.copy2(source, target_bgl / source.name)
print('Copied:', source_bgl, '->', target_bgl)
```

### Ghi chú về `kaggle_runtime.yaml`

File `kaggle_runtime.yaml` không nằm sẵn trên GitHub. Nó được notebook tạo
tại bước **Tạo cấu hình Kaggle**, sau khi đã clone project và chuyển vào đúng
thư mục:

```text
/kaggle/working/lv-new/LAnoBERT/configs/kaggle_runtime.yaml
```

Cell tạo file:

```python
import yaml

with open(CONFIG, encoding='utf-8') as handle:
    runtime = yaml.safe_load(handle)

runtime.setdefault('train', {})['num_train_epochs'] = EPOCHS
runtime.setdefault('inference', {})['max_eval_samples'] = MAX_EVAL_SAMPLES

with open('configs/kaggle_runtime.yaml', 'w', encoding='utf-8') as handle:
    yaml.safe_dump(runtime, handle, sort_keys=False)
```

Kiểm tra file:

```python
runtime_config = PROJECT / 'configs/kaggle_runtime.yaml'
print(runtime_config)
print(runtime_config.exists())
```

Nếu chưa có file, hãy chạy cell tạo cấu hình trước cell train và inference.

### Checkpoint 7 — tạo runtime config local

Đã tạo file cấu hình chạy thử Kaggle tại:

```text
configs/kaggle_runtime.yaml
```

Các thiết lập smoke run:

```text
num_train_epochs: 1
max_eval_samples: 10000
seed: 42
```

File này có thể dùng trực tiếp từ local:

```bash
cd /Users/ruby/dev/lv-new/LAnoBERT
python3 -m lanobert.train --config configs/kaggle_runtime.yaml
python3 -m lanobert.inference --config configs/kaggle_runtime.yaml
```

Khi chạy trên Kaggle, file sẽ nằm tương ứng tại:

```text
/kaggle/working/lv-new/LAnoBERT/configs/kaggle_runtime.yaml
```

### Checkpoint 8 — lỗi `overwrite_output_dir` trên Kaggle

Kaggle dừng ở bước khởi tạo `TrainingArguments` với:

```text
TypeError: TrainingArguments.__init__() got an unexpected keyword argument
'overwrite_output_dir'
```

Split và preprocessing đã thành công trước lỗi:

```text
train=3,461,232
eval=34,961
```

Nguyên nhân là phiên bản `transformers` trong runtime Kaggle không hỗ trợ
tham số này. Tham số không cần thiết cho pipeline hiện tại vì `output_dir`
đã được truyền rõ ràng. Đã xóa `overwrite_output_dir=True` khỏi
`lanobert/train.py`.

Sau khi push commit sửa lên GitHub, cần clone lại repository trên Kaggle rồi
chạy lại cell train. Không cần chạy lại preprocessing nếu các file trong
`data/BGL/` vẫn còn.

### Checkpoint 9 — lỗi `warmup_ratio` trên Kaggle

Sau khi bỏ `overwrite_output_dir`, Kaggle tiếp tục báo:

```text
TypeError: TrainingArguments.__init__() got an unexpected keyword argument
'warmup_ratio'
```

Điều này xác nhận API `TrainingArguments` trong runtime Kaggle khác với phiên
bản dự kiến trong requirements. Đã cập nhật `lanobert/train.py` để:

- đọc chữ ký constructor bằng `inspect.signature`;
- ánh xạ `eval_strategy` sang `evaluation_strategy` nếu cần;
- chỉ truyền tham số được phiên bản hiện tại hỗ trợ;
- in danh sách tham số bị bỏ qua để không che giấu khác biệt môi trường.

Các bước split, preprocessing và tokenization vẫn hợp lệ. Sau khi push commit
mới, clone lại GitHub trên Kaggle rồi chạy lại cell train.

### Ghi chú về số epoch: smoke run và baseline

`EPOCHS = 1` trong notebook/Kaggle runtime chỉ là **smoke run**, dùng để kiểm
tra:

- pipeline có chạy hết từ dữ liệu đến inference hay không;
- GPU và dependencies có hoạt động hay không;
- model/checkpoint có được tạo đúng hay không;
- các lỗi tương thích `transformers` có còn hay không.

Đây **không phải** cấu hình baseline dùng cho luận văn.

Cấu hình baseline chính thức trong [configs/bgl.yaml](configs/bgl.yaml) là:

```yaml
train:
  num_train_epochs: 10
```

Sau khi smoke run thành công, phải chạy lại Kaggle với:

```python
EPOCHS = 10
MAX_EVAL_SAMPLES = None
```

Kết quả của lần chạy 10 epochs trên toàn bộ test set mới được ghi là E0
LAnoBERT baseline. Không dùng kết quả của lần chạy 1 epoch/10.000 mẫu để
đối chiếu chính thức với E1--E3.

### Checkpoint 10 — config baseline chính thức

Đã tạo config riêng cho pipeline baseline:

```text
configs/bgl_baseline.yaml
```

Config này giữ cùng preprocessing, tokenizer, batch size, learning rate,
seed và scoring như `configs/bgl.yaml`, nhưng tách riêng artifact và đặt:

```text
run_name: bgl_baseline_e0
num_train_epochs: 10
max_eval_samples: null
```

Artifact baseline sẽ được ghi riêng tại:

```text
outputs/BGL/baseline_tokenizer/
outputs/BGL/baseline_model/
outputs/BGL/baseline_results/
```

Sau khi smoke run 1 epoch pass, chạy baseline bằng:

```bash
cd /Users/ruby/dev/lv-new/LAnoBERT
python3 -m lanobert.tokenizer --config configs/bgl_baseline.yaml
python3 -m lanobert.train --config configs/bgl_baseline.yaml
python3 -m lanobert.inference --config configs/bgl_baseline.yaml
```

Trên Kaggle, dùng cùng các lệnh qua `subprocess.run(..., cwd=PROJECT)` và
truyền `configs/bgl_baseline.yaml`. Không chạy `ensure_data.sh` lại nếu các
file dữ liệu đã tồn tại; config baseline dùng lại các file BGL đã preprocess.

### Checkpoint 14 — Kaggle validation pass

Cell kiểm tra trước training đã thành công trên Kaggle:

```text
Project: /kaggle/working/lv-new/LAnoBERT
Tokenizer vocab size: 1000
Train corpus exists: True
Train corpus size: 264.44 MB
Dataset examples: 3,496,193
Masked labels in probe batch: 22
Validation OK: có thể bắt đầu training.
```

Điều này xác nhận lỗi tokenizer `vocab=5` đã được sửa đúng và dữ liệu MLM
không bị rỗng. Có thể chạy smoke training 1 epoch trong
`configs/kaggle_runtime.yaml`. Chỉ chấp nhận run nếu log có `loss` hữu hạn và
lớn hơn 0, `grad_norm` khác 0, và `eval_loss` không phải `nan`. Sau smoke run
thành công mới chạy E0 bằng `configs/bgl_baseline.yaml` với 10 epochs.

### Checkpoint 15 — Kaggle SIGKILL do giới hạn bộ nhớ smoke run

Sau validation, training bị dừng với `Signals.SIGKILL: 9`. Đây không phải
exception của Python hay lỗi tokenizer. Code cũ pre-tokenize toàn bộ
3,496,193 dòng và giữ các list `input_ids`, `attention_mask` và
`special_tokens_mask` trong RAM; cell validation trước đó cũng tạo dataset
toàn bộ trước khi gọi subprocess train. Kaggle có thể giết tiến trình khi
vượt giới hạn RAM.

Đã sửa `LogLineDataset` hỗ trợ `limit`, truyền `train.max_train_samples` từ
config, và đặt smoke config ở 10,000 dòng. Batch smoke cũng giảm từ 32 xuống
8 để giảm áp lực VRAM. Notebook validation dùng cùng giới hạn 10,000 dòng.
Config E0 baseline không bị thay đổi và vẫn dùng toàn bộ dữ liệu với batch 32.

### Checkpoint 16 — phân biệt smoke test và baseline khoa học

Không được dùng `configs/kaggle_runtime.yaml` để báo cáo baseline. Đây chỉ là
smoke test kỹ thuật với:

```text
max_train_samples: 10000
num_train_epochs: 1
per_device_train_batch_size: 8
max_eval_samples: 10000
```

Các thay đổi sau không được coi là thay đổi phương pháp khi chạy E0:

- sửa `BertTokenizerFast` để truyền đúng vocabulary trong Transformers hiện
  tại; nếu không, model chỉ có 5 special tokens và run hoàn toàn không hợp lệ;
- lọc keyword theo signature của `TrainingArguments` để tương thích phiên bản
  Transformers trên Kaggle;
- validation guard trước training.

Ngược lại, giới hạn số dòng, số epoch, batch size hoặc evaluation samples là
thay đổi thực nghiệm và chỉ được dùng cho smoke test. E0 phải chạy bằng
`configs/bgl_baseline.yaml`, dùng toàn bộ corpus, 10 epochs, batch 32 và toàn
 bộ evaluation set. Kết quả smoke test không được đưa vào bảng kết quả luận
 văn.
