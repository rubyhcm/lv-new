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
