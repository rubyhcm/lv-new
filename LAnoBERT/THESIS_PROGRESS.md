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
