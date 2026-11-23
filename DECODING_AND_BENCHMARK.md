# Sửa decoding và benchmark ASR y khoa

Benchmark chính đã hoàn tất ngày **19/11/2026**. Toàn bộ 3 model đã chạy đủ
3.437 mẫu; WER/CER đã được tính lại từ transcript và kiểm chứng ID/hash/reference.
Xem [bảng chính](results/medical-main/comparison.md),
[CSV](results/medical-main/comparison.csv) và
[kết quả kiểm chứng](results/medical-main/verification.json).

PhoWhisper-large: WER **24,14%**, CER **20,24%**. Kyle gốc: WER **26,19%**,
CER **20,44%**. Whisper-large-v3: WER **74,30%**, CER **58,83%**.
Đây là kết quả của cấu hình benchmark này, không phải xếp hạng tốt nhất cho
mọi audio y khoa. Whisper-v3 thường sinh câu lạc nội dung; ba mẫu đầu đã được
đối chiếu với pipeline Transformers chính thức dùng eager attention, cho cùng
transcript với lần chạy SDPA. Không điều chỉnh decoding theo các lỗi test này.

Hai Whisper được đo ở Kaggle version 2; Kyle được đo ở version 4 sau sửa lỗi
môi trường đọc dữ liệu. Cùng loại GPU T4, FP16, batch size 4; không phải cùng
một phiên GPU. Log các phiên được lưu tại `results/medical-kaggle/history`.

## Lần sửa trước đây đã làm gì?

Nguồn: `evaluation/decoding/local.py`, `local_decoding_fix/local_decoding_fix_report.json`.
Model được sửa là checkpoint Kyle đã continued fine-tune trên VietMed (F1),
không phải checkpoint Kyle gốc. Việc sửa không huấn luyện lại trọng số.

Pipeline ban đầu tạo dự đoán trên 3.437 mẫu VietMed test. Script đánh dấu một
transcript đáng ngờ khi ít nhất một điều kiện đúng:

- Số từ/giây > 8.
- Cùng một từ lặp liên tiếp ít nhất 4 lần.
- Số từ > max(40, 7 × thời lượng audio tính bằng giây).

869 mẫu bị đánh dấu được giải mã lại bằng:

```yaml
strategy: greedy_batch
greedy:
  max_symbols_per_step: 1
  loop_labels: false
  use_cuda_graph_decoder: false
```

`max_symbols_per_step=1` giới hạn số token RNNT được phát tại một bước encoder,
hạn chế decoder phát chuỗi token kéo dài/lặp. Đây không phải giới hạn một từ
cho cả câu. Hai cờ còn lại chọn đường giải mã không dùng label loop/CUDA graph;
thử nghiệm cũ thay cả ba cờ, nên không chứng minh riêng cờ nào gây ra mức cải thiện.

Transcript của 2.568 mẫu còn lại được giữ từ lần giải mã ban đầu. Do đó kết quả
26,91% là của **pipeline thích ứng gồm hai lượt**, không phải bằng chứng rằng
giải mã toàn bộ tập bằng file `.nemo` đã lưu sẽ đạt đúng 26,91%.

| Chỉ số | Ban đầu | Sau sửa thích ứng |
|---|---:|---:|
| Số mẫu test | 3.437 | 3.437 |
| WER | 177,47% | 26,91% |
| CER | 161,21% | 21,86% |
| Mẫu có dấu hiệu lặp theo heuristic | 869 | 6 |

**Giới hạn khi báo cáo:** cấu hình được chọn trên 8 mẫu lỗi từ test và kiểm tra
thêm 24 mẫu cũng thuộc test (vị trí 1000–1023). Kết quả này đã chịu ảnh hưởng của
test khi chọn cấu hình, cần trình bày là thí nghiệm chẩn đoán/sửa lỗi. Thời gian
179,956 giây và RTF 0,00830 trong summary cũ chỉ tính lượt giải mã lại 869 mẫu,
nhưng chia cho thời lượng toàn bộ test; chúng không phải thời gian/RTF đầy đủ
của pipeline hai lượt. Hai lượt còn chạy trên GPU khác nhau (T4 và RTX 3050),
không cộng các con số này để xếp hạng tốc độ với benchmark mới.

## Bảng so sánh chính mới

Ba checkpoint gốc được chọn:

1. `kyle/vi-asr-fastconformer-114m`.
2. `vinai/PhoWhisper-large`.
3. `openai/whisper-large-v3`.

Bảng chính được chạy trên notebook Kaggle riêng tư
<https://www.kaggle.com/code/k63ftutrnththonguyn/vietmed-three-model-benchmark>.
Người dùng đã cho phép gửi `evaluation/datasets/kaggle/benchmark.py` và
`kernel-metadata.json` tới đích này và chạy GPU. Không gửi token/audio cá nhân.
Ba model dùng cùng GPU T4, cùng FP16 và batch size 4; hai Whisper dùng Transformers
SDPA thay vì lượng tử hóa. Script chỉ đánh giá, không huấn luyện.

Sau khi tải và kiểm chứng, bảng đo nằm tại `results/medical-main/comparison.md`;
bản dữ liệu máy đọc là `comparison.json` và `comparison.csv`. Chỉ công bố khi
cả ba model hoàn tất đúng toàn bộ test; không điền số model card/subset vào bảng.

Giao thức:

- Cùng toàn bộ 3.437 mẫu test trong `local_decoding_fix/test.parquet`, giữ thứ tự.
- Kiểm tra SHA-256 audio gốc so với parquet, resample một lần thành bộ WAV PCM16
  mono 16 kHz dùng chung; lưu hash audio và manifest.
- Cùng chuẩn hóa NFC, lowercase, đổi dấu câu/ký hiệu thành khoảng trắng, gom
  khoảng trắng; giữ nguyên chữ số và dấu tiếng Việt.
- WER/CER tính trên toàn corpus. Lưu transcript từng mẫu và số lỗi S/D/I.
  WER dùng đơn vị tách bằng khoảng trắng (không chạy bộ tách từ ghép tiếng Việt).
- Kyle dùng decoding gốc của checkpoint, không áp dụng cấu hình đã chọn trên test.
- Hai Whisper dùng cùng greedy decoding, Vietnamese/transcribe, temperature=0,
  không VAD, không prompt, không truyền transcript mẫu trước sang mẫu sau,
  không fallback theo confidence của đầu ra.
- Cả ba checkpoint được pin commit; Kyle kiểm chứng SHA-256 tên/dtype/shape/bytes
  của 705 tensor so với checkpoint Hugging Face gốc. Không so hash cả archive vì
  NeMo `save_to` đóng gói lại làm đổi hash archive. Hai Whisper đọc checkpoint gốc
  từ repository tác giả.
- Trong benchmark Kaggle, RTF gồm IO, tiền xử lý và decoding, loại trừ load model
  và warm-up một mẫu. Kyle dùng NeMo, Whisper dùng Transformers SDPA; backend
  vẫn có thể ảnh hưởng tốc độ.
- Chưa có recall thuật ngữ y khoa hợp lệ vì chưa có danh mục thuật ngữ đã kiểm chứng.
  WER/CER trên VietMed đo lỗi phiên âm miền y khoa, không tự chứng minh độ đúng
  riêng của tên thuốc, liều lượng, phủ định.

## Chạy lại trên Kaggle

```powershell
kaggle kernels push -p evaluation/datasets/kaggle --accelerator NvidiaTeslaT4 --timeout 14400
kaggle kernels status k63ftutrnththonguyn/vietmed-three-model-benchmark
kaggle kernels output k63ftutrnththonguyn/vietmed-three-model-benchmark -p results\medical-kaggle --file-pattern '(^medical_benchmark/.*\.(json|jsonl|csv|md)$|\.log$)' -o -q
& E:\SpeechToText_Medical_runtime\.venv\Scripts\python.exe -m evaluation.datasets.collect_results
```

`evaluation/datasets/collect_results.py` kiểm tra 3.437 ID/hash/reference giống nhau giữa
ba lần chạy; đối chiếu hash reference với test đã kiểm tra cục bộ; tính lại
WER/CER/S/D/I từ transcript; kiểm tra precision, batch, GPU và mẫu số RTF, rồi mới
chép số liệu sang bảng chính. Log và output tải về vẫn được giữ nguyên riêng.
Với các lần chạy sửa môi trường riêng lẻ, giữ lại thư mục kết quả của model đã
hoàn tất; script ghép lại bảng từ số liệu đã kiểm chứng thay vì dùng bảng thiếu
model của phiên sửa lỗi cuối. Script Kaggle trong source hiện được đặt lại để
chạy cả ba model khi tái lập từ đầu.

## Phương án cục bộ (GPU 4 GB)

Môi trường cũ: `E:\SpeechToText_Medical_runtime\.venv\Scripts\python.exe`.
Thư viện và model bổ sung nằm trong `.benchmark_runtime`, không sửa môi trường cũ.
Phương án dự phòng này dùng CTranslate2 INT8/FP16 cho Whisper, khác cấu hình bảng
Kaggle chính; output tách tại `results/medical-local`. Tải local đã được tạm dừng
khi chuyển sang Kaggle, nên chưa có số liệu Whisper cục bộ.

```powershell
$python = 'E:\SpeechToText_Medical_runtime\.venv\Scripts\python.exe'
$env:PYTHONPATH = Join-Path $PWD '.benchmark_runtime\deps'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:HF_HOME = Join-Path $PWD '.benchmark_runtime\hf'
$env:TEMP = Join-Path $PWD '.benchmark_runtime\tmp'
$env:TMP = $env:TEMP
& $python -m evaluation.datasets.benchmark prepare
& $python -m evaluation.datasets.prepare_models phowhisper
& $python -m evaluation.datasets.prepare_models whisper
& $python -m evaluation.datasets.benchmark run --model kyle
& $python -m evaluation.datasets.benchmark run --model phowhisper
& $python -m evaluation.datasets.benchmark run --model whisper
& $python -m evaluation.datasets.benchmark report
```

Các lệnh `run` tiếp tục từ những batch đã lưu nếu bị gián đoạn; summary chỉ được
ghi khi chạy xong toàn bộ mẫu. `--limit N` tạo smoke run ở thư mục riêng, không
được đưa vào bảng toàn tập. Không thay đổi giao thức khi resume một lần chạy cũ.
