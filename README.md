# Speech-to-text y khoa tiếng Việt

Mã được tách thành **huấn luyện**, **đánh giá dataset**, **đánh giá audio/video** và **suy luận**.

## Ba lệnh chính

| File | Công việc |
|---|---|
| `train.py` | Huấn luyện VietMed, kiểm tra dev, chọn và xuất checkpoint |
| `test_datasets.py` | Chấm WER/CER/RTF trên VIVOS, FLEURS, VietMed; nhận model gốc hoặc checkpoint fine-tune |
| `test_audio_video.py` | Quy trình riêng cho podcast **5 phút 54 giây** |

`train.py` không tự chấm tập test. Các file trong `tests/` là unit test mã nguồn, khác với lệnh đánh giá ASR ở trên.

## Cấu trúc mã

```text
SpeechToText_Medical/
├── train.py
├── test_datasets.py
├── test_audio_video.py
├── run_app.ps1
├── asr_common/                  # Chuẩn hóa văn bản và đường dẫn dùng chung
├── training/
│   └── vietmed/
│       ├── train.py             # Recipe huấn luyện, không tự chạy khi import
│       ├── train.ipynb          # Notebook train/dev và export model
│       └── kernel-metadata.json
├── evaluation/
│   ├── datasets/                # Benchmark dataset và kiểm chứng kết quả
│   │   └── kaggle/              # Job ba model gốc trên VietMed
│   ├── audio_video/             # Chuẩn bị/chỉnh reference/chấm podcast 5m54s
│   │   ├── cli.py
│   │   ├── prepare_input.py
│   │   ├── revise_reference.py
│   │   ├── score.py
│   │   ├── kaggle/              # Job Kyle gốc + PhoWhisper-large
│   │   └── legacy/              # Thử nghiệm video cũ dùng model fine-tune
│   └── decoding/                # Chẩn đoán và sửa decoding, tách khỏi train
├── inference/                   # Web app và nhận dạng audio, không chấm WER
├── tools/kaggle/                # Theo dõi job, tải artifact, chẩn đoán môi trường
├── tests/unit/                  # Unit test
├── video_eval/                  # Audio, phụ đề và kết quả podcast
│   ├── source_hls.wav
│   ├── reference_subtitle.original.txt
│   ├── reference_subtitle.corrected.txt
│   ├── kaggle_input/            # Payload dataset riêng tư
│   └── two_model_results/       # WER/CER và transcript hai model
├── results/                     # Kết quả benchmark và output Kaggle đã tải
├── archive/                     # Bản publish notebook cũ để đối chiếu
└── output/pdf/                  # PDF báo cáo
```

Các thư mục `kaggle_v5_report/`, `kaggle_v5_output/`, `local_decoding_fix/` và các thư mục log cũ là artifact thực nghiệm đã lưu. Mã chạy hiện tại nằm trong các package ở trên.

## Nội dung lưu trên GitHub

Repository lưu mã nguồn, notebook, cấu hình, unit test và các bảng kết quả tổng hợp. Audio/video, phụ đề tham chiếu, dataset, checkpoint, transcript từng mẫu, cache và thông tin đăng nhập được loại trừ bằng `.gitignore`; chúng vẫn được giữ trên máy hiện tại. Vì vậy cây thư mục trên mô tả cả artifact cục bộ, không phải mọi file đều có trong bản clone.

Unit test chạy bằng dữ liệu giả lập. Các lệnh chấm model hoặc `test_audio_video.py verify` cần thêm audio, reference và output tương ứng từ lần chạy; bản clone chỉ có bảng tổng hợp thì chưa đủ để kiểm chứng lại toàn bộ số liệu. Chuẩn bị input riêng trước khi chạy lại theo hướng dẫn trong từng thư mục.

## Môi trường

Python 3.10 là môi trường cục bộ đã kiểm thử trên máy này:

```powershell
$python = 'E:\SpeechToText_Medical_runtime\.venv\Scripts\python.exe'
& $python -m pip install -r requirements.txt
```

Các lệnh dưới đây chạy từ thư mục gốc dự án. Có thể đặt `ASR_RUNTIME_DIR` nếu môi trường/cache model nằm ở nơi khác.

- `requirements.txt`: thư viện ASR cơ bản.
- `requirements/evaluation.txt`: bổ sung Transformers/PhoWhisper.
- `requirements/ctranslate2.txt`: phương án Whisper lượng tử hóa trên GPU 4 GB.
- `requirements/tools.txt`: công cụ Kaggle.

Kaggle runner tự cài phiên bản đã pin trong mã; không dùng bộ yêu cầu cục bộ để thay môi trường Kaggle.

## 1. Huấn luyện

Notebook chính: [training/vietmed/train.ipynb](training/vietmed/train.ipynb).

```powershell
python train.py --help
kaggle kernels push -p training/vietmed --accelerator NvidiaTeslaT4
```

Recipe dùng train để tối ưu trọng số, dev để chọn checkpoint và xuất `fastconformer_vietmed_best.nemo`. Script Python tương đương chỉ thực thi khi truyền `--run`; đây là recipe dành cho môi trường Kaggle đã cấu hình input/GPU.

Chi tiết: [training/README.md](training/README.md).

## 2. Đánh giá dataset

```powershell
# Kyle gốc, thử nhanh 10 mẫu mỗi dataset
python test_datasets.py --datasets vivos fleurs vietmed --samples 10 --batch-size 1

# Chấm checkpoint fine-tune riêng, không huấn luyện lại
python test_datasets.py --checkpoint path/to/fastconformer_vietmed_best.nemo --datasets vietmed --samples 3437 --output-dir results/finetuned-test

# Job ba model gốc trên toàn bộ VietMed test
kaggle kernels push -p evaluation/datasets/kaggle --accelerator NvidiaTeslaT4
```

Ba model trong bảng chính: `kyle/vi-asr-fastconformer-114m`, `vinai/PhoWhisper-large`, `openai/whisper-large-v3`.

| Dataset | Phạm vi |
|---|---|
| VIVOS test | Tiếng đọc phòng thu |
| FLEURS vi-VN test | Tiếng đọc đa dạng |
| VietMed test | Hội thoại/y khoa; trọng tâm báo cáo |

[Kết quả benchmark ba model](results/medical-main/comparison.md) và [báo cáo ngắn gọn](REPORT.md).

## 3. Test riêng audio/video 5 phút 54 giây

```powershell
python test_audio_video.py --help
python test_audio_video.py results       # Xem số liệu đã lưu
python test_audio_video.py verify        # Kiểm tra hash và tính lại WER/CER, không sửa file
python test_audio_video.py prepare       # Chuẩn bị input và phụ đề sạch
python test_audio_video.py score         # Chấm lại transcript đã có với reference hiện tại
```

Job hai model trên Kaggle:

```powershell
kaggle kernels push -p evaluation/audio_video/kaggle --accelerator NvidiaTeslaT4
```

Dataset riêng tư đã có; payload tương ứng ở `video_eval/kaggle_input/`. Khi cần tạo dataset lần đầu, dùng `kaggle datasets create -p video_eval/kaggle_input -t`. Input được pin hash trong runner; thay input phải cập nhật manifest và hash tương ứng.

Chi tiết workflow: [evaluation/audio_video/README.md](evaluation/audio_video/README.md).

**Tình trạng reference:** phụ đề đã hiệu chỉnh với hỗ trợ ASR độc lập, còn đoạn cần nghe xác nhận; các số podcast hiện là số đo tạm tính theo bản reference đã nêu. Xem [review phụ đề](video_eval/SUBTITLE_REVIEW.md) và [kết quả podcast](video_eval/two_model_results/comparison.md).

## 4. Giao diện nhận dạng

```powershell
.\run_app.ps1
# Hoặc trong môi trường Python đã cài thư viện
python -m inference.app
```

Mở `http://127.0.0.1:7860` để thử audio/microphone. Nhận dạng file dài bằng `python -m inference.transcribe_youtube --help`.

## 5. Kiểm tra mã nguồn

```powershell
python -m unittest discover -s tests -v
```

Các lệnh `--help`, `results`, `verify` không khởi động huấn luyện hay chạy lại model.

WER/CER giữ nguyên chữ số và dấu tiếng Việt. Số liệu fine-tune + sửa decoding được ghi riêng vì đã dùng một phần test để chọn cấu hình. Xem [giải thích decoding và giao thức](DECODING_AND_BENCHMARK.md).
