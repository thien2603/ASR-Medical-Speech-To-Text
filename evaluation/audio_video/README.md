# Test podcast 5 phút 54 giây

Entry point riêng ở gốc dự án: **`test_audio_video.py`**.

| Bước | Lệnh |
|---|---|
| Chuẩn bị audio/phụ đề sạch | `python test_audio_video.py prepare` |
| Tạo lại draft reference từ bằng chứng độc lập | `python test_audio_video.py revise-reference` |
| Chấm lại transcript đã có | `python test_audio_video.py score` |
| Xem bảng kết quả | `python test_audio_video.py results` |
| Kiểm chứng lại mà không ghi file | `python test_audio_video.py verify` |
| Chạy Kyle cục bộ | `python test_audio_video.py run-local --checkpoint path/to/original.nemo` |

`revise-reference` tạo lại bản nháp từ bản sạch và hai nguồn ASR độc lập. Không dùng lệnh này để ghi đè một reference đã được người nghe duyệt mà chưa lưu lại sửa đổi của người nghe.

## Các file liên quan

- `video_eval/source_hls.wav`: audio gốc 354,267 giây.
- `video_eval/reference_subtitle.original.txt`: phụ đề nguyên bản.
- `video_eval/reference_subtitle.cleaned.txt`: chỉ sạch định dạng, giữ chuỗi từ.
- `video_eval/reference_subtitle.corrected.txt`: bản hiệu chỉnh đang chấm.
- `video_eval/subtitle_correction_audit.json`: từng sửa đổi và nguồn bằng chứng.
- `video_eval/kaggle_input/`: audio 16 kHz, reference ban đầu và manifest có hash.
- `video_eval/two_model_results/`: transcript, WER/CER và alignment hai model.
- `results/podcast-kaggle/podcast_evaluation/`: output Kaggle đã tải về.

Đường dẫn trong danh sách trên tính từ gốc repository.

## Chạy lại trên Kaggle

```powershell
kaggle kernels push -p evaluation/audio_video/kaggle --accelerator NvidiaTeslaT4
kaggle kernels status k63ftutrnththonguyn/medical-podcast-two-model-evaluation
kaggle kernels output k63ftutrnththonguyn/medical-podcast-two-model-evaluation -p results/podcast-kaggle --file-pattern '^podcast_evaluation/.*\.(json|txt|csv)$' --page-size 200 -o -q
python test_audio_video.py score
```

`kaggle/evaluate.py` dùng hai model gốc Kyle và PhoWhisper-large, đồng thời lấy bằng chứng từ các ASR khác để hỗ trợ sửa reference. Input/chunk đã pin hash; cả hai ứng viên nhận cùng waveform. Đổi reference chỉ cần chấm lại transcript, không cần chạy lại model.

Phụ đề hiện chưa được nghe duyệt toàn bộ; hai cụm ở 2:56-3:15 đang chờ xác nhận. Số podcast là mức độ khớp với reference đã nêu, không phải kết quả trên transcript chuẩn thủ công.

`legacy/evaluate_finetuned_video.py` là thử nghiệm cũ dùng Kyle fine-tune và decoder đã sửa; không thay thế phép so sánh hai model gốc.
