# WER/CER trên podcast của dự án

Audio: 354.267 giây, 17 đoạn giống nhau cho cả hai model.

| Model gốc | WER | CER | Thời gian nhận dạng |
|---|---:|---:|---:|
| kyle/vi-asr-fastconformer-114m | 18.09% | 12.34% | 1.27 giây |
| vinai/PhoWhisper-large | 10.26% | 7.08% | 22.96 giây |

Reference được sửa từ phụ đề ban đầu với bằng chứng từ Whisper-large-v2 độc lập, chưa được con người nghe duyệt toàn bộ. Đây là số đo theo bản tham chiếu đã nêu, chưa phải benchmark với transcript chuẩn thủ công.

Không sử dụng đầu ra Kyle/PhoWhisper để sửa reference. Giữ chữ số và dấu tiếng Việt khi chấm. Mô hình nhận dạng cùng waveform/chunk; WER/CER được tính lại cục bộ từ transcript. Các từ, số tuổi hoặc chỗ còn nghi vấn cần xem subtitle_correction_audit.json trước khi dùng số liệu như kết quả chính thức.
