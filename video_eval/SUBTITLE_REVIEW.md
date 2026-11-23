# Hiệu chỉnh phụ đề podcast

Đã nhận dạng xong Kyle gốc và PhoWhisper-large. Reference được sửa với hỗ trợ của Whisper-large-v2 và wav2vec2 tiếng Việt độc lập; chưa được con người nghe duyệt toàn bộ.

Bản gốc: reference_subtitle.original.txt. Bản sạch định dạng: reference_subtitle.cleaned.txt. Bản hiệu chỉnh: reference_subtitle.corrected.txt; đồng thời cập nhật reference_subtitle.txt.

## Các đề xuất sửa đã áp dụng

| Trước | Sau | Khoảng audio |
|---|---|---|
| À, trên thực tế là cái khi | Thì thật ra là cái khi | 23.05-41.50 giây |
| chỉ số ung thư trên thực tế là | chỉ số ung thư thật ra là | 63.45-87.10 giây |
| nhưng mà thực ra nó cũng | nhưng mà thật ra nó cũng | 87.10-109.70 giây |
| mà thực ra nó có bệnh | mà thật ra nó có bệnh | 87.10-109.70 giây |
| tỷ lệ mà nó trẻ quá | tỷ lệ mà nó trẻ hóa | 175.95-194.95 giây, 255.15-276.90 giây |
| Nhưng mà riêng đó lẽ là người bình thường | Nhưng mà riêng đó là người bình thường | 213.95-232.40 giây, 232.40-255.15 giây |

Các sửa đổi trên là đề xuất biên tập có hỗ trợ ASR, chưa được nghe xác minh trực tiếp. Không dùng transcript của Kyle/PhoWhisper để sửa reference.

## Cần nghe xác nhận

- 2:56-3:15: cụm 'trong y gia' và '20 năm thì chỉ là trước 40 tuổi'. Giữ lời cũ tới khi người nghe xác nhận, đặc biệt không tự đổi con số hoặc chiều trước/trên tuổi.
- 1:50-2:12: cụm 'phát hiện chuyển trên cái niêm mạc'. Nguồn độc lập chưa thống nhất; giữ lời cũ.

[Nghe đoạn số tuổi](review_clips/age_02m56_03m15.mp3).

## Số liệu tạm tính

| Model | WER | CER |
|---|---:|---:|
| kyle/vi-asr-fastconformer-114m | 18.09% | 12.34% |
| vinai/PhoWhisper-large | 10.26% | 7.08% |

Các metric đã tính lại từ transcript nhưng chỉ phản ánh mức độ khớp bản reference đã nêu. Cần nghe duyệt reference trước khi coi đây là benchmark với đáp án chuẩn thủ công.
