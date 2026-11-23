# So sánh chính: ASR trên VietMed test

Mỗi model: toàn bộ 3.437 mẫu / 6,022 giờ; GPU Tesla T4, FP16, batch size 4.

| Model gốc | WER (%) ↓ | CER (%) ↓ | RTF ↓ | Thời gian suy luận (s) | Peak GPU (GiB) |
|---|---:|---:|---:|---:|---:|
| kyle/vi-asr-fastconformer-114m | 26.19 | 20.44 | 0.0036 | 78.67 | 0.68 |
| vinai/PhoWhisper-large | **24.14** | 20.24 | 0.0839 | 1818.52 | 3.98 |
| openai/whisper-large-v3 | 74.30 | 58.83 | 0.0714 | 1548.02 | 4.10 |

WER/CER tính trên toàn corpus theo cùng chuẩn hóa; giữ chữ số và dấu tiếng Việt. RTF gồm đọc WAV, tiền xử lý và decoding, loại trừ load model và warm-up. Kyle dùng NeMo, hai Whisper dùng Transformers. Các lần chạy được thực hiện trong những phiên Kaggle khác nhau sau khi sửa lỗi môi trường, cùng loại GPU và cùng precision/batch size.

Chỉ so sánh chất lượng phiên âm trên benchmark này; chưa đo riêng recall tên thuốc, liều lượng hoặc phủ định. Kết quả phụ thuộc cấu hình giải mã đã ghi trong config.json. Bản Kyle fine-tune + adaptive decoding WER 26,91% là thí nghiệm bổ sung có chọn cấu hình dựa trên test, không trộn vào bảng chính.

Các metric đã được tính lại từ predictions.jsonl và đối chiếu ID/hash audio/reference. Xem verification.json, summary.json và transcript từng model để kiểm tra lại.

Whisper-large-v3 sinh nhiều câu lạc nội dung: transcript chuẩn hóa phổ biến nhất xuất hiện 1101 lần. Ba mẫu đầu đã được đối chiếu bằng direct eager và pipeline Transformers chính thức (batch=1), cho cùng đầu ra với lần SDPA batch=4. Đây là kiểm tra ba mẫu, không chứng minh nguyên nhân của mọi lỗi trên toàn corpus; xem whisper_implementation_diagnostic.json và whisper_repeated_outputs.json.
