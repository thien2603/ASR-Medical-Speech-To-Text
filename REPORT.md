# Báo cáo đánh giá nhận dạng giọng nói y khoa tiếng Việt

## 1. Mục tiêu và cách đánh giá

So sánh ba model ASR gốc và kiểm tra kết quả bản Kyle đã fine-tune của dự án.

Tập đánh giá là **VietMed test**, gồm **3.437 mẫu**, tương đương **6,02 giờ audio**.
Ba model gốc dùng cùng audio, transcript chuẩn và cách chấm điểm. Các lần chạy
dùng cùng loại GPU Tesla T4, FP16, batch size 4, trong các phiên Kaggle khác nhau.

- **WER:** tỷ lệ lỗi theo đơn vị tách bằng khoảng trắng.
- **CER:** tỷ lệ lỗi ký tự.

Hai chỉ số này **càng thấp càng tốt**. Khi chấm điểm, giữ nguyên chữ số và dấu
tiếng Việt. Thời gian dưới đây là thời gian nhận dạng toàn bộ tập, không gồm
tải model và chạy khởi động.

## 2. Kết quả ba model gốc

| Model | WER | CER | Thời gian xử lý (giây) |
|---|---:|---:|---:|
| `kyle/vi-asr-fastconformer-114m` | 26,19% | 20,44% | **78,67** |
| `vinai/PhoWhisper-large` | **24,14%** | **20,24%** | 1.818,52 |
| `openai/whisper-large-v3` | 74,30% | 58,83% | 1.548,02 |

**PhoWhisper-large có tỷ lệ lỗi thấp nhất; Kyle xử lý nhanh nhất trong lần đo này.**
Thời gian còn phụ thuộc thư viện suy luận: Kyle dùng NeMo, hai Whisper dùng Transformers.

Whisper-large-v3 thường sinh câu lạc nội dung, chẳng hạn câu kêu gọi đăng ký
kênh. Kiểm tra chéo ba mẫu đầu bằng pipeline chính thức vẫn cho cùng đầu ra.
Kết quả 74,30% phản ánh cấu hình và tập dữ liệu này, không phải kết luận cho mọi audio.

## 3. Kết quả bản fine-tune của dự án

| Phiên bản | WER | CER |
|---|---:|---:|
| Kyle fine-tune + sửa decoding thích ứng | 26,91% | 21,86% |

Bản fine-tune từng gặp lỗi lặp từ. Dự án đã giới hạn số token phát ở mỗi bước
giải mã xuống 1 và giải mã lại **869 mẫu bị đánh dấu bất thường**, giữ transcript
cũ của các mẫu còn lại. WER giảm từ **177,47% xuống 26,91%**.

So với Kyle gốc, WER của bản fine-tune sau sửa vẫn cao hơn **0,72 điểm phần trăm**.
**Thử nghiệm fine-tune hiện chưa cho thấy cải thiện.**

Kết quả này được trình bày riêng vì cấu hình sửa decoding đã được chọn dựa trên
một phần tập test. Đây là kết quả của pipeline hai lượt, không phải một lần chạy
toàn bộ tập bằng một cấu hình decoding duy nhất.

## 4. Kết luận

- **Ưu tiên độ chính xác trên benchmark này:** PhoWhisper-large.
- **Ưu tiên tốc độ trong lần đo này:** Kyle gốc.
- **Bản fine-tune hiện tại:** chưa tốt hơn Kyle gốc; cần đánh giá lại với cấu hình
  chọn trên dev và một tập test chưa dùng để điều chỉnh.

Báo cáo đo chất lượng phiên âm trên VietMed; chưa chấm riêng độ đúng của tên
thuốc, liều lượng và phủ định. Các số WER/CER đã được tính lại từ transcript.

Nguồn số liệu: [bảng chính](results/medical-main/comparison.md),
[kết quả fine-tune](local_decoding_fix/finetuned_adaptive_summary_local.json),
[kiểm chứng benchmark](results/medical-main/verification.json).
