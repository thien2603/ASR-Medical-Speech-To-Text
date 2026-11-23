# Đánh giá ASR

| Thư mục | Công việc |
|---|---|
| `datasets/` | Chấm VIVOS, FLEURS, VietMed; model gốc hoặc `.nemo` fine-tune |
| `audio_video/` | Quy trình riêng cho file podcast 354,267 giây |
| `decoding/` | Thí nghiệm chẩn đoán/sửa lỗi decoder |

Đánh giá không tối ưu trọng số. Các runner Kaggle là payload độc lập, nên phần chuẩn hóa tối thiểu được giữ trong từng runner; mã cục bộ dùng `asr_common/text.py` để cùng quy tắc chấm.
