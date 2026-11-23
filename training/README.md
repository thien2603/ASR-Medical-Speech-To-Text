# Huấn luyện

- `vietmed/train.ipynb`: notebook chính trên Kaggle.
- `vietmed/train.py`: script tương đương, đã bỏ IPython `%pip` và có `main()`.
- `vietmed/kernel-metadata.json`: cấu hình publish notebook.

Train/dev được dùng để tối ưu, chọn checkpoint và phân tích lỗi trên dev. Tập test không được chấm tự động trong recipe này. Chuẩn bị manifest test để kiểm tra trùng lặp dữ liệu không đồng nghĩa dùng test chọn model.

Từ gốc dự án:

```powershell
python train.py --help
kaggle kernels push -p training/vietmed --accelerator NvidiaTeslaT4
```

Sau khi có file `.nemo`, chạy đánh giá riêng:

```powershell
python test_datasets.py --checkpoint path/to/model.nemo --datasets vietmed --samples 3437 --output-dir results/finetuned-test
```

Các notebook publish cũ nguyên bản được giữ trong `archive/`; không phải entry point hiện tại. Artifact của lần fine-tune lịch sử nằm ở `kaggle_v5_report/` và `kaggle_v5_output/`.
