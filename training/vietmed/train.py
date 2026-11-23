"""Standalone equivalent of the adjacent notebook; execution requires --run."""
import argparse


def run_training():
    from IPython.display import display
    # ---
    # jupyter:
    #   jupytext:
    #     formats: ipynb,py:percent
    #     text_representation:
    #       extension: .py
    #       format_name: percent
    #       format_version: '1.3'
    #       jupytext_version: 1.17.2
    #   kernelspec:
    #     display_name: Python 3
    #     language: python
    #     name: python3
    # ---

    # %% [markdown]
    # # Continued fine-tuning FastConformer-RNNT tiếng Việt trên VietMed
    #
    # Notebook này tạo một baseline supervised sạch cho model gated
    # `kyle/vi-asr-fastconformer-114m` trên Kaggle T4 16 GB. Pipeline gồm:
    #
    # 1. tự tìm metadata/audio VietMed trong `/kaggle/input`;
    # 2. chuyển toàn bộ audio sang WAV mono 16 kHz và tạo NeMo manifest;
    # 3. kiểm tra trùng lặp giữa train/dev/test;
    # 4. đo baseline trên dev;
    # 5. full fine-tuning FastConformer-RNNT (mặc định F1);
    # 6. chọn checkpoint theo **dev WER**, xuất checkpoint; test chạy bằng lệnh riêng;
    # 7. xuất model `.nemo`, dự đoán, metric, config và log.
    #
    # **Lưu ý trước khi chạy**
    #
    # - Kaggle → Settings → Accelerator → chọn **GPU T4**.
    # - Model là gated. Hãy chấp nhận điều kiện trên Hugging Face, sau đó thêm Kaggle
    #   Secret tên `HF_TOKEN` hoặc `Token_HF`. Không dán token vào notebook.
    # - Attach VietMed làm Kaggle Input. Nếu auto-discovery không xác định đúng metadata,
    #   điền bốn đường dẫn trong cell Configuration.
    # - Không dùng test để chọn hyperparameter, checkpoint, tokenizer hay danh sách thuật ngữ.

    # %% [markdown]
    # ## 1. Configuration

    # %%
    from pathlib import Path

    MODEL_ID = "kyle/vi-asr-fastconformer-114m"
    SEED = 42

    # Để None để tự tìm dưới /kaggle/input; có thể ghi đè bằng đường dẫn cụ thể.
    DATA_ROOT = None
    TRAIN_METADATA = None
    DEV_METADATA = None
    TEST_METADATA = None

    WORKING_DIR = Path("/kaggle/working")
    PREPARED_DIR = WORKING_DIR / "vietmed_prepared"
    EXPERIMENT_ROOT = WORKING_DIR / "experiments" / "fastconformer_vietmed"

    EXPERIMENT_NAME = "F1"  # mặc định chỉ chạy F1 để tiết kiệm GPU quota
    EXPERIMENTS = {
        "F1": {"lr": 1e-5, "use_spec_augment": True},
        "F2": {"lr": 3e-6, "use_spec_augment": True},
        "F3": {"lr": 3e-5, "use_spec_augment": True},
        "F4": {"lr": 1e-5, "use_spec_augment": False},
    }
    USE_SPEC_AUGMENT = EXPERIMENTS[EXPERIMENT_NAME]["use_spec_augment"]

    TRAIN_BATCH_SIZE = 2
    VAL_BATCH_SIZE = 2
    ACCUMULATE_GRAD_BATCHES = 8  # effective batch xấp xỉ 16
    MAX_DURATION = 20.0
    MIN_DURATION = 0.3
    MAX_EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 5
    NUM_WORKERS = 2

    N_BASELINE_SAMPLES = 500  # None = toàn bộ dev
    N_DEV_EVAL_SAMPLES = 500  # giữ bằng baseline để so sánh công bằng
    MEDICAL_TERMS_PATH = None  # ví dụ: /kaggle/input/my-terms/medical_terms.txt
    RESUME_IF_AVAILABLE = True

    # None = full fine-tuning. Chỉ đổi thành "decoder_joint" hoặc "last_encoder_blocks"
    # nếu full fine-tuning vẫn OOM sau khi đã giảm batch/max_duration.
    PARTIAL_TUNING = None
    LAST_N_ENCODER_BLOCKS = 4

    assert EXPERIMENT_NAME in EXPERIMENTS
    assert N_BASELINE_SAMPLES == N_DEV_EVAL_SAMPLES, (
        "Để so sánh công bằng, baseline và fine-tuned phải dùng cùng số mẫu dev."
    )

    RUN_DIR = EXPERIMENT_ROOT / EXPERIMENT_NAME
    CHECKPOINT_DIR = RUN_DIR / "checkpoints"
    for directory in [PREPARED_DIR, RUN_DIR, CHECKPOINT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    print("Experiment:", EXPERIMENT_NAME, EXPERIMENTS[EXPERIMENT_NAME])
    print("Run directory:", RUN_DIR)

    # %% [markdown]
    # ## 2. GPU / environment check

    # %%
    import os
    import platform
    import subprocess
    import sys

    print("Python:", sys.version)
    print("Platform:", platform.platform())
    subprocess.run(["nvidia-smi"], check=False)

    # %% [markdown]
    # ## 3. Install dependencies
    #
    # NeMo 3.0 hiện là bản mới nhất, nhưng notebook pin `nemo_toolkit[asr]==2.7.3`:
    # đây là nhánh NeMo 2.x tương thích với checkpoint và API fine-tuning đã kiểm chứng
    # (`ASRModel.from_pretrained`, `setup_*_data`, `setup_optimization`). Kaggle hiện dùng
    # Python 3.13; giữ nguyên scientific stack gốc đã kiểm thử để tránh thay NumPy binary
    # giữa phiên. Không cài lại PyTorch để giữ CUDA build mà Kaggle cung cấp.

    # %%
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '-q', '--upgrade-strategy', 'only-if-needed', 'numpy==2.1.3', 'scipy==1.16.3', 'torchmetrics==1.9.0', 'pandas==2.3.3', 'protobuf==5.29.5', 'librosa==0.11.0', 'soundfile>=0.12.1', 'nemo_toolkit[asr]==2.7.3', 'jiwer==3.1.0'], check=True)

    # %% [markdown]
    # Nếu cell import kế tiếp báo lỗi do Kaggle đã import sẵn package cũ, dùng
    # **Run → Restart Session**, sau đó chạy lại từ đầu. Token và dataset không bị ghi vào code.

    # %% [markdown]
    # ## 4. Imports, phiên bản và seed

    # %%
    import gc
    import hashlib
    import io
    import json
    import math
    import random
    import re
    import shutil
    import time
    import unicodedata
    import warnings
    from collections import Counter
    from importlib.metadata import version

    import jiwer
    import librosa
    import lightning.pytorch as pl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import soundfile as sf
    import torch
    import yaml
    from huggingface_hub import login
    from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from nemo.collections.asr.models import ASRModel
    from omegaconf import OmegaConf
    from tqdm.auto import tqdm

    try:
        import nemo.collections.asr as nemo_asr
    except Exception as exc:
        raise RuntimeError(
            "Không import được nemo.collections.asr. Hãy Restart Session sau cell cài đặt "
            "và chạy lại notebook. Lỗi gốc: " + repr(exc)
        ) from exc

    print("PyTorch:", torch.__version__)
    print("Torch CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("NeMo:", version("nemo_toolkit"))
    print("Lightning:", pl.__version__)
    if not torch.cuda.is_available():
        raise RuntimeError("Không thấy GPU. Trong Kaggle Settings, hãy chọn Accelerator = GPU T4.")
    print("GPU:", torch.cuda.get_device_name(0))

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    pl.seed_everything(SEED, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    def get_hf_token():
        """Đọc token mà không in token ra log."""
        token = os.environ.get("HF_TOKEN") or os.environ.get("TOKEN_HF")
        if not token:
            try:
                from kaggle_secrets import UserSecretsClient
                secrets_client = UserSecretsClient()
                for secret_name in ("HF_TOKEN", "Token_HF", "TOKEN_HF"):
                    try:
                        token = secrets_client.get_secret(secret_name)
                    except Exception:
                        token = None
                    if token:
                        break
            except Exception:
                token = None
        if not token:
            raise RuntimeError(
                "Thiếu Hugging Face token. Hãy chấp nhận điều kiện của model, tạo "
                "Kaggle Secret tên HF_TOKEN hoặc Token_HF, rồi chạy lại cell này."
            )
        login(token=token, add_to_git_credential=False)
        return "authenticated"


    print("Hugging Face:", get_hf_token())

    # Ghi lại môi trường, không ghi secrets.
    environment_lines = [
        f"python={sys.version}",
        f"platform={platform.platform()}",
        f"torch={torch.__version__}",
        f"torch_cuda={torch.version.cuda}",
        f"gpu={torch.cuda.get_device_name(0)}",
        f"nemo_toolkit={version('nemo_toolkit')}",
        f"lightning={pl.__version__}",
        f"model_id={MODEL_ID}",
        f"seed={SEED}",
        "",
        "# pip freeze",
        subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout,
    ]
    (WORKING_DIR / "environment.txt").write_text("\n".join(environment_lines), encoding="utf-8")

    # %% [markdown]
    # ## 5. Discover VietMed files
    #
    # Auto-discovery chỉ chọn split theo tên file/thư mục (`train`, `dev`/`valid`, `test`).
    # Nó **không random split**. Nếu dữ liệu attach có layout lạ hoặc nhiều bản VietMed,
    # cell sẽ dừng để bạn điền đường dẫn ở Configuration thay vì đoán sai.

    # %%
    AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}
    METADATA_EXTENSIONS = {".csv", ".json", ".jsonl", ".parquet"}


    def choose_data_root(configured_root=None):
        if configured_root:
            root = Path(configured_root)
            if not root.exists():
                raise FileNotFoundError(root)
            return root

        input_root = Path("/kaggle/input")
        if not input_root.exists():
            raise FileNotFoundError("Không thấy /kaggle/input. Hãy attach VietMed vào notebook.")
        folders = sorted(p for p in input_root.iterdir() if p.is_dir())
        print("Kaggle input folders:")
        for folder in folders:
            print(" -", folder)
        candidates = [p for p in folders if "vietmed" in p.name.lower()]
        if len(candidates) == 1:
            return candidates[0]
        if len(folders) == 1:
            warnings.warn("Không thấy tên 'vietmed'; dùng input folder duy nhất.")
            return folders[0]
        raise RuntimeError(
            f"Không xác định duy nhất DATA_ROOT. Candidates={candidates}. "
            "Hãy điền DATA_ROOT trong Configuration."
        )


    def discover_files(root):
        root = Path(root)
        audio_files = sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)
        metadata_files = sorted(p for p in root.rglob("*") if p.suffix.lower() in METADATA_EXTENSIONS)
        return audio_files, metadata_files


    def infer_split_metadata(metadata_files, configured=None):
        configured = configured or {}
        result = {}
        aliases = {
            "train": ("train",),
            "dev": ("dev", "valid", "validation", "val"),
            "test": ("test",),
        }
        for split, tokens in aliases.items():
            if configured.get(split):
                result[split] = Path(configured[split])
                continue
            matches = []
            for path in metadata_files:
                lowered_parts = [part.lower() for part in path.parts]
                stem_tokens = set(re.split(r"[^a-z0-9]+", path.stem.lower()))
                if any(token in stem_tokens or token in lowered_parts for token in tokens):
                    matches.append(path)
            # Tránh ưu tiên các manifest do notebook tạo ở /kaggle/working (root chỉ ở input).
            if len(matches) == 1:
                result[split] = matches[0]
            elif len(matches) == 0:
                result[split] = None
            else:
                print(f"Nhiều metadata candidates cho {split}:")
                for match in matches:
                    print(" -", match)
                raise RuntimeError(
                    f"Không thể chọn metadata {split}. Hãy điền {split.upper()}_METADATA."
                )
        if result["train"] is None or result["dev"] is None:
            raise RuntimeError(
                "Cần ít nhất train và dev metadata chính thức; không tự random split. "
                "Hãy điền TRAIN_METADATA và DEV_METADATA."
            )
        return result


    DATA_ROOT = choose_data_root(DATA_ROOT)
    audio_files, metadata_files = discover_files(DATA_ROOT)
    split_metadata = infer_split_metadata(
        metadata_files,
        {"train": TRAIN_METADATA, "dev": DEV_METADATA, "test": TEST_METADATA},
    )

    print("DATA_ROOT:", DATA_ROOT)
    print(f"Audio files discovered: {len(audio_files):,}")
    print(f"Metadata files discovered: {len(metadata_files):,}")
    for path in metadata_files[:30]:
        print(" metadata:", path)
    print("Official split mapping:", {k: str(v) if v else None for k, v in split_metadata.items()})

    # %% [markdown]
    # ## 6. Build NeMo manifests
    #
    # Mỗi mẫu được decode, downmix mono, resample 16 kHz và ghi PCM16 WAV vào
    # `/kaggle/working/vietmed_prepared/<split>`. `raw_text` được giữ trong CSV audit;
    # NeMo manifest dùng `normalized_text`.

    # %%
    TEXT_COLUMN_CANDIDATES = ["text", "transcription", "transcript", "sentence", "normalized_text"]
    AUDIO_COLUMN_CANDIDATES = [
        "audio_filepath", "audio_path", "path", "file", "filename", "audio", "speech"
    ]


    def normalize_vi_text(text):
        """Chuẩn hóa bảo thủ: giữ chữ số và toàn bộ dấu tiếng Việt."""
        text = unicodedata.normalize("NFC", str(text)).lower().strip()
        # Dấu câu/ký hiệu thành khoảng trắng; không xóa Unicode letters/numbers/marks.
        text = "".join(
            " " if unicodedata.category(char)[0] in {"P", "S"} else char
            for char in text
        )
        return re.sub(r"\s+", " ", text).strip()


    def read_metadata(path):
        path = Path(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".jsonl":
            return pd.read_json(path, lines=True)
        if path.suffix.lower() == ".json":
            try:
                return pd.read_json(path)
            except ValueError:
                return pd.read_json(path, lines=True)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        raise ValueError(f"Metadata extension chưa hỗ trợ: {path}")


    def find_column(frame, candidates, kind):
        lower_map = {str(column).lower(): column for column in frame.columns}
        for candidate in candidates:
            if candidate in lower_map:
                return lower_map[candidate]
        raise KeyError(f"Không tìm thấy cột {kind}. Columns={list(frame.columns)}")


    def build_audio_index(paths):
        by_name, by_stem = {}, {}
        for path in paths:
            by_name.setdefault(path.name.lower(), []).append(path)
            by_stem.setdefault(path.stem.lower(), []).append(path)
        return by_name, by_stem


    AUDIO_BY_NAME, AUDIO_BY_STEM = build_audio_index(audio_files)


    def resolve_audio_value(value, metadata_path):
        """Trả về Path hoặc bytes cho cả metadata thường và Arrow audio struct."""
        if isinstance(value, dict):
            embedded = value.get("bytes")
            if embedded is not None:
                return bytes(embedded)
            value = value.get("path") or value.get("audio_filepath")
        if value is None or (isinstance(value, float) and math.isnan(value)):
            raise ValueError("audio value rỗng")
        candidate = Path(str(value))
        attempts = [candidate]
        if not candidate.is_absolute():
            attempts.extend([metadata_path.parent / candidate, DATA_ROOT / candidate])
        for attempt in attempts:
            if attempt.exists():
                return attempt.resolve()
        name_matches = AUDIO_BY_NAME.get(candidate.name.lower(), [])
        if len(name_matches) == 1:
            return name_matches[0].resolve()
        stem_matches = AUDIO_BY_STEM.get(candidate.stem.lower(), [])
        if len(stem_matches) == 1:
            return stem_matches[0].resolve()
        raise FileNotFoundError(
            f"Không resolve được audio '{value}' (name matches={len(name_matches)}, "
            f"stem matches={len(stem_matches)})"
        )


    def decode_audio(source):
        handle = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else str(source)
        waveform, sample_rate = sf.read(handle, always_2d=True, dtype="float32")
        waveform = waveform.mean(axis=1)  # downmix channel -> mono
        if sample_rate != 16000:
            waveform = librosa.resample(
                waveform, orig_sr=sample_rate, target_sr=16000, res_type="soxr_hq"
            )
        if waveform.size == 0 or not np.isfinite(waveform).all():
            raise ValueError("audio rỗng hoặc có NaN/Inf")
        return np.clip(waveform, -1.0, 1.0), int(sample_rate)


    def build_manifest(split, metadata_path):
        metadata_path = Path(metadata_path)
        frame = read_metadata(metadata_path)
        text_column = find_column(frame, TEXT_COLUMN_CANDIDATES, "transcript")
        audio_column = find_column(frame, AUDIO_COLUMN_CANDIDATES, "audio")
        output_audio_dir = PREPARED_DIR / split
        output_audio_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = PREPARED_DIR / f"{split}_manifest.jsonl"
        audit_path = PREPARED_DIR / f"{split}_prepared.csv"
        error_path = PREPARED_DIR / f"{split}_errors.csv"

        records, audit_rows, errors = [], [], []
        for row_number, (_, row) in enumerate(tqdm(frame.iterrows(), total=len(frame), desc=split)):
            raw_text = "" if pd.isna(row[text_column]) else str(row[text_column])
            normalized_text = normalize_vi_text(raw_text)
            try:
                if not normalized_text:
                    raise ValueError("transcript rỗng sau normalization")
                source = resolve_audio_value(row[audio_column], metadata_path)
                waveform, source_sr = decode_audio(source)
                duration = float(len(waveform) / 16000)
                if duration <= 0:
                    raise ValueError("duration <= 0")
                output_path = (output_audio_dir / f"{row_number:08d}.wav").resolve()
                sf.write(output_path, waveform, 16000, subtype="PCM_16")
                if not output_path.exists():
                    raise FileNotFoundError(output_path)
                record = {
                    "audio_filepath": str(output_path),
                    "duration": duration,
                    "text": normalized_text,
                }
                records.append(record)
                audit_rows.append({
                    **record,
                    "raw_text": raw_text,
                    "normalized_text": normalized_text,
                    "source_sample_rate": source_sr,
                    "source_audio": "<embedded-bytes>" if isinstance(source, bytes) else str(source),
                })
            except Exception as exc:
                errors.append({
                    "row": row_number,
                    "audio_value": repr(row.get(audio_column)),
                    "error": repr(exc),
                })

        if not records:
            raise RuntimeError(f"Không tạo được sample hợp lệ cho split {split}.")
        with manifest_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        pd.DataFrame(audit_rows).to_csv(audit_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(errors, columns=["row", "audio_value", "error"]).to_csv(
            error_path, index=False, encoding="utf-8-sig"
        )
        print(
            f"{split}: {len(records):,}/{len(frame):,} samples; "
            f"{len(errors):,} lỗi -> {manifest_path}"
        )
        return manifest_path, audit_path, error_path


    manifests = {}
    for split, metadata_path in split_metadata.items():
        if metadata_path is not None:
            manifests[split] = build_manifest(split, metadata_path)[0]

    assert "train" in manifests and "dev" in manifests
    print({key: str(value) for key, value in manifests.items()})

    # %% [markdown]
    # ## 7. Dataset statistics và leakage checks

    # %%
    def read_manifest(path):
        with Path(path).open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


    def manifest_statistics(path):
        rows = read_manifest(path)
        durations = np.asarray([row["duration"] for row in rows], dtype=float)
        text_chars = np.asarray([len(row["text"]) for row in rows], dtype=int)
        text_words = np.asarray([len(row["text"].split()) for row in rows], dtype=int)
        return {
            "samples": int(len(rows)),
            "hours": float(durations.sum() / 3600),
            "duration_min": float(durations.min()),
            "duration_median": float(np.median(durations)),
            "duration_p90": float(np.quantile(durations, 0.90)),
            "duration_max": float(durations.max()),
            "transcript_chars_median": float(np.median(text_chars)),
            "transcript_chars_p90": float(np.quantile(text_chars, 0.90)),
            "transcript_words_median": float(np.median(text_words)),
            "over_max_duration": int((durations > MAX_DURATION).sum()),
        }


    dataset_stats = {split: manifest_statistics(path) for split, path in manifests.items()}
    stats_frame = pd.DataFrame(dataset_stats).T
    display(stats_frame)
    (WORKING_DIR / "dataset_statistics.json").write_text(
        json.dumps(dataset_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"Train p90={dataset_stats['train']['duration_p90']:.2f}s; "
        f"{dataset_stats['train']['over_max_duration']:,} file > MAX_DURATION={MAX_DURATION}s."
    )
    if dataset_stats["train"]["duration_p90"] > MAX_DURATION:
        warnings.warn(
            "p90 dài hơn MAX_DURATION. Các utterance dài hơn sẽ bị NeMo lọc; "
            "hãy cân nhắc segment dữ liệu thay vì tăng max_duration trên T4."
        )


    def sha256_file(path, chunk_size=1024 * 1024):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
        return digest.hexdigest()


    split_paths = {
        split: {Path(row["audio_filepath"]).resolve() for row in read_manifest(path)}
        for split, path in manifests.items()
    }
    for left, right in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        if left in split_paths and right in split_paths:
            overlap = split_paths[left] & split_paths[right]
            if overlap:
                raise RuntimeError(f"Leakage: {len(overlap)} audio paths trùng giữa {left}/{right}.")

    print("Hashing prepared audio để kiểm tra nội dung trùng giữa splits...")
    split_hashes = {
        split: {sha256_file(path): str(path) for path in tqdm(paths, desc=f"hash {split}")}
        for split, paths in split_paths.items()
    }
    for left, right in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        if left in split_hashes and right in split_hashes:
            duplicate_hashes = set(split_hashes[left]) & set(split_hashes[right])
            if duplicate_hashes:
                examples = [
                    (split_hashes[left][digest], split_hashes[right][digest])
                    for digest in list(duplicate_hashes)[:5]
                ]
                raise RuntimeError(
                    f"Leakage: {len(duplicate_hashes)} audio trùng nội dung giữa {left}/{right}. "
                    f"Examples={examples}"
                )
    print("Leakage check: PASS (path và SHA-256 không trùng giữa các split).")

    # %% [markdown]
    # ## 8. Load pretrained FastConformer-RNNT

    # %%
    torch.cuda.empty_cache()
    model = ASRModel.from_pretrained(model_name=MODEL_ID, map_location="cpu")

    model_class = type(model).__name__
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    decoder_target = str(OmegaConf.select(model.cfg, "decoder._target_", default=""))
    is_transducer = (
        "rnnt" in model_class.lower()
        or "transducer" in model_class.lower()
        or "rnnt" in decoder_target.lower()
        or hasattr(model, "joint")
    )

    print("Class:", model_class)
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print("Decoder config:\n", OmegaConf.to_yaml(model.cfg.decoder))
    print("RNNT/Transducer:", is_transducer)
    if not is_transducer:
        raise TypeError("Checkpoint không được nhận diện là RNNT/Transducer; dừng để tránh train sai.")

    required_config_keys = ["train_ds", "validation_ds", "optim", "decoder", "loss"]
    missing_keys = [key for key in required_config_keys if OmegaConf.select(model.cfg, key) is None]
    if missing_keys:
        raise KeyError(f"Model config thiếu keys cần thiết: {missing_keys}")
    print("Verified config keys:", required_config_keys)

    PRETRAINED_NEMO = WORKING_DIR / "vi-asr-fastconformer-114m-pretrained.nemo"
    model.save_to(str(PRETRAINED_NEMO))
    print("Saved original checkpoint:", PRETRAINED_NEMO)

    # %% [markdown]
    # ## 9. Baseline evaluation trên dev
    #
    # WER/CER chính là corpus-level `jiwer.wer`/`jiwer.cer`. `sample_wer` chỉ dùng
    # để xếp hạng error analysis, không dùng làm metric tổng.

    # %%
    def hypothesis_to_text(item):
        if isinstance(item, str):
            return item
        if hasattr(item, "text"):
            return item.text
        return str(item)


    def safe_sample_wer(reference, hypothesis):
        if not reference:
            return 0.0 if not hypothesis else 1.0
        return float(jiwer.wer(reference, hypothesis))


    @torch.inference_mode()
    def evaluate_model(asr_model, manifest_path, prediction_path, summary_path, limit=None, batch_size=2):
        rows = read_manifest(manifest_path)
        if limit is not None:
            rows = rows[: min(int(limit), len(rows))]  # cùng prefix deterministic trước/sau
        audio_paths = [row["audio_filepath"] for row in rows]
        raw_references = [row["text"] for row in rows]
        references = [normalize_vi_text(text) for text in raw_references]
        total_audio_seconds = float(sum(row["duration"] for row in rows))

        asr_model.eval()
        asr_model = asr_model.cuda()
        torch.cuda.synchronize()
        started = time.perf_counter()
        predictions = asr_model.transcribe(
            audio_paths, batch_size=batch_size, verbose=True, return_hypotheses=False
        )
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - started
        hypotheses = [normalize_vi_text(hypothesis_to_text(item)) for item in predictions]
        sample_wers = [safe_sample_wer(ref, hyp) for ref, hyp in zip(references, hypotheses)]

        prediction_frame = pd.DataFrame({
            "audio_filepath": audio_paths,
            "duration": [row["duration"] for row in rows],
            "raw_reference": raw_references,
            "reference": references,
            "prediction": hypotheses,
            "sample_wer": sample_wers,
        })
        prediction_frame.to_csv(prediction_path, index=False, encoding="utf-8-sig")
        summary = {
            "samples": len(rows),
            "wer": float(jiwer.wer(references, hypotheses)),
            "cer": float(jiwer.cer(references, hypotheses)),
            "exact_match": float(np.mean(np.asarray(references) == np.asarray(hypotheses))),
            "median_sample_wer": float(np.median(sample_wers)),
            "p90_sample_wer": float(np.quantile(sample_wers, 0.90)),
            "total_audio_seconds": total_audio_seconds,
            "inference_seconds": inference_seconds,
            "rtf": float(inference_seconds / total_audio_seconds) if total_audio_seconds else None,
            "normalization": "NFC + lowercase + punctuation/symbols to space + whitespace collapse",
        }
        Path(summary_path).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary, prediction_frame


    BASELINE_PREDICTIONS = WORKING_DIR / "baseline_predictions.csv"
    BASELINE_SUMMARY = WORKING_DIR / "baseline_summary.json"
    baseline_summary, baseline_predictions = evaluate_model(
        model,
        manifests["dev"],
        BASELINE_PREDICTIONS,
        BASELINE_SUMMARY,
        limit=N_BASELINE_SAMPLES,
        batch_size=VAL_BATCH_SIZE,
    )
    display(pd.DataFrame([baseline_summary]))

    # Giải phóng activation/cache trước khi cấu hình train.
    model = model.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    # %% [markdown]
    # ## 10. Configure fine-tuning
    #
    # Mặc định là **full fine-tuning**: encoder + RNNT predictor/decoder + joint đều trainable;
    # tokenizer và RNNT loss gốc được giữ nguyên. Mild SpecAugment sử dụng đúng config class
    # có trong checkpoint.

    # %%
    def print_vram(prefix="VRAM"):
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"{prefix}: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB, peak={peak:.2f} GB")
        return {"allocated_gb": allocated, "reserved_gb": reserved, "peak_gb": peak}


    def find_encoder_blocks(encoder):
        for name in ["layers", "conformer_layers", "encoder_layers"]:
            blocks = getattr(encoder, name, None)
            if blocks is not None and hasattr(blocks, "__len__"):
                return blocks, name
        raise AttributeError(
            "Không tìm thấy list encoder blocks ở layers/conformer_layers/encoder_layers. "
            "Hãy inspect model.encoder trước khi partial tuning."
        )


    def configure_trainable_parameters(asr_model, option=None, last_n=4):
        asr_model.unfreeze()
        if option is None:
            label = "FULL: encoder + predictor/decoder + joint"
        elif option == "decoder_joint":
            for parameter in asr_model.encoder.parameters():
                parameter.requires_grad = False
            label = "PARTIAL A: decoder/predictor + joint; encoder frozen"
        elif option == "last_encoder_blocks":
            for parameter in asr_model.encoder.parameters():
                parameter.requires_grad = False
            blocks, block_name = find_encoder_blocks(asr_model.encoder)
            if last_n < 1 or last_n > len(blocks):
                raise ValueError(f"last_n={last_n} không hợp lệ; encoder có {len(blocks)} blocks")
            for block in blocks[-last_n:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            label = f"PARTIAL B: last {last_n}/{len(blocks)} {block_name} + decoder + joint"
        else:
            raise ValueError(f"PARTIAL_TUNING không hợp lệ: {option}")
        total = sum(parameter.numel() for parameter in asr_model.parameters())
        trainable = sum(parameter.numel() for parameter in asr_model.parameters() if parameter.requires_grad)
        print(label)
        print(f"Trainable: {trainable:,}/{total:,} ({100 * trainable / total:.2f}%)")
        return trainable


    def enable_gradient_checkpointing_if_supported(asr_model):
        """Chỉ gọi thủ công sau khi inspect; không giả định API mà NeMo/model không có."""
        candidates = [asr_model, getattr(asr_model, "encoder", None)]
        for target in candidates:
            for method_name in ["gradient_checkpointing_enable", "enable_gradient_checkpointing"]:
                method = getattr(target, method_name, None)
                if callable(method):
                    method()
                    print(f"Enabled via {type(target).__name__}.{method_name}()")
                    return True
        print("Model/version này không expose API gradient checkpointing đã biết; không tự sửa internals.")
        return False


    class PeakVRAMCallback(Callback):
        def on_train_start(self, trainer, pl_module):
            torch.cuda.reset_peak_memory_stats()
            print_vram("Train start")

        def on_train_epoch_end(self, trainer, pl_module):
            print_vram(f"Epoch {trainer.current_epoch} end")


    train_cfg = OmegaConf.create({
        "manifest_filepath": str(manifests["train"]),
        "sample_rate": 16000,
        "batch_size": TRAIN_BATCH_SIZE,
        "shuffle": True,
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
        "max_duration": MAX_DURATION,
        "min_duration": MIN_DURATION,
        "is_tarred": False,
    })
    validation_cfg = OmegaConf.create({
        "manifest_filepath": str(manifests["dev"]),
        "sample_rate": 16000,
        "batch_size": VAL_BATCH_SIZE,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
        "max_duration": MAX_DURATION,
        "min_duration": MIN_DURATION,
        "is_tarred": False,
    })
    optim_cfg = OmegaConf.create({
        "name": "adamw",
        "lr": EXPERIMENTS[EXPERIMENT_NAME]["lr"],
        "weight_decay": 1e-3,
        "sched": {
            "name": "CosineAnnealing",
            "warmup_ratio": 0.05,
            "min_lr": 1e-6,
        },
    })

    # Config mild hơn checkpoint gốc: ít time masks hơn để hạn chế che âm tiết y khoa.
    mild_spec_augment_cfg = OmegaConf.create({
        "_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
        "freq_masks": 2,
        "time_masks": 4,
        "freq_width": 18,
        "time_width": 0.04,
    })

    training_config = {
        "model_id": MODEL_ID,
        "seed": SEED,
        "experiment_name": EXPERIMENT_NAME,
        "experiment": EXPERIMENTS[EXPERIMENT_NAME],
        "partial_tuning": PARTIAL_TUNING,
        "last_n_encoder_blocks": LAST_N_ENCODER_BLOCKS,
        "trainer": {
            "accelerator": "gpu",
            "devices": 1,
            "precision": "16-mixed",
            "max_epochs": MAX_EPOCHS,
            "accumulate_grad_batches": ACCUMULATE_GRAD_BATCHES,
            "gradient_clip_val": 1.0,
            "check_val_every_n_epoch": 1,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        },
        "train_ds": OmegaConf.to_container(train_cfg, resolve=True),
        "validation_ds": OmegaConf.to_container(validation_cfg, resolve=True),
        "optim": OmegaConf.to_container(optim_cfg, resolve=True),
        "spec_augment": OmegaConf.to_container(mild_spec_augment_cfg, resolve=True)
        if USE_SPEC_AUGMENT else None,
        "dataset_statistics": dataset_stats,
    }
    TRAINING_CONFIG_PATH = WORKING_DIR / "training_config.yaml"
    TRAINING_CONFIG_PATH.write_text(
        yaml.safe_dump(training_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(CHECKPOINT_DIR),
        filename="epoch={epoch:02d}-val_wer={val_wer:.4f}",
        monitor="val_wer",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    early_stopping = EarlyStopping(
        monitor="val_wer",
        mode="min",
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=0.0,
        verbose=True,
    )
    csv_logger = CSVLogger(save_dir=str(RUN_DIR), name="logs")
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",  # T4: FP16, không giả định BF16
        max_epochs=MAX_EPOCHS,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback, early_stopping, PeakVRAMCallback()],
        logger=csv_logger,
        log_every_n_steps=10,
        enable_checkpointing=True,
        deterministic=True,
    )

    model.set_trainer(trainer)
    model.setup_training_data(train_cfg)
    model.setup_multiple_validation_data(validation_cfg)
    configure_trainable_parameters(model, PARTIAL_TUNING, LAST_N_ENCODER_BLOCKS)
    model.setup_optimization(optim_cfg)
    if USE_SPEC_AUGMENT:
        model.spec_augment = ASRModel.from_config_dict(mild_spec_augment_cfg)
    else:
        model.spec_augment = None
    print_vram("After configuration")

    # %% [markdown]
    # ### Thứ tự xử lý nếu gặp CUDA OOM
    #
    # 1. đổi `TRAIN_BATCH_SIZE = 1`, giữ effective batch bằng cách tăng
    #    `ACCUMULATE_GRAD_BATCHES = 16`;
    # 2. giảm `MAX_DURATION` (ví dụ 20 → 16 → 12 giây); các câu dài nên segment từ trước;
    # 3. gọi `enable_gradient_checkpointing_if_supported(model)` **chỉ khi cell xác nhận hỗ trợ**;
    # 4. cuối cùng mới đặt `PARTIAL_TUNING="decoder_joint"`, hoặc
    #    `PARTIAL_TUNING="last_encoder_blocks"` với 2–4 block cuối.
    #
    # Notebook không tự chuyển fallback: kết quả experiment phải minh bạch.

    # %% [markdown]
    # ## 11. Train (checkpoint best + last, hỗ trợ resume trong cùng session)

    # %%
    last_checkpoint = CHECKPOINT_DIR / "last.ckpt"
    resume_checkpoint = str(last_checkpoint) if RESUME_IF_AVAILABLE and last_checkpoint.exists() else None
    print("Resume checkpoint:", resume_checkpoint)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()
    trainer.fit(model, ckpt_path=resume_checkpoint)
    training_seconds = time.perf_counter() - training_started
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Training time: {training_seconds / 3600:.2f} hours")
    print_vram("Training completed")

    # %% [markdown]
    # ## 12. Plot train loss / dev WER

    # %%
    metrics_files = sorted((RUN_DIR / "logs").glob("version_*/metrics.csv"))
    if not metrics_files:
        raise FileNotFoundError("Không thấy CSVLogger metrics.csv")
    metrics_path = metrics_files[-1]
    metrics = pd.read_csv(metrics_path)
    display(metrics.tail())

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    if "train_loss" in metrics:
        train_metrics = metrics.dropna(subset=["train_loss"])
        axes[0].plot(train_metrics["step"], train_metrics["train_loss"])
        axes[0].set(title="Training loss", xlabel="step", ylabel="loss")
    if "val_wer" in metrics:
        val_metrics = metrics.dropna(subset=["val_wer"])
        x_axis = val_metrics["epoch"] if "epoch" in val_metrics else val_metrics["step"]
        axes[1].plot(x_axis, val_metrics["val_wer"], marker="o")
        axes[1].set(title="Validation WER", xlabel="epoch", ylabel="WER")
    plt.tight_layout()
    plt.show()

    # %% [markdown]
    # ## 13. Load best checkpoint và export `.nemo`

    # %%
    def load_nemo_from_lightning_checkpoint(base_nemo_path, checkpoint_path):
        """Restore đúng architecture/tokenizer rồi nạp Lightning state_dict."""
        restored = ASRModel.restore_from(str(base_nemo_path), map_location="cpu")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = payload.get("state_dict", payload)
        missing, unexpected = restored.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint/model mismatch. missing={missing[:20]}, unexpected={unexpected[:20]}"
            )
        return restored, payload


    best_checkpoint = checkpoint_callback.best_model_path
    if not best_checkpoint:
        candidates = sorted(CHECKPOINT_DIR.glob("epoch=*.ckpt"))
        if not candidates:
            raise FileNotFoundError("Không thấy best checkpoint.")
        best_checkpoint = str(candidates[-1])
    print("Best checkpoint:", best_checkpoint)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    finetuned_model, best_payload = load_nemo_from_lightning_checkpoint(
        PRETRAINED_NEMO, best_checkpoint
    )
    BEST_NEMO = WORKING_DIR / "fastconformer_vietmed_best.nemo"
    finetuned_model.save_to(str(BEST_NEMO))
    best_epoch = int(best_payload.get("epoch", -1))
    print("Best epoch:", best_epoch)
    print("Exported:", BEST_NEMO)

    # %% [markdown]
    # ## 14. Dev evaluation sau fine-tuning

    # %%
    FINETUNED_PREDICTIONS = WORKING_DIR / "finetuned_predictions.csv"
    FINETUNED_SUMMARY = WORKING_DIR / "finetuned_summary.json"
    finetuned_summary, finetuned_predictions = evaluate_model(
        finetuned_model,
        manifests["dev"],
        FINETUNED_PREDICTIONS,
        FINETUNED_SUMMARY,
        limit=N_DEV_EVAL_SAMPLES,
        batch_size=VAL_BATCH_SIZE,
    )

    comparison = pd.DataFrame({
        "Metric": ["WER", "CER", "Exact match", "RTF"],
        "Baseline": [
            baseline_summary["wer"], baseline_summary["cer"],
            baseline_summary["exact_match"], baseline_summary["rtf"],
        ],
        "Fine-tuned": [
            finetuned_summary["wer"], finetuned_summary["cer"],
            finetuned_summary["exact_match"], finetuned_summary["rtf"],
        ],
    })
    comparison["Improvement"] = [
        comparison.loc[0, "Baseline"] - comparison.loc[0, "Fine-tuned"],
        comparison.loc[1, "Baseline"] - comparison.loc[1, "Fine-tuned"],
        comparison.loc[2, "Fine-tuned"] - comparison.loc[2, "Baseline"],
        comparison.loc[3, "Baseline"] - comparison.loc[3, "Fine-tuned"],
    ]
    display(comparison)

    # %% [markdown]
    # ## 15. Final test evaluation
    #
    # Chỉ chạy sau khi F1/hyperparameter đã cố định bằng dev. Test không tham gia early
    # stopping hay model selection.

    # %%
    print("Held-out evaluation is separate: python test_datasets.py --checkpoint <model.nemo> --datasets vietmed --samples 3437 --output-dir results/finetuned-test")

    # %% [markdown]
    # ## 16. Error analysis và medical-term metrics
    #
    # `medical_terms.txt` cần là danh sách do chuyên gia/nguồn ngoài xác định **trước khi**
    # xem transcript test. Mỗi dòng một cụm. Ba ví dụ mặc định chỉ để kiểm tra code.

    # %%
    HIGH_WER_PATH = WORKING_DIR / "high_wer_samples.csv"
    high_wer = (
        finetuned_predictions[
            ["audio_filepath", "duration", "reference", "prediction", "sample_wer"]
        ]
        .sort_values("sample_wer", ascending=False)
        .head(30)
    )
    high_wer.to_csv(HIGH_WER_PATH, index=False, encoding="utf-8-sig")
    display(high_wer)


    def load_medical_terms(path=None):
        if path:
            terms = Path(path).read_text(encoding="utf-8").splitlines()
        else:
            warnings.warn(
                "Đang dùng 3 cụm minh họa. Để đánh giá thật, cung cấp medical_terms.txt "
                "được tạo độc lập với dev/test references."
            )
            terms = [
                "chạy thận nhân tạo",
                "thuyên tắc huyết khối tĩnh mạch",
                "ký sinh trùng sốt rét",
            ]
        return sorted({normalize_vi_text(term) for term in terms if normalize_vi_text(term)})


    def count_nonoverlap(text, phrase):
        # Boundary whitespace phù hợp với normalized Vietnamese tokens.
        pattern = rf"(?<!\S){re.escape(phrase)}(?!\S)"
        return len(re.findall(pattern, text))


    def medical_term_metrics(prediction_frame, terms):
        reference_occurrences = 0
        correctly_recalled = 0
        false_positive_insertions = 0
        detail_rows = []
        for term in terms:
            ref_count = sum(count_nonoverlap(text, term) for text in prediction_frame["reference"])
            hyp_count = sum(count_nonoverlap(text, term) for text in prediction_frame["prediction"])
            matched = sum(
                min(count_nonoverlap(ref, term), count_nonoverlap(hyp, term))
                for ref, hyp in zip(prediction_frame["reference"], prediction_frame["prediction"])
            )
            false_positive = sum(
                max(0, count_nonoverlap(hyp, term) - count_nonoverlap(ref, term))
                for ref, hyp in zip(prediction_frame["reference"], prediction_frame["prediction"])
            )
            reference_occurrences += ref_count
            correctly_recalled += matched
            false_positive_insertions += false_positive
            detail_rows.append({
                "term": term,
                "reference_occurrences": ref_count,
                "correctly_recalled": matched,
                "exact_phrase_recall": matched / ref_count if ref_count else None,
                "hypothesis_occurrences": hyp_count,
                "false_positive_insertions": false_positive,
            })
        summary = {
            "terms": len(terms),
            "reference_occurrences": reference_occurrences,
            "medical_term_recall": (
                correctly_recalled / reference_occurrences if reference_occurrences else None
            ),
            "exact_medical_phrase_recall": (
                correctly_recalled / reference_occurrences if reference_occurrences else None
            ),
            "false_positive_medical_term_insertions": false_positive_insertions,
            "false_positive_rate_per_hypothesis": (
                false_positive_insertions / len(prediction_frame) if len(prediction_frame) else None
            ),
        }
        return summary, pd.DataFrame(detail_rows)


    medical_terms = load_medical_terms(MEDICAL_TERMS_PATH)
    medical_summary, medical_detail = medical_term_metrics(finetuned_predictions, medical_terms)
    display(pd.DataFrame([medical_summary]))
    display(medical_detail)
    medical_detail.to_csv(WORKING_DIR / "medical_term_details.csv", index=False, encoding="utf-8-sig")
    (WORKING_DIR / "medical_term_summary.json").write_text(
        json.dumps(medical_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # %% [markdown]
    # ## 17. Save/export artifacts và final summary
    #
    # Để resume ở session Kaggle sau: dùng **Save Version** và chọn lưu output, hoặc tạo
    # Kaggle Dataset/Model từ thư mục output. Ở notebook mới, attach output đó, đặt
    # `PRETRAINED_NEMO` tới file `.nemo` để continued fine-tune, hoặc copy `last.ckpt`
    # về `CHECKPOINT_DIR` và giữ đúng base `.nemo` + config rồi bật resume.

    # %%
    final_summary = {
        "model_id": MODEL_ID,
        "experiment": EXPERIMENT_NAME,
        "baseline_dev_wer": baseline_summary["wer"],
        "best_dev_wer": finetuned_summary["wer"],
        "best_dev_cer": finetuned_summary["cer"],
        "best_epoch": best_epoch,
        "best_checkpoint": best_checkpoint,
        "gpu_peak_vram_gb": peak_vram_gb,
        "training_seconds": training_seconds,
        "dev_inference_rtf": finetuned_summary["rtf"],
    }
    (WORKING_DIR / "final_summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    required_outputs = [
        WORKING_DIR / "fastconformer_vietmed_best.nemo",
        WORKING_DIR / "baseline_summary.json",
        WORKING_DIR / "finetuned_summary.json",
        WORKING_DIR / "baseline_predictions.csv",
        WORKING_DIR / "finetuned_predictions.csv",
        WORKING_DIR / "training_config.yaml",
        WORKING_DIR / "environment.txt",
    ]
    for path in required_outputs:
        if not path.exists():
            raise FileNotFoundError(f"Thiếu required output: {path}")

    print("\nFINAL SUMMARY")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    print("\nRequired outputs:")
    for path in required_outputs:
        print(f"[OK] {path} ({path.stat().st_size / 1024**2:.2f} MB)")

    # Bundle các artifact nhỏ và experiment logs/checkpoints; file .nemo vẫn giữ riêng.
    bundle_root = WORKING_DIR / "vietmed_run_bundle"
    bundle_root.mkdir(exist_ok=True)
    for path in required_outputs[1:] + [
        WORKING_DIR / "dataset_statistics.json",
        WORKING_DIR / "final_summary.json",
        WORKING_DIR / "high_wer_samples.csv",
        WORKING_DIR / "medical_term_summary.json",
        WORKING_DIR / "medical_term_details.csv",
    ]:
        if path.exists():
            shutil.copy2(path, bundle_root / path.name)
    shutil.copytree(RUN_DIR, bundle_root / "experiment", dirs_exist_ok=True)
    archive = shutil.make_archive(str(WORKING_DIR / "vietmed_run_bundle"), "zip", bundle_root)
    print("Bundle for download/save as Kaggle output:", archive)

    # %% [markdown]
    # ### Chạy F2–F4
    #
    # Sau khi F1 hoàn tất, đổi `EXPERIMENT_NAME` ở Configuration thành `F2`, `F3` hoặc
    # `F4`, rồi chạy lại. Mỗi experiment có thư mục checkpoint/log riêng. Chọn cấu hình
    # bằng **dev WER**; chỉ chạy final test một lần sau khi đã chốt cấu hình.
    #
    # ### Những thứ cố ý không có trong notebook đầu tiên
    #
    # Không đổi tokenizer, không LM/contextual biasing, không pseudo-label/self-training,
    # không lấy vocabulary/medical terms từ test. Đây là supervised continued fine-tuning
    # tối giản để phép so sánh trước/sau có thể kiểm chứng.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Install the notebook dependencies and execute the pipeline')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return
    run_training()


if __name__ == '__main__':
    main()
