"""Standalone equivalent of the adjacent notebook; execution requires --run."""
import argparse


def run_decoding_diagnostic():
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
    # # VietMed FastConformer — baseline full-test và sửa RNNT decoding lặp
    #
    # Notebook này dùng output của lần fine-tune V5, không train lại:
    #
    # 1. đo model pretrained trên toàn bộ test;
    # 2. phát hiện output RNNT lặp bằng tín hiệu chỉ phụ thuộc audio + hypothesis;
    # 3. thử các giới hạn `max_symbols_per_step` và engine decoding an toàn;
    # 4. kiểm tra chất lượng trên một prefix dev không dùng để phát hiện lỗi test;
    # 5. re-decode chỉ các mẫu đáng ngờ và xuất model có decoding config an toàn.
    #
    # Vì test đã được xem ở V5, kết quả sửa lỗi là **diagnostic**, không phải test benchmark
    # hoàn toàn độc lập. Một test set mới/ẩn vẫn cần thiết trước khi công bố chất lượng.

    # %% [markdown]
    # ## 1. Cấu hình

    # %%
    from pathlib import Path

    SEED = 42
    WORKING_DIR = Path("/kaggle/working")
    INPUT_DIR = Path("/kaggle/input")

    PROBE_SAMPLES = 48
    DEV_GUARD_SAMPLES = 96
    TRANSCRIBE_BATCH_SIZE = 2

    # Detector không dùng reference nên có thể dùng khi inference thực tế.
    MAX_WORDS_PER_SECOND = 8.0
    MAX_CONSECUTIVE_WORD_RUN = 4
    MIN_ABNORMAL_WORD_COUNT = 40
    MAX_WORDS_PER_AUDIO_SECOND = 7.0

    # %% [markdown]
    # ## 2. Cài dependencies tương thích Kaggle Python 3.13

    # %%
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '-q', '--upgrade-strategy', 'only-if-needed', 'numpy==2.1.3', 'scipy==1.16.3', 'torchmetrics==1.9.0', 'pandas==2.3.3', 'protobuf==5.29.5', 'librosa==0.11.0', 'soundfile>=0.12.1', 'nemo_toolkit[asr]==2.7.3', 'jiwer==3.1.0'], check=True)

    # %%
    import copy
    import gc
    import json
    import random
    import re
    import time
    import unicodedata
    from itertools import groupby

    import jiwer
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    from nemo.collections.asr.models import ASRModel
    from omegaconf import OmegaConf, open_dict

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("Notebook cần Kaggle GPU.")
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)

    # %% [markdown]
    # ## 3. Tìm artifact V5 và remap manifest

    # %%
    def find_one(name):
        matches = sorted(INPUT_DIR.rglob(name))
        if len(matches) != 1:
            raise FileNotFoundError(f"Cần đúng 1 file {name}, tìm thấy: {matches[:10]}")
        return matches[0]


    FINETUNED_NEMO = find_one("fastconformer_vietmed_best.nemo")
    PRETRAINED_NEMO = find_one("vi-asr-fastconformer-114m-pretrained.nemo")
    TEST_MANIFEST = find_one("test_manifest.jsonl")
    DEV_MANIFEST = find_one("dev_manifest.jsonl")
    V5_TEST_PREDICTIONS = find_one("test_predictions.csv")
    V5_DEV_PREDICTIONS = find_one("finetuned_predictions.csv")
    V5_ROOT = FINETUNED_NEMO.parent

    print("V5 root:", V5_ROOT)
    print("Fine-tuned model:", FINETUNED_NEMO)
    print("Pretrained model:", PRETRAINED_NEMO)


    def remap_audio_path(old_path):
        parts = Path(old_path).parts
        if "vietmed_prepared" not in parts:
            raise ValueError(f"Đường dẫn manifest không có vietmed_prepared: {old_path}")
        start = parts.index("vietmed_prepared")
        candidate = V5_ROOT.joinpath(*parts[start:])
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return str(candidate)


    def read_manifest(path):
        rows = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                row["audio_filepath"] = remap_audio_path(row["audio_filepath"])
                rows.append(row)
        return rows


    test_rows = read_manifest(TEST_MANIFEST)
    dev_rows = read_manifest(DEV_MANIFEST)
    print("Rows:", {"dev": len(dev_rows), "test": len(test_rows)})
    assert len(test_rows) == 3437

    # %% [markdown]
    # ## 4. Metric và detector lặp dùng được khi production

    # %%
    def normalize_vi_text(text):
        text = unicodedata.normalize("NFC", str(text)).lower().strip()
        text = "".join(
            " " if unicodedata.category(char)[0] in {"P", "S"} else char
            for char in text
        )
        return re.sub(r"\s+", " ", text).strip()


    def hypothesis_to_text(item):
        if isinstance(item, str):
            return item
        if hasattr(item, "text"):
            return item.text
        return str(item)


    def max_word_run(text):
        words = str(text).split()
        return max((sum(1 for _ in run) for _, run in groupby(words)), default=0)


    def repetition_features(prediction, duration):
        words = str(prediction).split()
        word_count = len(words)
        duration = max(float(duration), 1e-6)
        words_per_second = word_count / duration
        longest_run = max_word_run(prediction)
        abnormal_length = word_count > max(
            MIN_ABNORMAL_WORD_COUNT, MAX_WORDS_PER_AUDIO_SECOND * duration
        )
        suspicious = (
            words_per_second > MAX_WORDS_PER_SECOND
            or longest_run >= MAX_CONSECUTIVE_WORD_RUN
            or abnormal_length
        )
        return {
            "word_count": word_count,
            "words_per_second": words_per_second,
            "max_word_run": longest_run,
            "abnormal_length": bool(abnormal_length),
            "suspicious_repetition": bool(suspicious),
        }


    def add_repetition_features(frame):
        feature_columns = [
            "word_count",
            "words_per_second",
            "max_word_run",
            "abnormal_length",
            "suspicious_repetition",
        ]
        base = frame.drop(columns=feature_columns, errors="ignore").reset_index(drop=True)
        feature_rows = [
            repetition_features(prediction, duration)
            for prediction, duration in zip(base["prediction"], base["duration"])
        ]
        return pd.concat(
            [base, pd.DataFrame(feature_rows)], axis=1
        )


    def safe_sample_wer(reference, hypothesis):
        if not reference:
            return 0.0 if not hypothesis else 1.0
        return float(jiwer.wer(reference, hypothesis))


    def summarize_predictions(frame, inference_seconds=None):
        references = frame["reference"].fillna("").astype(str).tolist()
        hypotheses = frame["prediction"].fillna("").astype(str).tolist()
        sample_wers = [
            safe_sample_wer(reference, hypothesis)
            for reference, hypothesis in zip(references, hypotheses)
        ]
        durations = frame["duration"].astype(float)
        enriched = add_repetition_features(frame)
        total_audio_seconds = float(durations.sum())
        return {
            "samples": len(frame),
            "wer": float(jiwer.wer(references, hypotheses)),
            "cer": float(jiwer.cer(references, hypotheses)),
            "exact_match": float(np.mean(np.asarray(references) == np.asarray(hypotheses))),
            "median_sample_wer": float(np.median(sample_wers)),
            "p90_sample_wer": float(np.quantile(sample_wers, 0.90)),
            "suspicious_repetition_count": int(enriched["suspicious_repetition"].sum()),
            "suspicious_repetition_rate": float(enriched["suspicious_repetition"].mean()),
            "median_words_per_second": float(enriched["words_per_second"].median()),
            "total_audio_seconds": total_audio_seconds,
            "inference_seconds": inference_seconds,
            "rtf": (
                float(inference_seconds / total_audio_seconds)
                if inference_seconds is not None and total_audio_seconds
                else None
            ),
        }


    def prediction_frame(rows, predictions):
        references = [normalize_vi_text(row["text"]) for row in rows]
        hypotheses = [normalize_vi_text(hypothesis_to_text(item)) for item in predictions]
        return pd.DataFrame({
            "audio_filepath": [row["audio_filepath"] for row in rows],
            "duration": [float(row["duration"]) for row in rows],
            "reference": references,
            "prediction": hypotheses,
            "sample_wer": [
                safe_sample_wer(reference, hypothesis)
                for reference, hypothesis in zip(references, hypotheses)
            ],
        })


    @torch.inference_mode()
    def transcribe_rows(model, rows, batch_size=TRANSCRIBE_BATCH_SIZE):
        model.eval().cuda()
        torch.cuda.synchronize()
        started = time.perf_counter()
        predictions = model.transcribe(
            [row["audio_filepath"] for row in rows],
            batch_size=batch_size,
            verbose=True,
            return_hypotheses=False,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return prediction_frame(rows, predictions), elapsed


    def write_json(path, value):
        Path(path).write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # %% [markdown]
    # ## 5. Xác nhận detector trên output V5

    # %%
    v5_test = pd.read_csv(V5_TEST_PREDICTIONS).fillna("")
    v5_test["reference"] = v5_test["reference"].map(normalize_vi_text)
    v5_test["prediction"] = v5_test["prediction"].map(normalize_vi_text)
    v5_test = add_repetition_features(v5_test)

    v5_default_summary = summarize_predictions(v5_test)
    write_json(WORKING_DIR / "v5_default_test_summary.json", v5_default_summary)
    print(json.dumps(v5_default_summary, ensure_ascii=False, indent=2))

    suspicious_positions = np.flatnonzero(
        v5_test["suspicious_repetition"].to_numpy(dtype=bool)
    ).tolist()
    print("Suspicious V5 rows:", len(suspicious_positions))
    print("First/last suspicious positions:", suspicious_positions[:10], suspicious_positions[-10:])
    assert len(suspicious_positions) > 0

    # %% [markdown]
    # ## 6. Baseline pretrained trên toàn bộ test

    # %%
    baseline_model = ASRModel.restore_from(str(PRETRAINED_NEMO), map_location="cpu")
    baseline_default_frame, baseline_default_seconds = transcribe_rows(
        baseline_model, test_rows
    )
    baseline_default_summary = summarize_predictions(
        baseline_default_frame, baseline_default_seconds
    )
    baseline_default_frame = add_repetition_features(baseline_default_frame)
    baseline_default_frame.to_csv(
        WORKING_DIR / "baseline_full_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_json(WORKING_DIR / "baseline_full_test_summary.json", baseline_default_summary)
    print(json.dumps(baseline_default_summary, ensure_ascii=False, indent=2))

    baseline_model.cpu()
    del baseline_model
    torch.cuda.empty_cache()
    gc.collect()

    # %% [markdown]
    # ## 7. So sánh các cấu hình decoding trên probe lỗi + dev guard
    #
    # `loop_labels=False` và tắt CUDA graph decoder giúp loại trừ lỗi state/graph.
    # `max_symbols_per_step` giới hạn số token RNNT được phát tại một acoustic frame.

    # %%
    def decoding_config(model, strategy, max_symbols_per_step, beam_size=4):
        cfg = OmegaConf.create(OmegaConf.to_container(model.cfg.decoding, resolve=True))
        with open_dict(cfg):
            cfg.strategy = strategy
            cfg.greedy.max_symbols_per_step = int(max_symbols_per_step)
            cfg.greedy.loop_labels = False
            cfg.greedy.use_cuda_graph_decoder = False
            cfg.beam.beam_size = int(beam_size)
            cfg.beam.max_symbols_per_step = int(max_symbols_per_step)
            cfg.beam.return_best_hypothesis = True
            cfg.beam.score_norm = True
            cfg.beam.allow_cuda_graphs = False
        return cfg


    CANDIDATES = {
        "greedy_m10_safe_engine": {
            "strategy": "greedy_batch", "max_symbols": 10, "batch_size": 2
        },
        "greedy_m5_safe_engine": {
            "strategy": "greedy_batch", "max_symbols": 5, "batch_size": 2
        },
        "greedy_m2_safe_engine": {
            "strategy": "greedy_batch", "max_symbols": 2, "batch_size": 2
        },
        "greedy_m1_safe_engine": {
            "strategy": "greedy_batch", "max_symbols": 1, "batch_size": 2
        },
    }

    probe_positions = suspicious_positions[:PROBE_SAMPLES]
    probe_rows = [test_rows[position] for position in probe_positions]
    dev_guard_rows = dev_rows[:DEV_GUARD_SAMPLES]

    v5_dev = pd.read_csv(V5_DEV_PREDICTIONS).fillna("").iloc[:DEV_GUARD_SAMPLES].copy()
    v5_dev["reference"] = v5_dev["reference"].map(normalize_vi_text)
    v5_dev["prediction"] = v5_dev["prediction"].map(normalize_vi_text)
    default_probe = v5_test.iloc[probe_positions].copy()

    candidate_results = [{
        "name": "v5_default",
        "strategy": "greedy_batch",
        "max_symbols": 10,
        "probe_wer": summarize_predictions(default_probe)["wer"],
        "probe_cer": summarize_predictions(default_probe)["cer"],
        "probe_suspicious": summarize_predictions(default_probe)["suspicious_repetition_count"],
        "dev_guard_wer": summarize_predictions(v5_dev)["wer"],
        "dev_guard_cer": summarize_predictions(v5_dev)["cer"],
        "seconds": 0.0,
    }]

    finetuned_model = ASRModel.restore_from(str(FINETUNED_NEMO), map_location="cpu")

    for name, candidate in CANDIDATES.items():
        print("\nCandidate:", name)
        cfg = decoding_config(
            finetuned_model,
            strategy=candidate["strategy"],
            max_symbols_per_step=candidate["max_symbols"],
        )
        finetuned_model.change_decoding_strategy(cfg, verbose=False)
        probe_frame, probe_seconds = transcribe_rows(
            finetuned_model, probe_rows, batch_size=candidate["batch_size"]
        )
        dev_frame, dev_seconds = transcribe_rows(
            finetuned_model, dev_guard_rows, batch_size=candidate["batch_size"]
        )
        probe_summary = summarize_predictions(probe_frame, probe_seconds)
        dev_summary = summarize_predictions(dev_frame, dev_seconds)
        candidate_results.append({
            "name": name,
            "strategy": candidate["strategy"],
            "max_symbols": candidate["max_symbols"],
            "probe_wer": probe_summary["wer"],
            "probe_cer": probe_summary["cer"],
            "probe_suspicious": probe_summary["suspicious_repetition_count"],
            "dev_guard_wer": dev_summary["wer"],
            "dev_guard_cer": dev_summary["cer"],
            "seconds": probe_seconds + dev_seconds,
        })
        probe_frame.to_csv(
            WORKING_DIR / f"probe_{name}.csv", index=False, encoding="utf-8-sig"
        )

    candidate_frame = pd.DataFrame(candidate_results)
    candidate_frame.to_csv(
        WORKING_DIR / "decoding_candidates.csv", index=False, encoding="utf-8-sig"
    )
    display(candidate_frame)

    default_dev_wer = float(
        candidate_frame.loc[candidate_frame["name"] == "v5_default", "dev_guard_wer"].iloc[0]
    )
    eligible = candidate_frame[
        (candidate_frame["name"] != "v5_default")
        & (candidate_frame["probe_suspicious"] == 0)
        & (candidate_frame["dev_guard_wer"] <= default_dev_wer + 0.03)
    ].copy()
    if eligible.empty:
        eligible = candidate_frame[candidate_frame["name"] != "v5_default"].copy()
        eligible["penalty"] = eligible["probe_suspicious"] * 100 + eligible["probe_wer"]
        selected_name = eligible.sort_values(
            ["penalty", "dev_guard_wer", "probe_wer"]
        ).iloc[0]["name"]
    else:
        selected_name = eligible.sort_values(
            ["dev_guard_wer", "probe_wer", "max_symbols"]
        ).iloc[0]["name"]

    selected = CANDIDATES[str(selected_name)]
    print("Selected decoding:", selected_name, selected)

    # %% [markdown]
    # ## 8. Re-decode thích ứng các mẫu V5 bị lặp

    # Chỉ những output bị detector đánh dấu mới chạy lại bằng cấu hình an toàn. Các output
    # bình thường giữ nguyên, nên tránh làm giảm chất lượng toàn bộ tập vì một cấu hình quá
    # bảo thủ.

    # %%
    selected_cfg = decoding_config(
        finetuned_model,
        strategy=selected["strategy"],
        max_symbols_per_step=selected["max_symbols"],
    )
    finetuned_model.change_decoding_strategy(selected_cfg, verbose=True)

    repair_rows = [test_rows[position] for position in suspicious_positions]
    repair_frame, repair_seconds = transcribe_rows(
        finetuned_model, repair_rows, batch_size=selected["batch_size"]
    )

    adaptive_frame = v5_test[
        ["audio_filepath", "duration", "reference", "prediction", "sample_wer"]
    ].copy()
    for local_index, original_position in enumerate(suspicious_positions):
        adaptive_frame.loc[original_position, "prediction"] = repair_frame.loc[
            local_index, "prediction"
        ]
    adaptive_frame["sample_wer"] = [
        safe_sample_wer(reference, hypothesis)
        for reference, hypothesis in zip(
            adaptive_frame["reference"], adaptive_frame["prediction"]
        )
    ]
    adaptive_frame = add_repetition_features(adaptive_frame)
    adaptive_summary = summarize_predictions(adaptive_frame, repair_seconds)
    adaptive_frame.to_csv(
        WORKING_DIR / "finetuned_adaptive_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_json(WORKING_DIR / "finetuned_adaptive_summary.json", adaptive_summary)
    print(json.dumps(adaptive_summary, ensure_ascii=False, indent=2))

    # %% [markdown]
    # ## 9. Nếu baseline cũng lặp, áp dụng cùng fallback

    # %%
    baseline_suspicious_positions = np.flatnonzero(
        baseline_default_frame["suspicious_repetition"].to_numpy(dtype=bool)
    ).tolist()
    baseline_adaptive_frame = baseline_default_frame.copy()
    baseline_repair_seconds = 0.0

    if baseline_suspicious_positions:
        baseline_model = ASRModel.restore_from(str(PRETRAINED_NEMO), map_location="cpu")
        baseline_model.change_decoding_strategy(selected_cfg, verbose=False)
        baseline_repair_rows = [test_rows[position] for position in baseline_suspicious_positions]
        baseline_repair_frame, baseline_repair_seconds = transcribe_rows(
            baseline_model,
            baseline_repair_rows,
            batch_size=selected["batch_size"],
        )
        for local_index, original_position in enumerate(baseline_suspicious_positions):
            baseline_adaptive_frame.loc[original_position, "prediction"] = (
                baseline_repair_frame.loc[local_index, "prediction"]
            )
        baseline_model.cpu()
        del baseline_model
        torch.cuda.empty_cache()
        gc.collect()

    baseline_adaptive_frame["sample_wer"] = [
        safe_sample_wer(reference, hypothesis)
        for reference, hypothesis in zip(
            baseline_adaptive_frame["reference"], baseline_adaptive_frame["prediction"]
        )
    ]
    baseline_adaptive_frame = add_repetition_features(
        baseline_adaptive_frame[
            ["audio_filepath", "duration", "reference", "prediction", "sample_wer"]
        ]
    )
    baseline_adaptive_summary = summarize_predictions(
        baseline_adaptive_frame, baseline_default_seconds + baseline_repair_seconds
    )
    baseline_adaptive_frame.to_csv(
        WORKING_DIR / "baseline_adaptive_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_json(WORKING_DIR / "baseline_adaptive_summary.json", baseline_adaptive_summary)

    # %% [markdown]
    # ## 10. Export model với decoding an toàn và báo cáo cuối

    # %%
    FIXED_NEMO = WORKING_DIR / "fastconformer_vietmed_decoding_fixed.nemo"
    finetuned_model.save_to(str(FIXED_NEMO))

    recommended_decoding = {
        "selected_candidate": str(selected_name),
        "strategy": selected["strategy"],
        "max_symbols_per_step": int(selected["max_symbols"]),
        "adaptive_detector": {
            "max_words_per_second": MAX_WORDS_PER_SECOND,
            "max_consecutive_word_run": MAX_CONSECUTIVE_WORD_RUN,
            "min_abnormal_word_count": MIN_ABNORMAL_WORD_COUNT,
            "max_words_per_audio_second": MAX_WORDS_PER_AUDIO_SECOND,
        },
    }
    (WORKING_DIR / "recommended_decoding.yaml").write_text(
        yaml.safe_dump(recommended_decoding, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    comparison = pd.DataFrame([
        {"system": "pretrained_default", **baseline_default_summary},
        {"system": "pretrained_adaptive", **baseline_adaptive_summary},
        {"system": "finetuned_v5_default", **v5_default_summary},
        {"system": "finetuned_adaptive", **adaptive_summary},
    ])
    comparison.to_csv(
        WORKING_DIR / "full_test_comparison.csv", index=False, encoding="utf-8-sig"
    )
    display(comparison)

    final_report = {
        "status": "complete",
        "diagnostic_note": (
            "Test V5 đã được quan sát; kết quả decoding fix là diagnostic, không phải "
            "đánh giá test độc lập."
        ),
        "selected_decoding": recommended_decoding,
        "candidate_results": candidate_results,
        "baseline_default": baseline_default_summary,
        "baseline_adaptive": baseline_adaptive_summary,
        "finetuned_v5_default": v5_default_summary,
        "finetuned_adaptive": adaptive_summary,
        "redecoded_finetuned_samples": len(suspicious_positions),
        "redecoded_baseline_samples": len(baseline_suspicious_positions),
        "fixed_model": str(FIXED_NEMO),
    }
    write_json(WORKING_DIR / "decoding_fix_report.json", final_report)

    print("FINAL DECODING FIX REPORT")
    print(json.dumps(final_report, ensure_ascii=False, indent=2))
    print("Fixed model:", FIXED_NEMO, f"({FIXED_NEMO.stat().st_size / 1024**2:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Install the notebook dependencies and execute the pipeline')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return
    run_decoding_diagnostic()


if __name__ == '__main__':
    main()
