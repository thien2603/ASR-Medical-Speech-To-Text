from __future__ import annotations

from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import (
    ApiListKernelSessionOutputRequest,
    KaggleApi,
)


OWNER = "k63ftutrnththonguyn"
SLUG = "vietmed-fastconformer-continued-fine-tune"
from asr_common.paths import PROJECT_ROOT

OUTPUT_DIR = PROJECT_ROOT / "kaggle_v5_report"
WANTED = {
    "baseline_summary.json",
    "dataset_statistics.json",
    "environment.txt",
    "final_summary.json",
    "finetuned_summary.json",
    "test_summary.json",
    "test_predictions.csv",
    "medical_term_summary.json",
    "high_wer_samples.csv",
    "training_config.yaml",
    "experiments/fastconformer_vietmed/F1/logs/version_0/metrics.csv",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    client = api.build_kaggle_client()

    page_token = ""
    found: set[str] = set()
    saved_log = False

    while True:
        request = ApiListKernelSessionOutputRequest()
        request.user_name = OWNER
        request.kernel_slug = SLUG
        if page_token:
            request.page_token = page_token

        response = client.kernels.kernels_api_client.list_kernel_session_output(
            request
        )

        if response.log and not saved_log:
            (OUTPUT_DIR / f"{SLUG}.log").write_text(
                response.log, encoding="utf-8"
            )
            saved_log = True

        for item in response.files:
            name = item.file_name.replace("\\", "/")
            if name not in WANTED or name in found:
                continue
            target = OUTPUT_DIR / name
            target.parent.mkdir(parents=True, exist_ok=True)
            download = requests.get(item.url, timeout=120)
            download.raise_for_status()
            target.write_bytes(download.content)
            found.add(name)
            print(f"Downloaded {name} ({len(download.content):,} bytes)")

        if found == WANTED:
            break
        page_token = response.next_page_token
        if not page_token:
            break

    missing = sorted(WANTED - found)
    if missing:
        raise RuntimeError(f"Missing expected output files: {missing}")
    print(f"Saved log: {saved_log}")


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
