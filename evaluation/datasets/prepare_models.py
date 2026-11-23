"""Pin and convert the two original Whisper checkpoints for a 4 GB GPU.

Run with the project's existing Python environment. Converted checkpoints use
INT8 weights/FP16 activations; no training or benchmark labels are involved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from asr_common.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
RUNTIME = ROOT / ".benchmark_runtime"
MODELS = {
    "phowhisper": ("vinai/PhoWhisper-large", "b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb", "pytorch_model.bin"),
    "whisper": ("openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1", "model.safetensors"),
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_weight(repo: str, revision: str, filename: str, source: Path) -> None:
    """Resume verified ranged downloads; full multi-GB responses stall here."""
    import requests
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    url = hf_hub_url(repo, filename, revision=revision)
    remote = get_hf_file_metadata(url, timeout=30)
    destination = source / filename
    if destination.exists() and destination.stat().st_size == remote.size:
        if file_hash(destination) == remote.etag:
            return
        raise RuntimeError("Existing downloaded weight fails the upstream SHA-256 check")
    partial = source / (filename + ".part")
    state_file = source / (filename + ".download.json")
    chunk_bytes = 8 * 1024 * 1024
    chunks = (remote.size + chunk_bytes - 1) // chunk_bytes
    done = set()
    if state_file.exists() and partial.exists():
        state = json.loads(state_file.read_text())
        if state["etag"] != remote.etag or state["size"] != remote.size:
            raise RuntimeError("Partial download belongs to another upstream checkpoint")
        done = set(state["completed"])
    source.mkdir(parents=True, exist_ok=True)
    if not partial.exists():
        with partial.open("wb") as handle:
            handle.truncate(remote.size)

    def fetch(index):
        start = index * chunk_bytes
        end = min(remote.size, start + chunk_bytes) - 1
        for attempt in range(5):
            try:
                # Distinct query prevents a proxy reusing a redirect signed for
                # another Range. Validate Content-Range before accepting bytes.
                with requests.get(url + f"?download=true&chunk={index}&attempt={attempt}",
                                  headers={"Range": f"bytes={start}-{end}"},
                                  timeout=(20, 45), stream=True) as response:
                    response.raise_for_status()
                    expected_range = f"bytes {start}-{end}/{remote.size}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                        raise RuntimeError(f"Unexpected Range response: {response.status_code}: {response.headers.get('Content-Range')}")
                    content = response.content
                if len(content) != end - start + 1:
                    raise RuntimeError("Incomplete range response")
                return index, content
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 8))

    pending = (index for index in range(chunks) if index not in done)
    # A bounded window keeps completed response buffers from occupying RAM while
    # the Windows disk is busy writing the checkpoint.
    with partial.open("r+b") as handle, ThreadPoolExecutor(max_workers=8) as pool:
        futures = {}
        for _ in range(8):
            index = next(pending, None)
            if index is not None:
                futures[pool.submit(fetch, index)] = index
        while futures:
            future = next(as_completed(futures))
            del futures[future]
            index, content = future.result()
            handle.seek(index * chunk_bytes)
            handle.write(content)
            done.add(index)
            del content
            handle.flush()
            state_file.write_text(json.dumps({"etag": remote.etag, "size": remote.size, "completed": sorted(done)}))
            if len(done) % 16 == 0 or len(done) == chunks:
                print(f"{repo}: {len(done)}/{chunks} chunks ({100*len(done)/chunks:.1f}%)", flush=True)
            index = next(pending, None)
            if index is not None:
                futures[pool.submit(fetch, index)] = index
    if file_hash(partial) != remote.etag:
        raise RuntimeError("Downloaded checkpoint SHA-256 differs from the author checkpoint")
    partial.replace(destination)
    state_file.unlink(missing_ok=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    repo, revision, weight_file = MODELS[args.model]
    source = RUNTIME / "sources" / args.model
    target = RUNTIME / "models" / args.model
    provenance_file = target / "provenance.json"
    if provenance_file.exists():
        provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
        if provenance["source_revision"] != revision or not (target / "model.bin").exists():
            raise RuntimeError("Existing conversion does not match the pinned source")
        print(f"Already converted: {target}", flush=True)
        return

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo, revision=revision, local_dir=str(source), max_workers=2,
        allow_patterns=["*.json", "merges.txt", "vocab.json"],
    )
    download_weight(repo, revision, weight_file, source)
    print(f"Downloaded {repo}@{revision}", flush=True)
    if args.download_only:
        return

    # Match the import order required by this Windows/PyTorch installation.
    import scipy.signal  # noqa: F401
    import ctranslate2
    from ctranslate2.converters import TransformersConverter
    source_weight = source / weight_file
    source_hash = file_hash(source_weight)
    print("Converting to int8_float16 (low-memory FP16 loading)...", flush=True)
    converter = TransformersConverter(
        str(source), load_as_float16=True, low_cpu_mem_usage=True,
        copy_files=["preprocessor_config.json"],
    )
    converter.convert(str(target), quantization="int8_float16")
    provenance = {
        "source_model": repo, "source_revision": revision,
        "source_weight_file": weight_file, "source_weight_sha256": source_hash,
        "converted_weight_sha256": file_hash(target / "model.bin"),
        "ctranslate2_version": ctranslate2.__version__,
        "quantization": "int8_float16", "training": False,
    }
    provenance_file.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    # Only remove this script's downloaded intermediate weight, after a verified
    # conversion. The user's pre-existing caches/checkpoints are never changed.
    source_weight.unlink()
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
