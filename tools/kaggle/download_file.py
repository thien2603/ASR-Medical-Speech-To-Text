from __future__ import annotations

import argparse
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import (
    ApiListKernelSessionOutputRequest,
    KaggleApi,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel")
    parser.add_argument("file_name")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    owner, slug = args.kernel.split("/", 1)
    api = KaggleApi()
    api.authenticate()
    client = api.build_kaggle_client()
    page_token = ""

    while True:
        request = ApiListKernelSessionOutputRequest()
        request.user_name = owner
        request.kernel_slug = slug
        if page_token:
            request.page_token = page_token
        response = client.kernels.kernels_api_client.list_kernel_session_output(
            request
        )
        for item in response.files:
            if item.file_name.replace("\\", "/") != args.file_name:
                continue
            args.destination.parent.mkdir(parents=True, exist_ok=True)
            partial = args.destination.with_suffix(args.destination.suffix + ".part")
            with requests.get(item.url, stream=True, timeout=120) as download:
                download.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in download.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            partial.replace(args.destination)
            print(f"Downloaded {item.file_name} -> {args.destination}")
            return
        page_token = response.next_page_token
        if not page_token:
            break
    raise FileNotFoundError(args.file_name)


if __name__ == "__main__":
    main()
