"""Read-only bounded monitoring of the already authorized Kaggle benchmark."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from asr_common.paths import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--kernel", default="k63ftutrnththonguyn/vietmed-three-model-benchmark")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    api = KaggleApi()
    api.authenticate()
    destination = args.output or PROJECT_ROOT / "results" / "medical-kaggle" / "status.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    previous = None
    while True:
        response = api.kernels_status(args.kernel)
        status = str(response.status).split(".")[-1]
        event = {"status": status, "checked_utc": datetime.now(timezone.utc).isoformat(),
                 "monitor_seconds": round(time.monotonic() - started, 1),
                 "failure_message": response.failure_message}
        destination.write_text(json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8")
        if status != previous or event["monitor_seconds"] >= args.seconds:
            print(json.dumps(event, ensure_ascii=False), flush=True)
            previous = status
        if status in {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED"} or event["monitor_seconds"] >= args.seconds:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
