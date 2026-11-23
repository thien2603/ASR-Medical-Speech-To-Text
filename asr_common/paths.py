"""Workspace locations shared by local commands, independent of the CWD."""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
VIDEO_DIR = PROJECT_ROOT / "video_eval"
PODCAST_INPUT_DIR = VIDEO_DIR / "kaggle_input"
EXTERNAL_RUNTIME = Path(os.environ.get("ASR_RUNTIME_DIR", r"E:\SpeechToText_Medical_runtime"))
